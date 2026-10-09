# -*- coding: utf-8 -*-
"""训练循环 / 验证 / 嵌入 / 提取（docs/04-network-spec.md §5~7）。"""
import json
import os
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import watermark_coding as wc
from data_loader import get_loader
from model import Decoder, Discriminator, Encoder_Decoder
from noise_layer import RandomCrop


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(device, *tensors):
    return [t.to(device) for t in tensors]


def _minmax01(t):
    """逐样本 min-max 归一化到 [0,1]（PIMoG 掩膜公式）。"""
    tmin = t.amin(dim=(1, 2, 3), keepdim=True)
    tmax = t.amax(dim=(1, 2, 3), keepdim=True)
    return (t - tmin) / (tmax - tmin).clamp_min(1e-8)


class Solver:
    """训练 / 嵌入 / 提取的统一入口（train 时要求 train+val loader）。"""

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion_MSE = nn.MSELoss()
        self.criterion_BCE = nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------ 模型构建
    def build_train(self):
        cfg = self.config
        self.net = Encoder_Decoder(cfg.distortion, msg_len=wc.L).to(self.device)
        self.net_D = Discriminator(64).to(self.device)
        self.optimizer_D = optim.Adam(self.net_D.parameters(), lr=cfg.lr)
        self.optimizer = optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.crop_layer = RandomCrop(cfg.crop_min_area) if cfg.crop else None
        self.start_epoch = 0
        self.best_acc = -1.0
        if cfg.resume:
            ckpt = torch.load(cfg.resume, map_location=self.device, weights_only=False)
            self.net.Encoder.load_state_dict(ckpt['encoder'])
            self.net.Decoder.load_state_dict(ckpt['decoder'])
            self.net_D.load_state_dict(ckpt['discriminator'])
            if ckpt.get('optimizer') is not None:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.optimizer_D.load_state_dict(ckpt['optimizer_D'])
            self.start_epoch = int(ckpt.get('epoch', -1)) + 1
            self.best_acc = float(ckpt.get('best_acc', -1.0))
            hist = f'{self.best_acc * 100:.2f}%' if self.best_acc >= 0 else '无'
            print(f'[train] 断点续训：加载 {cfg.resume}，从 epoch {self.start_epoch} '
                  f'继续，历史 best_msg_acc={hist}')

    def build_embed(self):
        self.encoder = Encoder_Decoder(self.config.distortion,
                                       msg_len=wc.L).Encoder.to(self.device)
        ckpt = torch.load(self.config.model_path, map_location=self.device, weights_only=False)
        if 'encoder' in ckpt:
            self.encoder.load_state_dict(ckpt['encoder'])
        else:
            self.encoder.load_state_dict(
                {k.replace('Encoder.', ''): v for k, v in ckpt.items()
                 if k.startswith('Encoder.')})
        self.encoder.eval()

    def build_extract(self):
        self.decoder = Decoder(msg_len=wc.L).to(self.device)
        ckpt = torch.load(self.config.model_path, map_location=self.device, weights_only=False)
        if 'decoder' in ckpt:
            self.decoder.load_state_dict(ckpt['decoder'])
        else:
            self.decoder.load_state_dict(
                {k.replace('Decoder.', ''): v for k, v in ckpt.items()
                 if k.startswith('Decoder.')})
        self.decoder.eval()

    # ------------------------------------------------------------ 训练
    def train(self, train_loader, val_loader):
        cfg = self.config
        seed_all(cfg.seed)
        self.build_train()
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(cfg.log_dir, exist_ok=True)
        os.makedirs(cfg.result_dir, exist_ok=True)
        tag = f'train_{cfg.distortion}' + ('_crop' if cfg.crop else '')
        # 续训时追加日志，避免覆盖历史记录
        log_mode = 'a' if self.start_epoch > 0 else 'w'
        txtfile = open(os.path.join(cfg.log_dir, tag + '.txt'), log_mode, encoding='utf-8')
        print(f'[train] device={self.device} distortion={cfg.distortion} '
              f'crop={cfg.crop} epochs={cfg.num_epoch} start_epoch={self.start_epoch} '
              f'train_imgs={len(train_loader.dataset)} val_imgs={len(val_loader.dataset)}')
        best_acc = self.best_acc
        start_time = time.time()

        for epoch in range(self.start_epoch, cfg.num_epoch):
            running = dict(msg=0.0, img=0.0, gan=0.0, stat=0.0)
            self.net.train()
            for i, batch in enumerate(train_loader):
                inputs, seq, _msg, _idx = batch
                inputs, seq = _to_device(self.device, inputs.float(), seq.float())
                inputs.requires_grad_(True)  # 梯度掩膜需要（PIMoG 同款）

                # ---- 前向：编码 → 噪声 → [裁剪] → 解码 ----
                encoded = self.net.Encoder(inputs, seq)
                noised = self.net.Noiser(encoded)
                if self.crop_layer is not None:
                    noised = self.crop_layer(noised)
                decoded = self.net.Decoder(noised.float())

                # ---- 序列损失（MSE，±1 目标）----
                loss_msg = self.criterion_MSE(decoded, seq)

                # ---- 梯度掩膜图像损失（PIMoG 式）----
                # retain_graph=True：后续 loss.backward() 还要遍历同一前向图
                # （PIMoG 用 create_graph=True 隐式达到同样效果，但那会构建二阶图、
                # 显存翻倍；掩膜已 detach 不参与梯度，二阶图纯属浪费——实测 256×256
                # batch4 下 4GB 显存 create_graph=False+retain_graph 可训练）
                inputgrad = torch.autograd.grad(loss_msg, inputs,
                                                create_graph=False, retain_graph=True)[0]
                mask = 1.0 - _minmax01(inputgrad) + 1.0  # 值域 [1,2]
                loss_img = self.criterion_MSE(encoded * mask.detach(),
                                              inputs * mask.detach())

                # ---- GAN：先判别器后生成侧 ----
                d_label_host = torch.full((inputs.shape[0], 1), 1.0, device=self.device)
                d_label_decoded = torch.full((inputs.shape[0], 1), 0.0, device=self.device)
                self.optimizer_D.zero_grad()
                d_loss_host = self.criterion_BCE(self.net_D(inputs.detach()), d_label_host)
                d_loss_host.backward()
                d_loss = self.criterion_BCE(self.net_D(encoded.detach()), d_label_decoded)
                d_loss.backward()
                self.optimizer_D.step()

                g_loss = self.criterion_BCE(self.net_D(encoded), d_label_host)

                # ---- 软统计特征损失（引导提取序列统计量逼近目标）----
                targets = torch.tensor(
                    np.stack([wc.message_to_target_features(m) for m in _msg]),
                    dtype=torch.float32, device=self.device)
                soft = wc.soft_statistics(decoded, tau=cfg.tau)
                loss_stat = self.criterion_MSE(soft, targets)

                loss = (cfg.lambda1 * loss_msg + cfg.lambda2 * loss_img
                        + cfg.lambda3 * g_loss + cfg.lambda4 * loss_stat)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running['msg'] += loss_msg.item()
                running['img'] += loss_img.item()
                running['gan'] += g_loss.item()
                running['stat'] += loss_stat.item()

                if (i + 1) % cfg.log_step == 0:
                    n = cfg.log_step
                    line = (f'[epoch {epoch + 1}/{cfg.num_epoch}, iter {i + 1}] '
                            f'loss={sum(running[k] for k in running) / n:.4f} '
                            f'msg={running["msg"] / n:.4f} img={running["img"] / n:.4f} '
                            f'gan={running["gan"] / n:.4f} stat={running["stat"] / n:.4f}')
                    print(line)
                    print(line, file=txtfile, flush=True)
                    running = dict(msg=0.0, img=0.0, gan=0.0, stat=0.0)

            # ---- 验证 ----
            msg_acc, bit_acc, ber, psnr = self.validate(val_loader)
            line = (f'[epoch {epoch + 1}] val msg_acc={msg_acc * 100:.2f}% '
                    f'bit_acc={bit_acc * 100:.2f}% seq_ber={ber * 100:.2f}% '
                    f'psnr={psnr:.2f}dB ({time.time() - start_time:.0f}s)')
            print(line)
            print(line, file=txtfile, flush=True)

            # ---- 保存（先更新 best_acc，让 checkpoint 携带最新历史最优）----
            save_best = msg_acc >= best_acc
            if save_best:
                best_acc = msg_acc
            ckpt = {'encoder': self.net.Encoder.state_dict(),
                    'decoder': self.net.Decoder.state_dict(),
                    'discriminator': self.net_D.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'optimizer_D': self.optimizer_D.state_dict(),
                    'epoch': epoch,
                    'best_acc': best_acc}
            if (epoch + 1) % cfg.model_save_step == 0:
                torch.save(ckpt, os.path.join(cfg.checkpoint_dir,
                                              f'checkpoint_epoch_{epoch + 1}.pth'))
            if save_best:
                torch.save(ckpt, os.path.join(cfg.checkpoint_dir, 'best.pth'))
        txtfile.close()
        print(f'[train] done, best val msg_acc={best_acc * 100:.2f}%')

    def validate(self, val_loader):
        """验证：消息准确率（整串一致）、位准确率、序列 BER、PSNR。

        PSNR 按 [−1,1] 值域（峰值=2）计算：10·log10(4/MSE)。
        """
        self.net.eval()
        total, correct, bits_ok, bits_total, ber_sum, ber_cnt = 0, 0, 0, 0, 0.0, 0
        mse_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, seq, msgs, _idx = batch
                inputs, seq = _to_device(self.device, inputs.float(), seq.float())
                encoded = self.net.Encoder(inputs, seq)
                mse_sum += float(((encoded - inputs) ** 2).mean()) * inputs.shape[0]
                noised = self.net.Noiser(encoded)
                if self.crop_layer is not None:
                    noised = self.crop_layer(noised)
                decoded = self.net.Decoder(noised.float())
                hard = torch.sign(decoded)
                hard[hard == 0] = 1.0
                ber_sum += float((hard != seq).float().mean()) * inputs.shape[0]
                ber_cnt += inputs.shape[0]
                for b in range(inputs.shape[0]):
                    out_msg = wc.sequence_to_message(hard[b].cpu().numpy())
                    gt = msgs[b]
                    total += 1
                    correct += int(out_msg == gt)
                    bits_ok += sum(a == b for a, b in zip(out_msg, gt))
                    bits_total += wc.MSG_LEN
        self.net.train()
        n = max(ber_cnt, 1)
        psnr = 10.0 * np.log10(4.0 / max(mse_sum / n, 1e-10))
        return (correct / max(total, 1), bits_ok / max(bits_total, 1),
                ber_sum / n, float(psnr))

    # ------------------------------------------------------------ 嵌入
    def embed(self, loader):
        cfg = self.config
        seed_all(cfg.seed)
        self.build_embed()
        out_dir = os.path.join(cfg.result_dir, 'embed')
        os.makedirs(out_dir, exist_ok=True)
        gt = {}
        psnrs = []
        with torch.no_grad():
            for batch in loader:
                inputs, seq, msgs, fnames = batch
                inputs, seq = _to_device(self.device, inputs.float(), seq.float())
                encoded = self.encoder(inputs, seq)
                for b in range(inputs.shape[0]):
                    iw = (encoded[b].clamp(-1, 1).cpu().numpy() + 1.0) / 2.0 * 255.0
                    iw = np.clip(np.round(iw), 0, 255).astype(np.uint8).transpose(1, 2, 0)
                    iw = cv2.cvtColor(iw, cv2.COLOR_RGB2BGR)
                    save_path = os.path.join(out_dir, fnames[b])
                    cv2.imwrite(save_path, iw)
                    orig = (inputs[b].cpu().numpy() + 1.0) / 2.0 * 255.0
                    mse = float(np.mean((iw.astype(np.float32)[..., ::-1] - orig.transpose(1, 2, 0)) ** 2))
                    psnr = 10 * np.log10(255.0 ** 2 / max(mse, 1e-10))
                    psnrs.append(psnr)
                    gt[fnames[b]] = msgs[b]
                    print(f'[embed] {fnames[b]} msg={msgs[b]} psnr={psnr:.2f}dB')
        gt_path = os.path.join(out_dir, 'gt.json')
        with open(gt_path, 'w', encoding='utf-8') as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)
        print(f'[embed] done: {len(gt)} images -> {out_dir} '
              f'(mean psnr={np.mean(psnrs):.2f}dB, gt: {gt_path})')
        return gt_path

    # ------------------------------------------------------------ 提取
    def extract(self, loader):
        cfg = self.config
        seed_all(cfg.seed)
        self.build_extract()
        gt = None
        if cfg.watermark_file and os.path.isfile(cfg.watermark_file):
            with open(cfg.watermark_file, 'r', encoding='utf-8') as f:
                gt = json.load(f)
        results = []
        with torch.no_grad():
            for batch in loader:
                inputs, fnames = batch
                inputs = inputs.float().to(self.device)
                decoded = self.decoder(inputs)
                hard = torch.sign(decoded)
                hard[hard == 0] = 1.0
                for b in range(inputs.shape[0]):
                    h = hard[b].cpu().numpy()
                    msg = wc.sequence_to_message(h)
                    feats = wc.features_from_sequence(h)
                    levels = wc.levels_from_features(feats)
                    rec = {'file': fnames[b], 'message': msg,
                           'levels': [int(x) for x in levels],
                           'features': {k: round(float(v), 4) for k, v in feats.items()}}
                    if gt and fnames[b] in gt:
                        g = gt[fnames[b]]
                        seq_gt = wc.message_to_sequence(g)
                        rec['gt'] = g
                        rec['msg_match'] = bool(msg == g)
                        rec['bit_acc'] = sum(a == b for a, b in zip(msg, g)) / wc.MSG_LEN
                        rec['seq_ber'] = float(np.mean((h != seq_gt)))
                    results.append(rec)
                    line = (f"[extract] {fnames[b]} -> {msg}"
                            + (f" (gt {g}) {'OK' if rec.get('msg_match') else 'MISMATCH'}"
                               if 'gt' in rec else ''))
                    print(line)
        if gt:
            n = len([r for r in results if 'gt' in r])
            if n:
                acc = np.mean([r['msg_match'] for r in results if 'gt' in r])
                bit = np.mean([r['bit_acc'] for r in results if 'gt' in r])
                ber = np.mean([r['seq_ber'] for r in results if 'gt' in r])
                print(f'[extract] summary over {n} images: msg_acc={acc * 100:.2f}% '
                      f'bit_acc={bit * 100:.2f}% seq_ber={ber * 100:.2f}%')
        os.makedirs(cfg.result_dir, exist_ok=True)
        rep = os.path.join(cfg.result_dir, 'extract_report.json')
        with open(rep, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f'[extract] report: {rep}')
        return results
