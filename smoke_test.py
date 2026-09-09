# -*- coding: utf-8 -*-
"""冒烟测试（docs/06-evaluation-spec.md §4）：不依赖外部数据集的端到端管线验证。

阶段 S1~S6，全部通过打印 `SMOKE TEST PASSED` 并返回 True。
"""
import os
import re
import shutil
import unittest

import cv2
import numpy as np
import torch

import watermark_coding as wc
from config import Config
from data_loader import get_loader
from model import Encoder_Decoder
from noise_layer import RandomCrop, ScreenShooting
from solver import Solver

SMOKE_DIR = 'smoke_data'


# ---------------------------------------------------------------- S1~S3
def run_coding_tests():
    from tests import test_coding
    suite = unittest.TestSuite()
    for name in ['test_roundtrip_random_messages', 'test_grid_exactness',
                 'test_tolerance_to_flips', 'test_gray_adjacency']:
        suite.addTest(test_coding.TestCoding(name))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return result.wasSuccessful()


# ---------------------------------------------------------------- 合成数据
def gen_synthetic(n, size, seed):
    rng = np.random.default_rng(seed)
    imgs = []
    for _ in range(n):
        img = np.zeros((size, size, 3), np.float32)
        gx = np.linspace(0, 1, size, dtype=np.float32)[None, :]
        gy = np.linspace(0, 1, size, dtype=np.float32)[:, None]
        for c in range(3):
            img[..., c] = (0.25 + 0.55 * (rng.random() * gx + rng.random() * gy)
                           + 0.2 * rng.random())
        for _ in range(int(rng.integers(3, 8))):
            y, x = int(rng.integers(0, size)), int(rng.integers(0, size))
            rad = float(rng.integers(8, 30))
            yy, xx = np.mgrid[0:size, 0:size]
            d = np.sqrt((yy - y) ** 2 + (xx - x) ** 2).astype(np.float32)
            blob = np.clip(1.0 - d / rad, 0.0, 1.0)
            color = rng.random(3).astype(np.float32)
            for c in range(3):
                img[..., c] = img[..., c] * (1.0 - 0.6 * blob) + 0.6 * blob * color[c]
        noise = cv2.GaussianBlur(
            rng.normal(0, 0.05, (size, size, 3)).astype(np.float32), (7, 7), 0)
        img = np.clip(img + noise, 0.0, 1.0)
        imgs.append(img)
    return imgs


def make_smoke_data(size=128, n=32, seed=0):
    img_dir = os.path.join(SMOKE_DIR, 'images')
    shutil.rmtree(img_dir, ignore_errors=True)
    os.makedirs(img_dir, exist_ok=True)
    for i, img in enumerate(gen_synthetic(n, size, seed)):
        cv2.imwrite(os.path.join(img_dir, f'{i:03d}.png'),
                    (np.clip(img, 0, 1)[..., ::-1] * 255).astype(np.uint8))
    return img_dir


# ---------------------------------------------------------------- 主流程
def run_smoke(cfg: Config) -> bool:
    print('=' * 60)
    print('SMOKE TEST (docs/06-evaluation-spec.md §4)')
    print('=' * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device={device}')
    ok = True

    # S1~S3 编码单测
    print('\n[S1~S3] 编码核心单测（往返/网格/容错/格雷码）')
    if not run_coding_tests():
        print('[S1~S3] FAILED')
        return False

    # 合成数据
    img_dir = make_smoke_data(cfg.image_size, 32, cfg.seed)
    print(f'\n合成数据生成: {img_dir} (32 张 {cfg.image_size}x{cfg.image_size})')

    # S4 前向形状
    print('\n[S4] 前向形状检查')
    try:
        net = Encoder_Decoder('ScreenShooting').to(device)
        x = torch.rand(2, 3, cfg.image_size, cfg.image_size, device=device)
        m = torch.randint(0, 2, (2, wc.L), device=device).float() * 2 - 1
        enc, noised, dec = net(x, m)
        assert enc.shape == (2, 3, cfg.image_size, cfg.image_size), enc.shape
        assert dec.shape == (2, wc.L), dec.shape
        crop_out = RandomCrop(cfg.crop_min_area)(noised)
        assert crop_out.shape == noised.shape
        print(f'[S4] PASS: enc={tuple(enc.shape)} dec={tuple(dec.shape)} '
              f'crop={tuple(crop_out.shape)}')
    except Exception as e:
        print(f'[S4] FAILED: {e!r}')
        return False

    # S5 短训练
    print('\n[S5] 短训练（合成数据、全损失）')
    smoke_cfg = Config(
        dataset_dir=SMOKE_DIR, image_size=cfg.image_size, batch_size=8,
        num_epoch=max(1, (cfg.smoke_iters + 3) // 4), log_step=4, model_save_step=1,
        eval_every=1, distortion='ScreenShooting', crop=False,
        checkpoint_dir=os.path.join(SMOKE_DIR, 'ckpt'),
        log_dir=os.path.join(SMOKE_DIR, 'logs'),
        result_dir=os.path.join(SMOKE_DIR, 'results'),
        num_workers=0, seed=cfg.seed)
    try:
        train_loader = get_loader(img_dir, cfg.image_size, smoke_cfg.batch_size,
                                  'train', 0, seed=cfg.seed, shuffle=True,
                                  drop_last=True)
        val_loader = get_loader(img_dir, cfg.image_size, smoke_cfg.batch_size,
                                'train', 0, seed=cfg.seed, shuffle=False,
                                drop_last=False)
        solver = Solver(smoke_cfg)
        solver.train(train_loader, val_loader)
        log_path = os.path.join(smoke_cfg.log_dir, 'train_ScreenShooting.txt')
        losses = []
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                mnum = re.search(r'loss=(-?[\d.]+)', line)
                if mnum:
                    losses.append(float(mnum.group(1)))
        assert losses, '训练日志中未找到 loss 记录'
        assert all(np.isfinite(v) for v in losses), f'损失含非法值: {losses}'
        assert losses[-1] <= max(losses[0] * 10, 1e-3), \
            f'损失疑似爆炸: {losses[0]:.3f} -> {losses[-1]:.3f}'
        print(f'[S5] PASS: 损失序列 {[f"{v:.2f}" for v in losses]}')
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'[S5] FAILED: {e!r}')
        return False

    # S6 端到端：embed 2 图 → 屏摄噪声 → extract
    print('\n[S6] 端到端（embed → ScreenShooting → extract）')
    try:
        embed_in = os.path.join(SMOKE_DIR, 'embed_in')
        shutil.rmtree(embed_in, ignore_errors=True)
        os.makedirs(embed_in, exist_ok=True)
        for f in sorted(os.listdir(img_dir))[:2]:
            shutil.copy(os.path.join(img_dir, f), os.path.join(embed_in, f))
        msg = '010101010101010101'
        e_cfg = Config(model_path=os.path.join(smoke_cfg.checkpoint_dir, 'best.pth'),
                       image_dir=embed_in, image_size=cfg.image_size,
                       batch_size=2, result_dir=os.path.join(SMOKE_DIR, 'results'),
                       message=msg, seed=cfg.seed)
        e_loader = get_loader(embed_in, cfg.image_size, 2, 'embed', 0,
                              message=msg, seed=cfg.seed, shuffle=False,
                              drop_last=False)
        gt_path = Solver(e_cfg).embed(e_loader)
        # 施加屏摄噪声
        noised_dir = os.path.join(SMOKE_DIR, 'noised')
        shutil.rmtree(noised_dir, ignore_errors=True)
        os.makedirs(noised_dir, exist_ok=True)
        embed_out = os.path.join(e_cfg.result_dir, 'embed')
        noiser = ScreenShooting().to(device)
        for f in sorted(os.listdir(embed_out)):
            if not f.lower().endswith(('.png', '.jpg', '.bmp')):
                continue
            img = cv2.imread(os.path.join(embed_out, f))
            rgb = np.ascontiguousarray(img[..., ::-1]).transpose(2, 0, 1)
            t = torch.from_numpy(rgb).float().to(device)
            t = t / 255.0 * 2.0 - 1.0
            t = noiser(t.unsqueeze(0))[0].clamp(-1, 1)
            t = (t + 1.0) / 2.0 * 255.0
            out = t.permute(1, 2, 0).cpu().numpy()[..., ::-1]
            cv2.imwrite(os.path.join(noised_dir, f), np.clip(out, 0, 255).astype(np.uint8))
        x_cfg = Config(model_path=os.path.join(smoke_cfg.checkpoint_dir, 'best.pth'),
                       image_dir=noised_dir, image_size=cfg.image_size,
                       batch_size=2, result_dir=os.path.join(SMOKE_DIR, 'results'),
                       watermark_file=gt_path, seed=cfg.seed)
        x_loader = get_loader(noised_dir, cfg.image_size, 2, 'extract', 0,
                              shuffle=False, drop_last=False)
        results = Solver(x_cfg).extract(x_loader)
        assert len(results) == 2, f'提取结果数 {len(results)} != 2'
        assert all(len(r['message']) == wc.MSG_LEN for r in results)
        print(f'[S6] PASS: 提取 {len(results)} 条 18bit 消息（未训练模型，准确率仅记录）')
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'[S6] FAILED: {e!r}')
        return False

    print('\n' + '=' * 60)
    print('SMOKE TEST PASSED')
    print('=' * 60)
    return True


if __name__ == '__main__':
    import sys
    sys.exit(0 if run_smoke(Config(smoke_iters=5)) else 1)
