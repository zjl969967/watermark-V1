# -*- coding: utf-8 -*-
"""配置规范实现（docs/03-architecture.md §3）。

单一 dataclass `Config` + argparse 构建器；CLI 四个子命令：
train / embed / extract / smoke。
"""
import argparse
from dataclasses import dataclass, field, asdict


@dataclass
class Config:
    # ---- 数据 ----
    dataset_dir: str = 'dataset'        # 训练数据根目录（其下 train/ val/）
    image_dir: str = ''                 # embed/extract 的图像目录
    image_size: int = 128
    batch_size: int = 16
    num_workers: int = 0                # Windows 下默认 0

    # ---- 模型 / 训练 ----
    num_epoch: int = 100
    lr: float = 1e-4
    lambda1: float = 3.0                # 序列 MSE（message）
    lambda2: float = 1.0                # 梯度掩膜图像损失
    lambda3: float = 0.001              # GAN
    lambda4: float = 1.0                # 软统计特征损失
    tau: float = 1.0                    # 软统计温度
    distortion: str = 'ScreenShooting'  # Identity | ScreenShooting
    crop: bool = False                  # D6：--crop 启用随机裁剪，默认关
    crop_min_area: float = 0.2

    # ---- 目录 / 日志 / 保存 ----
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'
    result_dir: str = 'results'
    model_path: str = ''                # embed/extract 加载的 checkpoint
    log_step: int = 40
    eval_every: int = 1
    model_save_step: int = 1

    # ---- 消息 / 真值 ----
    message: str = ''                   # 18bit 字符串；空 = 随机
    watermark_file: str = ''            # extract 的地面真值 JSON

    # ---- 其他 ----
    seed: int = 0
    smoke_iters: int = 5
    command: str = 'train'              # train|embed|extract|smoke

    @staticmethod
    def from_args(args: argparse.Namespace) -> 'Config':
        """仅把 parser 中出现过的字段覆盖进默认 Config。"""
        cfg = Config()
        known = set(asdict(cfg).keys())
        for k, v in vars(args).items():
            if k in known and v is not None:
                setattr(cfg, k, v)
        cfg.command = getattr(args, 'command', 'train')
        return cfg


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='statistics',
        description='STATISTICS: 抗屏幕拍摄的统计特征水印（docs/ 为规范来源）')
    sub = parser.add_subparsers(dest='command', required=True)

    # ---- train ----
    p_tr = sub.add_parser('train', help='训练编码器/解码器/判别器')
    p_tr.add_argument('--dataset_dir', type=str, default='dataset')
    p_tr.add_argument('--batch_size', type=int, default=16)
    p_tr.add_argument('--num_epoch', type=int, default=100)
    p_tr.add_argument('--lr', type=float, default=1e-4)
    p_tr.add_argument('--lambda1', type=float, default=3.0, help='序列 MSE 权重')
    p_tr.add_argument('--lambda2', type=float, default=1.0, help='图像损失权重')
    p_tr.add_argument('--lambda3', type=float, default=0.001, help='GAN 权重')
    p_tr.add_argument('--lambda4', type=float, default=1.0, help='统计损失权重')
    p_tr.add_argument('--tau', type=float, default=1.0, help='软统计温度')
    p_tr.add_argument('--distortion', type=str, default='ScreenShooting',
                      choices=['Identity', 'ScreenShooting'])
    p_tr.add_argument('--crop', action='store_true',
                      help='启用随机裁剪增强（D6：默认关闭）')
    p_tr.add_argument('--crop_min_area', type=float, default=0.2)
    p_tr.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    p_tr.add_argument('--log_dir', type=str, default='logs')
    p_tr.add_argument('--result_dir', type=str, default='results')
    p_tr.add_argument('--log_step', type=int, default=40)
    p_tr.add_argument('--eval_every', type=int, default=1)
    p_tr.add_argument('--model_save_step', type=int, default=1)
    p_tr.add_argument('--num_workers', type=int, default=0)
    p_tr.add_argument('--seed', type=int, default=0)

    # ---- embed ----
    p_em = sub.add_parser('embed', help='对目录图像嵌入消息并保存含水印图')
    p_em.add_argument('--model_path', type=str, required=True)
    p_em.add_argument('--image_dir', type=str, required=True)
    p_em.add_argument('--message', type=str, default='',
                      help='18bit 字符串；缺省随机（写入 gt.json）')
    p_em.add_argument('--image_size', type=int, default=128)
    p_em.add_argument('--result_dir', type=str, default='results')
    p_em.add_argument('--batch_size', type=int, default=8)
    p_em.add_argument('--num_workers', type=int, default=0)
    p_em.add_argument('--seed', type=int, default=0)

    # ---- extract ----
    p_ex = sub.add_parser('extract', help='从图像提取消息')
    p_ex.add_argument('--model_path', type=str, required=True)
    p_ex.add_argument('--image_dir', type=str, required=True)
    p_ex.add_argument('--watermark_file', type=str, default='',
                      help='地面真值 gt.json（embed 输出），用于统计准确率/BER')
    p_ex.add_argument('--image_size', type=int, default=128)
    p_ex.add_argument('--batch_size', type=int, default=8)
    p_ex.add_argument('--num_workers', type=int, default=0)
    p_ex.add_argument('--seed', type=int, default=0)

    # ---- smoke ----
    p_sm = sub.add_parser('smoke', help='合成数据冒烟测试（docs/06 §4）')
    p_sm.add_argument('--smoke_iters', type=int, default=5)
    p_sm.add_argument('--image_size', type=int, default=128)
    p_sm.add_argument('--seed', type=int, default=0)

    return parser
