# -*- coding: utf-8 -*-
"""CLI 入口（docs/03-architecture.md §4）。

用法：
  python main.py train   [--dataset_dir dataset/] [--crop] ...
  python main.py embed   --model_path checkpoints/best.pth --image_dir ... [--message ...]
  python main.py extract --model_path checkpoints/best.pth --image_dir ... [--watermark_file gt.json]
  python main.py smoke   [--smoke_iters 5]
"""
import os
import sys

from config import Config, build_arg_parser
from data_loader import get_loader, split_train_val
from solver import Solver


def cmd_train(cfg: Config):
    train_dir = os.path.join(cfg.dataset_dir, 'train')
    val_dir = os.path.join(cfg.dataset_dir, 'val')
    if not os.path.isdir(train_dir):
        sys.exit(f'错误：训练目录不存在 {train_dir}（请先按 dataset/README.md 放入数据）')
    train_paths, val_paths = None, None
    try:
        train_paths, val_paths = split_train_val(train_dir, val_dir, seed=cfg.seed)
    except FileNotFoundError as e:
        sys.exit(f'错误：{e}')
    train_loader = get_loader(train_dir, cfg.image_size, cfg.batch_size, 'train',
                              cfg.num_workers, seed=cfg.seed, shuffle=True,
                              drop_last=True, files=train_paths)
    val_loader = get_loader(val_dir, cfg.image_size, cfg.batch_size, 'train',
                            cfg.num_workers, seed=cfg.seed, shuffle=False,
                            drop_last=False, files=val_paths)
    solver = Solver(cfg)
    solver.train(train_loader, val_loader)


def cmd_embed(cfg: Config):
    loader = get_loader(cfg.image_dir, cfg.image_size, cfg.batch_size, 'embed',
                        cfg.num_workers, message=cfg.message, seed=cfg.seed,
                        shuffle=False, drop_last=False)
    Solver(cfg).embed(loader)


def cmd_extract(cfg: Config):
    loader = get_loader(cfg.image_dir, cfg.image_size, cfg.batch_size, 'extract',
                        cfg.num_workers, shuffle=False, drop_last=False)
    Solver(cfg).extract(loader)


def cmd_smoke(cfg: Config):
    from smoke_test import run_smoke
    ok = run_smoke(cfg)
    sys.exit(0 if ok else 1)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = Config.from_args(args)
    print('[config]', cfg)
    if cfg.command == 'train':
        cmd_train(cfg)
    elif cfg.command == 'embed':
        cmd_embed(cfg)
    elif cfg.command == 'extract':
        cmd_extract(cfg)
    elif cfg.command == 'smoke':
        cmd_smoke(cfg)
    else:
        parser.error(f'未知命令: {cfg.command}')


if __name__ == '__main__':
    main()
