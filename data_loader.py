# -*- coding: utf-8 -*-
"""数据加载（docs/05-dataset-spec.md）。

- 训练：dataset/train/（val/ 缺省时按 9:1 从 train 拆分）；每图随机 18bit 消息；
- 嵌入：任意图像目录 + 固定（或随机）消息；
- 提取：任意图像目录，仅图像。
预处理：保持比例 resize 短边到 128 → 中心裁剪 128×128 → [−1,1]。
"""
import os

import cv2
import numpy as np
import torch
from torch.utils import data

import watermark_coding as wc

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp'}


def _list_images(img_dir):
    img_dir = os.path.normpath(img_dir)
    files = sorted(
        f for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXT
    )
    if not files:
        raise FileNotFoundError(
            f'目录 {img_dir} 中没有图像（支持 {sorted(IMG_EXT)}）。'
            f'请先放入图像，说明见 dataset/README.md。')
    return files


def load_image(path, image_size):
    """cv2 读取 + 比例 resize 短边 + 中心裁剪 + [−1,1] 归一化 → CHW float32。"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        raise IOError(f'无法读取图像: {path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    scale = image_size / min(h, w)
    nh, nw = max(image_size, int(round(h * scale))), max(image_size, int(round(w * scale)))
    img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    y0 = (nh - image_size) // 2
    x0 = (nw - image_size) // 2
    img = img[y0:y0 + image_size, x0:x0 + image_size]
    img = np.ascontiguousarray(img).transpose(2, 0, 1)
    img = np.float32(img) / 255.0 * 2.0 - 1.0
    return img


class TrainDataset(data.Dataset):
    """训练集：每图返回 (image[3,H,W], sequence[192]±1, message_str[18], index)。

    消息由 per-index RNG 生成（seed 控制，保证复现且与 num_workers 无关）。
    files 参数可覆盖目录扫描（用于 9:1 自动拆分）。
    """

    def __init__(self, data_dir, image_size=128, seed=0, files=None):
        super().__init__()
        if files is None:
            files = [os.path.join(data_dir, f) for f in _list_images(data_dir)]
        self.img_paths = list(files)
        self.image_size = image_size
        self.seed = seed

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = load_image(self.img_paths[index], self.image_size)
        rng = np.random.default_rng(self.seed * 10_000_007 + index)
        msg = ''.join(rng.integers(0, 2, wc.MSG_LEN).astype(str))
        seq = wc.message_to_sequence(msg, rng=np.random.default_rng(self.seed + index))
        return torch.from_numpy(img), torch.from_numpy(seq), msg, index


class EmbedDataset(data.Dataset):
    """嵌入集：固定消息或逐图随机消息。返回 (image, sequence, message, filename)。"""

    def __init__(self, image_dir, image_size=128, message='', seed=0):
        super().__init__()
        self.files = _list_images(image_dir)
        self.img_paths = [os.path.join(image_dir, f) for f in self.files]
        self.image_size = image_size
        self.seed = seed
        if message:
            wc.parse_message(message)  # 校验
        self.message = message

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = load_image(self.img_paths[index], self.image_size)
        if self.message:
            msg = self.message
        else:
            rng = np.random.default_rng(self.seed * 10_000_007 + index)
            msg = ''.join(rng.integers(0, 2, wc.MSG_LEN).astype(str))
        seq = wc.message_to_sequence(msg)
        return torch.from_numpy(img), torch.from_numpy(seq), msg, self.files[index]


class ExtractDataset(data.Dataset):
    """提取集：仅图像。返回 (image, filename)。"""

    def __init__(self, image_dir, image_size=128):
        super().__init__()
        self.files = _list_images(image_dir)
        self.img_paths = [os.path.join(image_dir, f) for f in self.files]
        self.image_size = image_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img = load_image(self.img_paths[index], self.image_size)
        return torch.from_numpy(img), self.files[index]


def get_loader(image_dir, image_size=128, batch_size=16, dataset='train',
               num_workers=0, message='', seed=0, shuffle=True, drop_last=True,
               files=None):
    if dataset == 'train':
        ds = TrainDataset(image_dir, image_size, seed=seed, files=files)
    elif dataset == 'embed':
        ds = EmbedDataset(image_dir, image_size, message=message, seed=seed)
    elif dataset == 'extract':
        ds = ExtractDataset(image_dir, image_size)
    else:
        raise ValueError(f'未知 dataset 类型: {dataset}')
    return data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                           num_workers=num_workers, drop_last=drop_last)


def split_train_val(train_dir, val_dir, seed=0, val_ratio=0.1):
    """若 val_dir 为空则从 train_dir 按 9:1 拆出验证集。

    返回 (train_paths, val_paths)（均为完整文件路径列表）。
    """
    train_files = _list_images(train_dir)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_files))
    val_files = [f for f in os.listdir(val_dir)
                 if os.path.splitext(f)[1].lower() in IMG_EXT] if os.path.isdir(val_dir) else []
    if val_files:
        return ([os.path.join(train_dir, f) for f in train_files],
                [os.path.join(val_dir, f) for f in val_files])
    n_val = max(1, int(round(len(train_files) * val_ratio)))
    val_idx = set(order[:n_val].tolist())
    val_paths = [os.path.join(train_dir, train_files[i]) for i in sorted(val_idx)]
    train_paths = [os.path.join(train_dir, train_files[i]) for i in range(len(train_files))
                   if i not in val_idx]
    return train_paths, val_paths
