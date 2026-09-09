# -*- coding: utf-8 -*-
"""可微分屏摄噪声层 + 随机裁剪层（docs/04-network-spec.md §4）。

ScreenShooting / Identity 移植自 PIMoG 的 Noise_Layer.py（语义不变）：
- 透视畸变（kornia，幅度 d=2）、光照失真、摩尔纹、高斯噪声 σ=√0.001；
- 已知端口行为保留（Light_Distortion 的 direction 2/3/4 均等价 rot90×1；
  光照/摩尔纹在 numpy 中计算后转 torch，设备与透视结果一致）。

RandomCrop 为本项目新增（D6）：随机裁剪 20%~100% 面积并双线性缩放回原尺寸，
训练时由 --crop 启用，模拟"部分拍摄"。
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
from kornia.geometry.transform import (
    get_affine_matrix2d,
    get_perspective_transform,
    get_rotation_matrix2d,
    warp_affine,
    warp_perspective,
)


# ---------------------------------------------------------------- PIMoG 移植
def _compute_translation_matrix(translation: torch.Tensor) -> torch.Tensor:
    matrix: torch.Tensor = torch.eye(
        3, device=translation.device, dtype=translation.dtype)
    matrix = matrix.repeat(translation.shape[0], 1, 1)
    dx, dy = torch.chunk(translation, chunks=2, dim=-1)
    matrix[..., 0, 2:3] += dx
    matrix[..., 1, 2:3] += dy
    return matrix


def _compute_tensor_center(tensor: torch.Tensor) -> torch.Tensor:
    assert 2 <= len(tensor.shape) <= 4, f"Must be a 3D tensor as HW, CHW and BCHW. Got {tensor.shape}."
    height, width = tensor.shape[-2:]
    center_x: float = float(width - 1) / 2
    center_y: float = float(height - 1) / 2
    return torch.tensor([center_x, center_y], device=tensor.device, dtype=tensor.dtype)


def _compute_rotation_matrix(angle: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
    scale: torch.Tensor = torch.ones((angle.shape[0], 2))
    return get_rotation_matrix2d(center, angle, scale)


def translate(image, device, d=8):
    c = image.shape[0]
    h = image.shape[-2]
    w = image.shape[-1]
    trans = torch.ones(c, 2)
    for i in range(c):
        dx = random.uniform(-d, d)
        dy = random.uniform(-d, d)
        trans[i, :] = torch.tensor([[dx, dy]])
    translation_matrix: torch.Tensor = _compute_translation_matrix(trans)
    matrix = translation_matrix[..., :2, :3]
    is_unbatched: bool = image.ndimension() == 3
    if is_unbatched:
        image = torch.unsqueeze(image, dim=0)
    matrix = matrix.expand(image.shape[0], -1, -1).to(device)
    data_warp: torch.Tensor = warp_affine(image, matrix, dsize=(h, w),
                                          padding_mode='border').to(device)
    if is_unbatched:
        data_warp = torch.squeeze(data_warp, dim=0)
    return data_warp


def rotate(image, device, d=8):
    c = image.shape[0]
    h = image.shape[-2]
    w = image.shape[-1]
    angle = torch.ones(c)
    center = torch.ones(c, 2)
    for i in range(c):
        an = random.uniform(-d, d)
        angle[i] = torch.tensor([an])
        center[i, :] = torch.tensor([[h / 2 - 1, w / 2 - 1]])
    angle = angle.expand(image.shape[0])
    center = center.expand(image.shape[0], -1)
    rotation_matrix: torch.Tensor = _compute_rotation_matrix(angle, center)
    matrix = rotation_matrix[..., :2, :3]
    is_unbatched: bool = image.ndimension() == 3
    if is_unbatched:
        image = torch.unsqueeze(image, dim=0)
    matrix = matrix.expand(image.shape[0], -1, -1).to(device)
    data_warp: torch.Tensor = warp_affine(image, matrix, dsize=(h, w),
                                          padding_mode='border').to(device)
    if is_unbatched:
        data_warp = torch.squeeze(data_warp, dim=0)
    return data_warp


def perspective(image, device, d=8):
    c = image.shape[0]
    h = image.shape[2]
    w = image.shape[3]
    image_size = h
    points_src = torch.ones(c, 4, 2)
    points_dst = torch.ones(c, 4, 2)
    for i in range(c):
        points_src[i, :, :] = torch.tensor([[
            [0., 0.], [w - 1., 0.], [w - 1., h - 1.], [0., h - 1.],
        ]])
        tl_x = random.uniform(-d, d)
        tl_y = random.uniform(-d, d)
        bl_x = random.uniform(-d, d)
        bl_y = random.uniform(-d, d)
        tr_x = random.uniform(-d, d)
        tr_y = random.uniform(-d, d)
        br_x = random.uniform(-d, d)
        br_y = random.uniform(-d, d)
        points_dst[i, :, :] = torch.tensor([[
            [tl_x, tl_y],
            [tr_x + image_size, tr_y],
            [br_x + image_size, br_y + image_size],
            [bl_x, bl_y + image_size],
        ]])
    M = get_perspective_transform(points_src, points_dst).to(device)
    data_warp: torch.Tensor = warp_perspective(image.float(), M, dsize=(h, w)).to(device)
    return data_warp


def MoireGen(p_size, theta, center_x, center_y):
    z = np.zeros((p_size, p_size))
    for i in range(p_size):
        for j in range(p_size):
            z1 = 0.5 + 0.5 * np.cos(2 * np.pi * np.sqrt((i + 1 - center_x) ** 2 + (j + 1 - center_y) ** 2))
            z2 = 0.5 + 0.5 * np.cos(np.cos(theta / 180 * np.pi) * (j + 1) + np.sin(theta / 180 * np.pi) * (i + 1))
            z[i, j] = np.minimum(z1, z2)
    M = (z + 1) / 2
    return M


def Light_Distortion(c, embed_image):
    # 端口保留 PIMoG 原语义：c==0 时 direction 2/3/4 均等价 rot90×1。
    mask = np.zeros((embed_image.shape))
    mask_2d = np.zeros((embed_image.shape[2], embed_image.shape[3]))
    a = 0.7 + np.random.rand(1) * 0.2
    b = 1.1 + np.random.rand(1) * 0.2
    if c == 0:
        direction = np.random.randint(1, 5)
        for i in range(embed_image.shape[2]):
            mask_2d[i, :] = -((b - a) / (mask.shape[2] - 1)) * (i - mask.shape[3]) + a
        if direction == 1:
            O = mask_2d
        elif direction == 2:
            O = np.rot90(mask_2d, 1)
        elif direction == 3:
            O = np.rot90(mask_2d, 1)
        elif direction == 4:
            O = np.rot90(mask_2d, 1)
        for batch in range(embed_image.shape[0]):
            for channel in range(embed_image.shape[1]):
                mask[batch, channel, :, :] = mask_2d
    else:
        x = np.random.randint(0, mask.shape[2])
        y = np.random.randint(0, mask.shape[3])
        max_len = np.max([np.sqrt(x ** 2 + y ** 2),
                          np.sqrt((x - 255) ** 2 + y ** 2),
                          np.sqrt(x ** 2 + (y - 255) ** 2),
                          np.sqrt((x - 255) ** 2 + (y - 255) ** 2)])
        for i in range(mask.shape[2]):
            for j in range(mask.shape[3]):
                mask[:, :, i, j] = np.sqrt((i - x) ** 2 + (j - y) ** 2) / max_len * (a - b) + b
        O = mask
    return O


def Moire_Distortion(embed_image):
    Z = np.zeros((embed_image.shape))
    for i in range(3):
        theta = np.random.randint(0, 180)
        center_x = np.random.rand(1).item() * embed_image.shape[2]
        center_y = np.random.rand(1).item() * embed_image.shape[3]
        M = MoireGen(embed_image.shape[2], theta, center_x, center_y)
        Z[:, i, :, :] = M
    return Z


class ScreenShooting(nn.Module):
    def __init__(self):
        super(ScreenShooting, self).__init__()

    def forward(self, embed_image):
        noised_image = torch.zeros_like(embed_image)
        device = embed_image.device

        # perspective transform
        noised_image = perspective(embed_image, device, 2)

        # Light Distortion
        c = np.random.randint(0, 2)
        L = Light_Distortion(c, embed_image)

        # Moire Distortion
        Z = Moire_Distortion(embed_image) * 2 - 1
        Li = L.copy()
        Mo = Z.copy()
        noised_image = noised_image * torch.from_numpy(Li).to(device) * 0.85 \
            + torch.from_numpy(Mo).to(device) * 0.15

        # Gaussian noise
        noised_image = noised_image + 0.001 ** 0.5 * torch.randn(noised_image.size()).to(device)

        return noised_image


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, embed_image):
        return embed_image


# ---------------------------------------------------------------- 随机裁剪层
class RandomCrop(nn.Module):
    """随机裁剪 20%~100% 面积并双线性缩放回原尺寸（可微分，docs/04 §4.2）。

    语义：模拟"部分拍摄"。每样本独立随机：保留面积 a∈[min_area,1]，
    长宽比 aspect∈[3/4,4/3]，裁剪窗口位置均匀随机；仿射矩阵 = 以窗口中心
    为旋转中心、scale=(W/w_c, H/h_c) 的缩放，warp_affine 双线性采样。
    """

    def __init__(self, min_area: float = 0.2, min_aspect: float = 3 / 4,
                 max_aspect: float = 4 / 3):
        super(RandomCrop, self).__init__()
        self.min_area = float(min_area)
        self.min_aspect = float(min_aspect)
        self.max_aspect = float(max_aspect)

    def forward(self, x):
        if x.dim() == 3:
            unbatched = True
            x = x.unsqueeze(0)
        else:
            unbatched = False
        B, C, H, W = x.shape
        device = x.device
        centers = torch.zeros(B, 2, device=device, dtype=x.dtype)
        scales = torch.zeros(B, 2, device=device, dtype=x.dtype)
        for i in range(B):
            a = random.uniform(self.min_area, 1.0)
            aspect = random.uniform(self.min_aspect, self.max_aspect)
            h_c = min(H, max(1, int(round(math.sqrt(a * H * W / aspect)))))
            w_c = min(W, max(1, int(round(h_c * aspect))))
            h_c = min(H, max(1, int(round(w_c / aspect))))
            cx = random.uniform(w_c / 2.0, W - w_c / 2.0)
            cy = random.uniform(h_c / 2.0, H - h_c / 2.0)
            centers[i, 0] = cx
            centers[i, 1] = cy
            scales[i, 0] = W / w_c
            scales[i, 1] = H / h_c
        angles = torch.zeros(B, device=device, dtype=x.dtype)
        matrix = get_affine_matrix2d(torch.zeros_like(centers), centers, scales, angles)
        matrix = matrix[..., :2, :]  # kornia 0.6.12 返回 Bx3x3 齐次矩阵
        out = warp_affine(x, matrix, dsize=(H, W), padding_mode='border')
        if unbatched:
            out = out.squeeze(0)
        return out
