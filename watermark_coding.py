# -*- coding: utf-8 -*-
"""水印编码核心（docs/02-coding-spec.md 的唯一实现）。

消息(18bit) ↔ 统计特征(K=6, M=8级) ↔ 水印序列(±1, L=192=6块×32)。

要点：
- F1~F5 的网格统一取 p≥0.5 上半支 p_ℓ=0.5+(ℓ+0.5)/16，避免 σ²(p)=σ²(1−p)、
  κ(p)=κ(1−p) 的对称解码歧义；
- 块 k 承载特征 F_{k+1}：块 0→F1 比例(p)、块 1→F2 均值(p)、块 2→F3 方差(p)、
  块 3→F4 偏度(p)、块 4→F5 峰度(p)、块 5→F6 梯度能量(q, Markov 链)；
- F1~F5 块采用"精确计数构造"（块内恰好 c_ℓ 个 +1），测量值精确等于网格表值。
"""
import numpy as np
import torch

# ---- 常量（与 02 文档 §2 一致）----
K = 6           # 特征数
M = 8           # 量化级别
N = 32          # 每块符号数
L = K * N       # 序列长度 192
MSG_LEN = 18    # 消息位数 = K * log2(M)


def grid_tables():
    """返回量化网格表（02 文档 §4 的权威数值）。"""
    p_grid = 0.5 + (np.arange(M) + 0.5) / 16.0          # (8,)
    q_grid = (np.arange(M) + 0.5) / 8.0                 # (8,)
    c_grid = np.round(N * p_grid).astype(int)           # 17..31（奇数）
    r_grid = p_grid.copy()
    mu_grid = 2.0 * p_grid - 1.0
    var_grid = 4.0 * p_grid * (1.0 - p_grid)
    skew_grid = (1.0 - 2.0 * p_grid) / np.sqrt(p_grid * (1.0 - p_grid))
    kurt_grid = 1.0 / (p_grid * (1.0 - p_grid)) - 6.0
    g_grid = q_grid.copy()
    feature_grids = np.stack([r_grid, mu_grid, var_grid, skew_grid, kurt_grid, g_grid])
    return dict(
        p_grid=p_grid, q_grid=q_grid, c_grid=c_grid,
        r_grid=r_grid, mu_grid=mu_grid, var_grid=var_grid,
        skew_grid=skew_grid, kurt_grid=kurt_grid, g_grid=g_grid,
        feature_grids=feature_grids,
        norm_min=feature_grids.min(axis=1),
        norm_max=feature_grids.max(axis=1),
    )


_GRIDS = grid_tables()


def _rng(rng):
    if rng is None:
        return np.random.default_rng(0)
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def gray_encode(j: int) -> int:
    """3 bit 格雷码：j → j^(j>>1)。相邻 j 的编码仅差 1 bit。"""
    return j ^ (j >> 1)


def gray_decode(level: int) -> int:
    """3 bit 格雷码解码：level → j（连续异或）。"""
    return level ^ (level >> 1) ^ (level >> 2)


def parse_message(msg) -> np.ndarray:
    """18bit 消息字符串/数组 → bool 数组 (18,)。非法输入抛 ValueError。"""
    if isinstance(msg, str):
        msg = msg.strip()
        if len(msg) != MSG_LEN or any(c not in '01' for c in msg):
            raise ValueError(f'消息必须是 {MSG_LEN} 位 0/1 字符串，收到: {msg!r}')
        return np.array([c == '1' for c in msg], dtype=bool)
    arr = np.asarray(msg).ravel()
    if arr.size != MSG_LEN or not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f'消息必须是 {MSG_LEN} 维 0/1 数组，收到 shape={arr.shape}')
    return arr.astype(bool)


def message_to_levels(msg) -> np.ndarray:
    """消息 → 6 个量化级别 ℓ_k（每 3 bit 一组，MSB 在前）。"""
    bits = parse_message(msg)
    return np.array([
        int(bits[3 * k]) * 4 + int(bits[3 * k + 1]) * 2 + int(bits[3 * k + 2])
        for k in range(K)
    ], dtype=int)


def levels_to_message(levels) -> str:
    """6 个级别 → 18bit 消息字符串（MSB 在前）。"""
    levels = np.asarray(levels, dtype=int)
    out = []
    for lev in levels:
        out += [str((lev >> 2) & 1), str((lev >> 1) & 1), str(lev & 1)]
    return ''.join(out)


def _random_composition(total: int, parts: int, rng) -> np.ndarray:
    """把 total 随机拆成 parts 个非负整数（multinomial 均匀分布）。"""
    if parts == 0:
        return np.zeros(0, dtype=int)
    if parts == 1:
        return np.array([total], dtype=int)
    splits = np.sort(rng.choice(total + parts - 1, size=parts - 1, replace=False))
    borders = np.concatenate(([-1], splits, [total + parts - 1]))
    return np.diff(borders) - 1


def _markov_block_exact(t: int, rng) -> np.ndarray:
    """构造长度 N 的 ±1 块：恰好 t 个相邻跳变、+1/−1 计数各 16（平衡）。

    原理：t 个跳变 ⇔ t+1 段连续同号游程；以 + 开头、正负交替，
    奇(偶)游程长度之和需为 16，用随机组合拆分实现。
    """
    n_runs = t + 1
    n_odd = (n_runs + 1) // 2   # 0,2,4,... 号游程（+1）
    n_even = n_runs - n_odd     # 1,3,5,... 号游程（−1）
    odd_extra = _random_composition(16 - n_odd, n_odd, rng)
    even_extra = _random_composition(16 - n_even, n_even, rng)
    odd_len = odd_extra + 1
    even_len = even_extra + 1
    block = np.empty(N, dtype=np.float32)
    pos = 0
    for i in range(n_runs):
        if i % 2 == 0:
            ln = int(odd_len[i // 2])
            block[pos:pos + ln] = 1.0
        else:
            ln = int(even_len[i // 2])
            block[pos:pos + ln] = -1.0
        pos += ln
    assert pos == N, f'游程长度和 {pos} != {N}'
    assert int(np.sum(block == 1.0)) == 16, '块不平衡'
    return block


def levels_to_sequence(levels, rng=None) -> np.ndarray:
    """6 个级别 → ±1 序列 (192,)（02 文档 §5）。

    块 0~4（F1~F5）：恰好 c_ℓ 个 +1，其余 −1，随机打乱；
    块 5（F6）：精确 t_ℓ=round(31·q_ℓ) 个跳变的平衡 Markov 游程块
    （测量 ĝ=t_ℓ/31 与网格 q_ℓ 的最大偏差 0.5/31≈0.016，远小于半格 0.0625）。
    """
    levels = np.asarray(levels, dtype=int)
    if levels.shape != (K,) or np.any(levels < 0) or np.any(levels >= M):
        raise ValueError(f'levels 必须是 ({K},) 且取值 0..{M - 1}，收到 {levels}')
    rng = _rng(rng)
    seq = np.empty(L, dtype=np.float32)
    for k in range(K - 1):
        j = gray_decode(int(levels[k]))
        c = int(_GRIDS['c_grid'][j])
        block = np.ones(N, dtype=np.float32)
        block[c:] = -1.0
        rng.shuffle(block)
        seq[k * N:(k + 1) * N] = block
    # 块 5：精确跳变数构造
    j = gray_decode(int(levels[K - 1]))
    t = int(round(31.0 * float(_GRIDS['q_grid'][j])))
    seq[(K - 1) * N:] = _markov_block_exact(t, rng)
    return seq


def message_to_sequence(msg, rng=None) -> np.ndarray:
    """消息 → ±1 序列（构造确定：同一消息恒得同一序列，rng 缺省 seed=0）。"""
    return levels_to_sequence(message_to_levels(msg), rng=rng)


def threshold(v) -> np.ndarray:
    """硬阈值：sign(v)，约定 sign(0)=+1（02 文档 §6.1）。"""
    v = np.asarray(v, dtype=np.float64)
    return np.where(v >= 0.0, 1.0, -1.0)


def features_from_sequence(seq) -> dict:
    """硬阈值后逐块测量 6 个特征（02 文档 §6.2；ddof=0 总体统计量）。

    返回 dict：keys 'r','mu','var','skew','kurt','g'。
    """
    s = threshold(seq)
    if s.size != L:
        raise ValueError(f'序列长度必须为 {L}，收到 {s.size}')
    out = {}
    blocks = [s[k * N:(k + 1) * N] for k in range(K)]
    # F1 比例（块0）
    out['r'] = float(np.mean(blocks[0] == 1.0))
    # F2 均值（块1）
    out['mu'] = float(np.mean(blocks[1]))
    # F3 方差（块2）
    out['var'] = float(np.var(blocks[2]))
    # F4 偏度（块3）
    b3 = blocks[3]
    v3 = float(np.var(b3))
    if v3 > 0:
        out['skew'] = float(np.mean((b3 - np.mean(b3)) ** 3) / (v3 ** 1.5))
    else:
        out['skew'] = 0.0
    # F5 峰度（块4）
    b4 = blocks[4]
    v4 = float(np.var(b4))
    if v4 > 0:
        out['kurt'] = float(np.mean((b4 - np.mean(b4)) ** 4) / (v4 ** 2) - 3.0)
    else:
        out['kurt'] = 0.0
    # F6 梯度能量（块5）
    b5 = blocks[5]
    out['g'] = float(np.mean((b5[1:] - b5[:-1]) ** 2) / 4.0)
    return out


def levels_from_features(features) -> np.ndarray:
    """6 特征 → 6 级别（02 文档 §6.3~6.4；等距时取较小 ℓ）。"""
    if isinstance(features, dict):
        f = [features['r'], features['mu'], features['var'],
             features['skew'], features['kurt'], features['g']]
    else:
        f = np.asarray(features, dtype=np.float64).ravel()
    if len(f) != K:
        raise ValueError(f'特征数必须为 {K}，收到 {len(f)}')
    p_grid = _GRIDS['p_grid']
    q_grid = _GRIDS['q_grid']
    p_hat = np.empty(K - 1, dtype=np.float64)
    r, mu, var, skew, kurt = f[0], f[1], f[2], f[3], f[4]
    p_hat[0] = float(np.clip(r, 0.5, 1.0))
    p_hat[1] = float(np.clip((mu + 1.0) / 2.0, 0.5, 1.0))
    var_c = float(np.clip(var, 0.0, 1.0))
    p_hat[2] = (1.0 + np.sqrt(1.0 - var_c)) / 2.0
    p_hat[3] = (1.0 - skew / np.sqrt(skew ** 2 + 4.0)) / 2.0
    kc = float(kurt)
    p_hat[4] = 0.5 if kc <= -2.0 else (1.0 + np.sqrt(1.0 - 4.0 / (kc + 6.0))) / 2.0
    p_hat = np.clip(p_hat, 0.5, 1.0)
    levels = np.empty(K, dtype=int)
    for k in range(K - 1):
        levels[k] = gray_encode(int(np.argmin(np.abs(p_hat[k] - p_grid))))
    levels[K - 1] = gray_encode(int(np.argmin(np.abs(f[5] - q_grid))))
    return levels


def sequence_to_message(seq) -> str:
    """序列（实值或 ±1）→ 18bit 消息（内部硬阈值）。"""
    return levels_to_message(levels_from_features(features_from_sequence(seq)))


def message_to_target_features(msg) -> np.ndarray:
    """消息 → 归一化目标特征向量 (6,)，与 soft_statistics 输出同空间（损失用）。

    每特征取该消息级别对应网格位置 j=gray⁻¹(ℓ) 的网格值，再按网格 min/max 归一化。
    """
    levels = message_to_levels(msg)
    feats = np.empty(K, dtype=np.float64)
    for k in range(K):
        j = gray_decode(int(levels[k]))
        feats[k] = _GRIDS['feature_grids'][k, j]
    return (feats - _GRIDS['norm_min']) / (_GRIDS['norm_max'] - _GRIDS['norm_min'])


def soft_statistics(v: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """软统计特征（02 文档 §7）：解码器输出 v (B,192) → (B,6) 已逐特征 min-max 归一化。

    u=σ(v/τ)∈[0,1]，w=2u−1；块 k 用其对应特征公式计算，再按网格表 min/max 归一化。
    """
    if v.dim() == 1:
        v = v.unsqueeze(0)
    if v.shape[-1] != L:
        raise ValueError(f'序列长度必须为 {L}，收到 {v.shape[-1]}')
    u = torch.sigmoid(v / tau)
    w = 2.0 * u - 1.0
    u = u.view(-1, K, N)
    w = w.view(-1, K, N)

    s0 = u[:, 0, :].mean(dim=1)                        # F1 比例
    s1 = w[:, 1, :].mean(dim=1)                        # F2 均值
    s2 = w[:, 2, :].var(dim=1, unbiased=False)         # F3 方差
    b3 = w[:, 3, :]
    m3 = b3.mean(dim=1, keepdim=True)
    v3 = ((b3 - m3) ** 2).mean(dim=1) + 1e-8
    s3 = ((b3 - m3) ** 3).mean(dim=1) / (v3 ** 1.5)    # F4 偏度
    b4 = w[:, 4, :]
    m4 = b4.mean(dim=1, keepdim=True)
    v4 = ((b4 - m4) ** 2).mean(dim=1) + 1e-8
    s4 = ((b4 - m4) ** 4).mean(dim=1) / (v4 ** 2) - 3.0  # F5 峰度
    u5 = u[:, 5, :]
    s5 = ((u5[:, 1:] - u5[:, :-1]) ** 2).mean(dim=1)   # F6 梯度能量

    raw = torch.stack([s0, s1, s2, s3, s4, s5], dim=1)  # (B,6)
    lo = torch.tensor(_GRIDS['norm_min'], dtype=v.dtype, device=v.device).view(1, K)
    hi = torch.tensor(_GRIDS['norm_max'], dtype=v.dtype, device=v.device).view(1, K)
    return torch.clamp((raw - lo) / (hi - lo).clamp_min(1e-8), 0.0, 1.0)
