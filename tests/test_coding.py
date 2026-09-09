# -*- coding: utf-8 -*-
"""watermark_coding 单元测试（docs/02-coding-spec.md §9 验收）。"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import watermark_coding as wc  # noqa: E402


class TestCoding(unittest.TestCase):

    def test_roundtrip_random_messages(self):
        """§9.1：1000 个随机消息（含全 0/全 1）构造→解码零错误。"""
        rng = np.random.default_rng(123)
        for msg in ['0' * 18, '1' * 18]:
            seq = wc.message_to_sequence(msg)
            self.assertEqual(wc.sequence_to_message(seq), msg)
        for _ in range(1000):
            msg = ''.join(rng.integers(0, 2, wc.MSG_LEN).astype(str))
            seq = wc.message_to_sequence(msg, rng=rng)
            self.assertEqual(wc.sequence_to_message(seq), msg)

    def test_grid_exactness(self):
        """§9.2：每特征遍历 8 级别，测量值等于网格表值。

        F1~F5 精确构造 → 误差 ≤ 1e-6；F6 精确跳变构造 → 偏差 ≤ 0.5/31。
        级别经格雷码映射到网格位置 j（相邻 j 仅差 1 bit）。
        """
        g = wc.grid_tables()
        for k in range(wc.K):
            levels = np.zeros(wc.K, dtype=int)
            if k < wc.K - 1:
                grid_vals = [g['r_grid'], g['mu_grid'], g['var_grid'],
                             g['skew_grid'], g['kurt_grid']][k]
                keys = ['r', 'mu', 'var', 'skew', 'kurt']
                for lev in range(wc.M):
                    levels[k] = lev
                    j = wc.gray_decode(lev)
                    seq = wc.levels_to_sequence(levels, rng=0)
                    feats = wc.features_from_sequence(seq)
                    self.assertAlmostEqual(feats[keys[k]], grid_vals[j], places=6,
                                           msg=f'feature {keys[k]} level {lev}')
                    dec = wc.levels_from_features(feats)
                    self.assertEqual(dec[k], lev)
            else:
                for lev in range(wc.M):
                    levels[k] = lev
                    j = wc.gray_decode(lev)
                    seq = wc.levels_to_sequence(levels, rng=0)
                    feats = wc.features_from_sequence(seq)
                    self.assertLessEqual(abs(feats['g'] - g['g_grid'][j]), 0.5 / 31,
                                         msg=f'F6 level {lev}')
                    dec = wc.levels_from_features(feats)
                    self.assertEqual(dec[k], lev)

    def test_gray_adjacency(self):
        """格雷码相邻网格位置仅差 1 bit。"""
        for j in range(wc.M - 1):
            a, b = wc.gray_encode(j), wc.gray_encode(j + 1)
            self.assertEqual(bin(a ^ b).count('1'), 1)

    def test_tolerance_to_flips(self):
        """§9.3：随机翻转 1~2 个符号，≥90% 试验消息位准确率 ≥ 16/18。"""
        rng = np.random.default_rng(7)
        n_ok = 0
        trials = 200
        for _ in range(trials):
            msg = ''.join(rng.integers(0, 2, wc.MSG_LEN).astype(str))
            seq = wc.message_to_sequence(msg, rng=rng).copy()
            n_flip = int(rng.integers(1, 3))
            idx = rng.choice(wc.L, size=n_flip, replace=False)
            seq[idx] *= -1
            out = wc.sequence_to_message(seq)
            acc = sum(a == b for a, b in zip(out, msg))
            if acc >= 16:
                n_ok += 1
        self.assertGreaterEqual(n_ok / trials, 0.90,
                                f'容错通过率 {n_ok}/{trials} < 90%')

    def test_parse_validation(self):
        with self.assertRaises(ValueError):
            wc.message_to_sequence('01' * 10)
        with self.assertRaises(ValueError):
            wc.message_to_sequence('x' * 18)
        with self.assertRaises(ValueError):
            wc.sequence_to_message(np.zeros(10))
        with self.assertRaises(ValueError):
            wc.levels_to_sequence(np.array([0, 1, 2, 3, 4, 8]))

    def test_soft_statistics_shape(self):
        import torch
        v = torch.randn(4, wc.L)
        s = wc.soft_statistics(v, tau=1.0)
        self.assertEqual(tuple(s.shape), (4, wc.K))
        # 归一化后应落在 [0,1]
        self.assertTrue(bool((s >= 0).all() and (s <= 1).all()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
