#!/usr/bin/env python3
"""
第5块积木: AttentionRopeWrapper — 图像prefill时把FA层的rope换成mrope.

策略: 不改mlx-lm的Qwen3NextAttention源码, 在dspark加载后替换
attention实例的rope属性为一个双态对象:
  - disarm(默认): 完全透传原nn.RoPE — 零行为变化
  - arm(pos_ids): 图像prefill期间, rope()忽略offset, 用MRoPE按位置旋转
    (此时cache.offset恒为0 — prefill是首段, 无历史)
decode段(位置三行同步)自动回退透传 — 由调用方保证disarm.
"""
import mlx.core as mx
import mlx.nn as nn


class DualModeRope(nn.Module):
    """双态rope: 平时透传, arm后mrope"""

    def __init__(self, base_rope, mrope_section=(11, 11, 10), base=100000.0):
        super().__init__()
        self._base = base_rope          # 原nn.RoPE (保持引用, disarm透传)
        self._pos = None                 # arm时: (3, L) 位置
        self._mrope_section = list(mrope_section)
        self._base_hz = base
        # 预计算inv_freq (与mlx-vlm compute_inv_freq一致)
        dim = base_rope.dim if hasattr(base_rope, 'dim') else 64
        self._dim = dim

    # ---- 状态 ----
    def arm(self, pos_ids_3row):
        """pos_ids_3row: (3, L) mx.int32 — get_rope_index的输出"""
        self._pos = pos_ids_3row

    def disarm(self):
        self._pos = None

    # ---- rope接口(qwen3_next调用: self.rope(x, offset=cache.offset)) ----
    def __call__(self, x, offset=0):
        if self._pos is None:
            return self._base(x, offset=offset)
        # ---- mrope路径 ----
        # mlx-lm attention传入的是 (B, H, L, Dhead) — transpose后布局
        # (qwen3_next.py:134 transpose(0,2,1,3) 在rope之前)
        B, H, L, D = x.shape
        half = D // 2
        pos = self._pos.astype(mx.float32)  # (3, L)
        inv = self._base_hz ** (-mx.arange(0, half, dtype=mx.float32) / half)  # (half,)
        sec = self._mrope_section  # [11,11,10] 频率计数(半维)
        # 每个频率i属于哪一段
        seg_of = [0 if i < sec[0] else 1 if i < sec[0]+sec[1] else 2
                  for i in range(half)]
        # freqs: (L, half) — 每频率用其段的位置
        # freqs[l, i] = pos[seg_of[i], l] * inv[i]
        # 逐段构造避免int32大中间张量
        rows = []
        bounds = [0, sec[0], sec[0]+sec[1], half]
        for si in range(3):
            n = bounds[si+1] - bounds[si]
            if n <= 0:
                continue
            p_seg = pos[si].astype(mx.float32)               # (L,)
            inv_seg = inv[bounds[si]:bounds[si+1]]           # (n,)
            rows.append(mx.outer(p_seg, inv_seg))            # (L, n)
        freqs = mx.concatenate(rows, axis=-1)                # (L, half)
        # 旋转 (非交错pairing=旋转前后半分, mlx-lm nn.RoPE默认traditional=False=分半)
        x32 = x.astype(mx.float32)
        x1, x2 = x32[..., :half], x32[..., half:]
        cos = mx.cos(freqs)[None, None, :, :]            # (1,1,L,half)
        sin = mx.sin(freqs)[None, None, :, :]
        out = mx.concatenate([x1 * cos - x2 * sin,
                              x1 * sin + x2 * cos], axis=-1)
        return out.astype(x.dtype)
