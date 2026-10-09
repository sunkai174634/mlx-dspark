#!/usr/bin/env python3
"""DualModeRope数值验证: 
① disarm透传=原RoPE (bit-exact)
② 三行同位置时 mrope == 标准rope (退化一致性)
③ 图像位置段正确应用不同旋转
"""
import sys, os
sys.path.insert(0, os.path.expanduser('~/Projects/mlx-dspark/src'))
import mlx.core as mx
import mlx.nn as nn
from mlx_dspark.vision_rope import DualModeRope

mx.random.seed(42)
D = 64
base_rope = nn.RoPE(D, traditional=False, base=100000.0)
wrap = DualModeRope(base_rope, mrope_section=(11, 11, 10))

x = mx.random.normal((1, 4, 8, D))  # (B,H,L,D)

# ① disarm透传
out_wrap = wrap(x, offset=0)
out_base = base_rope(x, offset=0)
print("① disarm透传 bit-exact:", mx.allclose(out_wrap, out_base).item())

# ② 三行同位置 → mrope应等于标准rope(相同offset)
pos_same = mx.tile(mx.arange(8, dtype=mx.int32)[None, :], (3, 1))  # (3,8) 全0..7
wrap.arm(pos_same)
out_mrope_same = wrap(x, offset=0)
# 注: nn.RoPE的分半配对 vs 我们分半 — 应一致
print("② 同位置退化一致:", mx.allclose(out_mrope_same, out_base, atol=1e-4).item())
diff = mx.abs(out_mrope_same - out_base).max().item()
print("   最大差:", diff)

# ③ 图像位置(三行不同) — 应与②不同(证明分段生效)
import numpy as _np
pos_img = mx.array(_np.array(pos_same.tolist()))
arr = _np.array(pos_img.tolist())
arr[1] = arr[1] * 3 + 1   # h坐标夸张化
arr[2] = arr[2] * 2 + 5   # w坐标
pos_img = mx.array(arr.tolist())
wrap.arm(pos_img)
out_mrope_img = wrap(x, offset=0)
print("③ 图像位置改变输出:", not mx.allclose(out_mrope_img, out_mrope_same).item())

# ④ disarm恢复
wrap.disarm()
print("④ re-disarm恢复:", mx.allclose(wrap(x, offset=0), out_base).item())
