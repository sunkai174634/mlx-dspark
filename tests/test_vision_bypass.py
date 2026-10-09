#!/usr/bin/env python3
"""
验证核心假设: 包装orca(经mlx-lm qwen3_5路径)的embed_tokens,
让<|image_pad|> id (248056) 从预置的vision embedding表取值, 
其余id照常查表. 用小规模张量直接测包装逻辑 + 真实模型加载测全链路.
"""
import os
import mlx.core as mx
import mlx.nn as nn

MODEL = os.path.expanduser("~/Models/VLM/Qwen/Qwen3.8-27B-4bit-crack-orcarouter")
IMAGE_PAD = 248056

# ---- 1. 纯逻辑测试: 包装器 ----
class VisionEmbedBypass(nn.Module):
    """包装embed_tokens: image_pad id → _vision_table按位置取"""
    def __init__(self, embed):
        super().__init__()
        self._embed = embed
        self._vision_table = None   # (N, hidden) 每请求重置
        self._consumed = 0

    def __call__(self, ids):
        ids = mx.array(ids) if not isinstance(ids, mx.array) else ids
        out = self._embed(ids)
        if self._vision_table is not None:
            pad_rows = (ids == IMAGE_PAD)  # (L,)
            n_pad = int(pad_rows.sum().item())
            if n_pad:
                # 顺序取: 消耗指针推进(多图请求也支持)
                take = self._vision_table[self._consumed:self._consumed + n_pad]
                self._consumed += n_pad
                # MLX无nonzero: 用cumsum构造每行的take索引
                # pad行j(0-based) → take[j]; 非pad行 → 任意(会被where丢弃)
                rank = mx.cumsum(pad_rows.astype(mx.int32)) - 1  # pad行=0..n-1
                L = out.shape[0]
                # gather: take按rank取(非pad行的rank可能越界, clamp住)
                safe_rank = mx.clip(rank, 0, max(n_pad - 1, 0))
                gathered = take[safe_rank]              # (L, H)
                out = mx.where(pad_rows[:, None], gathered, out)
        return out

    def arm(self, table):
        self._vision_table = table
        self._consumed = 0

    def disarm(self):
        self._vision_table = None
        self._consumed = 0

print("=== 逻辑测试(假embedding) ===")
class FakeEmbed(nn.Module):
    def __call__(self, ids):
        return mx.zeros((ids.shape[0], 8)) + 1.0  # 全1
fe = FakeEmbed()
wrap = VisionEmbedBypass(fe)
ids = mx.array([100, IMAGE_PAD, IMAGE_PAD, IMAGE_PAD, 200])
wrap.arm(mx.array([[1.,2.,3.,4.,5.,6.,7.,8.],
                   [9.]*8, [10.]*8]))
out = wrap(ids)
print("输出行(应=1,1,2,3,10,1...即pad处取表):")
for i, row in enumerate(out.tolist()):
    print(f"  id={ids[i].item():>7} → {row[:3]}...")
wrap.disarm()
print("disarm后:", wrap(ids).tolist()[1][:2], "(应回到1.0)")
