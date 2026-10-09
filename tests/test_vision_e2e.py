import os, sys
sys.path.insert(0, os.path.expanduser('~/Projects/mlx-dspark/src'))
import mlx.core as mx
import mlx.nn as nn

MODEL = os.path.expanduser("~/Models/VLM/Qwen/Qwen3.8-27B-4bit-crack-orcarouter")
IMAGE_PAD = 248056
IMG = '/Users/roriksun/.hermes/tmp/crownHQ4/master_s180_v.jpg'

class VisionEmbedBypass(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self._embed = embed
        self._vision_table = None
        self._consumed = 0
    def __call__(self, ids):
        ids = mx.array(ids) if not isinstance(ids, mx.array) else ids
        out = self._embed(ids)
        if self._vision_table is not None:
            pad_rows = (ids == IMAGE_PAD)
            n_pad = int(pad_rows.sum().item())
            if n_pad:
                take = self._vision_table[self._consumed:self._consumed + n_pad]
                self._consumed += n_pad
                rank = mx.cumsum(pad_rows.astype(mx.int32)) - 1
                safe_rank = mx.clip(rank, 0, max(n_pad - 1, 0))
                gathered = take[safe_rank]
                out = mx.where(pad_rows[:, None], gathered, out)
        return out
    def arm(self, table): self._vision_table = table; self._consumed = 0
    def disarm(self): self._vision_table = None; self._consumed = 0

print("1) loading target via dspark loader…")
from mlx_dspark.load import load_target
target, tokenizer = load_target(MODEL)
inner = target._inner
mm = inner.model
print("   family:", target.family, "| is_vlm:", target.is_vlm)

# 3. 用mlx-vlm的processor+vision tower算真实embedding
print("2) computing real vision embeddings…")
from mlx_vlm import load as vlm_load
from mlx_vlm.utils import load_processor
from pathlib import Path
from PIL import Image
import mlx_vlm

# 只借processor+vision塔(独立轻载会重复载LM…用同一份weights不好借)
# 直接方案: vlm processor编码 → 手动跑model.vision_tower
proc_dir = MODEL
processor = load_processor(Path(MODEL))
img = Image.open(IMG).convert("RGB")

# qwen3.5系: processor(text, images) → pixel_values + grid_thw
text = "<|vision_start|>" + "<|image_pad|>"*0 + "<|vision_end|>描述"
# 先用processor拿图像张量和真实pad数
formatted = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>describe<|im_end|>\n<|im_start|>assistant\n"
enc = processor(text=formatted, images=[img], return_tensors="mlx")
pv = enc["pixel_values"]; grid = enc["image_grid_thw"]
n_real = int(grid.prod().item()) // 4  # merge 2x2
print("   pixel_values:", pv.shape, "| grid:", grid.tolist(), "| 真实pad数:", n_real)

# 4. vision tower前向 → embedding
print("3) vision tower forward…")
# orca经dspark载入是language-only路径 — vision_tower不在!
# 检查:
has_vt = hasattr(inner, 'vision_tower') or hasattr(mm, 'vision_tower')
print("   inner/mm 有vision_tower?", has_vt)
# qwen3_5经mlx-lm载入只有language部分 — vision塔被丢了(mlx-lm只解析text_config)
# 这是第2个障碍: dspark加载orca时vision塔权重就没进内存
import glob, json
idx = json.load(open(glob.glob(MODEL+'/*.index.json')[0])) if glob.glob(MODEL+'/*.index.json') else None
n_vis = sum(1 for k in (idx['weight_map'] if idx else []) if 'vision_tower' in k)
print(f"   权重文件里vision_tower张量数: {n_vis}")
