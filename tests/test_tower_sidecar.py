#!/usr/bin/env python3
"""
vision_tower侧载原型: 从orca safetensors直接读塔权重 → mlx-vlm的VisionModel

绕过mlx-lm sanitize()丢弃塔的问题: 不改语言模型加载, 独立实例化塔.
成功标准: 塔前向输出shape正确 (N_patches, 5120) 且与mlx-vlm整模型路径一致.
"""
import os, sys, json, glob
import mlx.core as mx
import mlx.nn as nn

MODEL = os.path.expanduser("~/Models/VLM/Qwen/Qwen3.8-27B-4bit-crack-orcarouter")

# 1. 塔的config
full_cfg = json.load(open(os.path.join(MODEL, "config.json")))
vis_cfg = full_cfg["vision_config"]
print("vision_config keys:", list(vis_cfg.keys())[:12])

# 2. 实例化mlx-vlm的Qwen3.5 VisionModel (继承Qwen3VL)
from mlx_vlm.models.qwen3_5.vision import VisionModel
from mlx_vlm.models.qwen3_5.config import VisionConfig
cfg = VisionConfig.from_dict(vis_cfg)
tower = VisionModel(cfg)

# 3. 从safetensors读塔权重(mx.load直接读分片, 树里有全部键)
print("loading tower weights from shard 1…")
weights = mx.load(os.path.join(MODEL, "model-00001-of-00003.safetensors"))
tower_weights = {k: v for k, v in weights.items() if k.startswith("vision_tower")}
print("塔张量:", len(tower_weights))

# 4. 装载 + 对齐键名 (safetensors键带vision_tower.前缀, 模块期望同名或经sanitize)
from mlx_vlm.models.qwen3_5.vision import VisionModel as VM
# 看VisionModel.sanitize期望什么键
import inspect
src = inspect.getsource(VM.sanitize) if hasattr(VM, 'sanitize') else "no sanitize"
print("sanitize源码片段:", src[:300] if isinstance(src, str) else src)


# 5. 装载(sanitize对齐) + 前向
sanitized = tower.sanitize(tower_weights)
trouble = tower.load_weights(list(sanitized.items()), strict=False)
missing = [k for k, _ in trouble.get("missing", [])] if isinstance(trouble, dict) else trouble
print("装载报告:", str(missing)[:200] if missing else "全部命中")

# 6. 前向: 用processor的真实输入
from mlx_vlm.utils import load_processor
from pathlib import Path
from PIL import Image
processor = load_processor(Path(MODEL))
img = Image.open('/Users/roriksun/.hermes/tmp/crownHQ4/master_s180_v.jpg').convert('RGB')
formatted = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>x<|im_end|>\n<|im_start|>assistant\n"
enc = processor(text=formatted, images=[img], return_tensors="mlx")
pv = enc["pixel_values"].astype(tower.patch_embed.proj.weight.dtype)
grid = enc["image_grid_thw"]
print("输入:", pv.shape, "grid:", grid.tolist())
mx.eval(tower.parameters())
hidden, _ = tower(pv, grid)
mx.eval(hidden)
print("✅ 塔前向输出:", hidden.shape, hidden.dtype)
# 期望: (1170, 5120) — merge后每patch 1行 × hidden 5120
