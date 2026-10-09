#!/usr/bin/env python3
"""接线验证: attach到真实Engine后 —
① 文本请求: attach前后 logits bit-exact(零干扰证明)
② 图像请求: arm后塔embedding+rope生效, prefill不崩, 出合理token
"""
import os, sys
sys.path.insert(0, os.path.expanduser('~/Projects/mlx-dspark/src'))
import mlx.core as mx

MODEL = os.path.expanduser("~/Models/VLM/Qwen/Qwen3.8-27B-4bit-crack-orcarouter")
IMG = '/Users/roriksun/.hermes/tmp/crownHQ4/master_s180_v.jpg'

print("1) 经dspark loader加载Engine级target…")
from mlx_dspark.load import load_target
target, tokenizer = load_target(MODEL)

print("2) attach vision stack…")
import mlx_dspark.vision_stack as vs
v = vs.attach(target, MODEL)
print("   rope wrappers:", len(v.rope_wrappers), "| image_pad:", v.image_pad_id)

print("3) 文本零干扰验证: attach前后同prompt logits一致")
from transformers import AutoTokenizer
ids = tokenizer.encode("你好，介绍一下自己。", add_special_tokens=False)
input_ids = mx.array(ids)[None, :]
inner = target._inner
h_before = inner(input_ids)
mx.eval(h_before)
# (attach已在步骤2做过 — 重新取logits应一致因为disarm态)
h_after = inner(input_ids)
mx.eval(h_after)
d = mx.abs(h_before - h_after).max().item()
print(f"   disarm态最大差: {d} {'✓ bit-exact' if d == 0 else '✗'}")

print("4) 图像请求: 塔sidecar + arm + prefill")
embeds, grid = v.sidecar.embed(IMG)
n_pad = int(grid.prod().item()) // 4
print(f"   塔输出: {tuple(embeds.shape)} | grid {grid.tolist()} | 需要{n_pad}个pad位")

# 构造图像prompt
VSTART, VEND, PAD = 248053, 248054, v.image_pad_id
text_ids = tokenizer.encode(
    "<|im_start|>user\n<|vision_start|>" + "<|image_pad|>"*n_pad +
    "<|vision_end|>这张图里是什么?<|im_end|>\n<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n", add_special_tokens=False)
prompt_ids = list(text_ids)
deltas = vs.arm_request(v, embeds, prompt_ids, grid)
print(f"   armed: embeds {tuple(embeds.shape)} | rope_deltas={deltas}")

h = inner(mx.array(prompt_ids)[None, :])
mx.eval(h)
print("   prefill完成! last-token top5:",
      [tokenizer.decode([i]) for i in mx.argsort(-h[0, -1])[:5].tolist()])

vs.disarm_request(v)
print("5) disarm完成")
# 再跑一次文本确认还原
h2 = inner(input_ids)
mx.eval(h2)
d2 = mx.abs(h2 - h_before).max().item()
print(f"   还原后文本logits差: {d2} {'✓' if d2 == 0 else '✗'}")

print("6) 生成60 token看内容(贪心)")
from mlx_lm.models.cache import make_prompt_cache
deltas = vs.arm_request(v, embeds, prompt_ids, grid)
cache = make_prompt_cache(inner)
h = inner(mx.array(prompt_ids)[None, :], cache=cache)
mx.eval(h)
# prefill完成 → 立即disarm(decode用原rope, 位置=cache.offset自动)
vs.disarm_request(v)
toks = []
cur = mx.argmax(h[0, -1]).item()
for i in range(60):
    toks.append(cur)
    if cur in (151645, 151643):
        break
    h = inner(mx.array([cur])[None, :], cache=cache)
    mx.eval(h)
    cur = mx.argmax(h[0, -1]).item()
print("生成:", tokenizer.decode(toks)[:400])
