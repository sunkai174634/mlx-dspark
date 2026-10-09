"""交叉库一致性: 同prompt下 dspark(mlx-lm)路径 vs mlx-vlm路径 的logits"""
import os, sys
sys.path.insert(0, os.path.expanduser('~/Projects/mlx-dspark/src'))
import mlx.core as mx
import numpy as np

MODEL = os.path.expanduser("~/Models/VLM/Qwen/Qwen3.8-27B-4bit-crack-orcarouter")

# 纯文本prompt
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok.encode("你好，请简单介绍一下自己。", add_special_tokens=False)
print("ids数:", len(ids))
print("prompt ids:", len(ids))

# --- 路径A: mlx-lm (dspark所用) ---
from mlx_lm.utils import load as lm_load
m_lm, t_lm = lm_load(MODEL)
from mlx_lm.models.cache import make_prompt_cache
cache = make_prompt_cache(m_lm)
h = m_lm(mx.array(ids)[None,:], cache=cache)
logits_lm = h[0, -1, :]
mx.eval(logits_lm)
print("mlx-lm last-token logits: 前5 =", logits_lm[:5].tolist())
del m_lm, cache
import gc; gc.collect(); mx.clear_cache()

# --- 路径B: mlx-vlm ---
from mlx_vlm import load as vlm_load
m_vlm, proc = vlm_load(MODEL)
h2 = m_vlm(mx.array(ids)[None,:])
if hasattr(h2, "logits"): h2 = h2.logits
logits_vlm = h2[0, -1, :]
mx.eval(logits_vlm)
print("mlx-vlm last-token logits: 前5 =", logits_vlm[:5].tolist())

top_lm = mx.argmax(logits_lm).item(); top_vl = mx.argmax(logits_vlm).item()
print("top1一致:", top_lm == top_vl, "| 分别:", tok.decode([top_lm]), "vs", tok.decode([top_vl]))
d = mx.abs(logits_lm - logits_vlm).max().item()
print("max |Δlogit|:", d, "| logits范围:", mx.abs(logits_vlm).max().item())

# top10重合度
t_lm = mx.argsort(-logits_lm)[:10].tolist()
t_vl = mx.argsort(-logits_vlm)[:10].tolist()
print("top10 mlx-lm:", [tok.decode([t]) for t in t_lm])
print("top10 mlx-vlm:", [tok.decode([t]) for t in t_vl])
print("重合:", len(set(t_lm) & set(t_vl)), "/10")
