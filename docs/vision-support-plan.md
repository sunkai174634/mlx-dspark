# mlx-dspark vision support (feat/vision-support branch)

## 目标

让 `mlx-dspark serve` 单端口同时支持文本(草稿加速)和图像(orca视觉塔)请求。

## 方案 (最小侵入)

```
server._chat 检测 image_url
  → mlx-vlm processor 编码图片 → vision_tower 前向 → image_embeds (N×hidden)
  → prompt 构造: "<|vision_start|>" + <|image_pad|>×N + "<|vision_end|>" + text
     (tokenizer.encode 能正确把字面特殊token转成 248053/248056/248054 — 已验证)
  → prompt_ids 传入 Engine.generate 前, 把 image_embeds 挂到 target 的包装 embed 上
  → 图像请求强制走 baseline 路径(无草稿) — 避免碰 verify 循环(lossless 红线)
     文本请求照旧 dspark 30 tok/s
```

## 关键验证记录

- orca 特殊 token: `<|vision_start|>`=248053 `<|image_pad|>`=248056 `<|vision_end|>`=248054
- 字面特殊 token 字符串经 `tok.encode()` 正确转 id ✓
- `apply_chat_template` 接受含特殊 token 字面量的 str content ✓ (返回 str 再 encode)
- Engine.generate → _generate_impl_inner → (mode 分支) 全部最终走 target.embed_tokens

## 改动点

1. `src/mlx_dspark/server.py::_chat` — 图像拦截分支 (~80 行)
2. `src/mlx_dspark/server.py::Engine` — `_vision_embeds` 挂载点 + baseline 强制 (~20 行)
3. `src/mlx_dspark/target.py` — embed_tokens 包装: image_pad id → 挂载表按序取 (~30 行)

## 风险

- 作者在 load.py 警告过 vlm 路线的内存损坏前科 (gemma-4, verify buffer 垃圾) — 
  我们绕开 verify (baseline-only) 规避
- 上游升级需 rebase — 记录在 CHANGELOG
