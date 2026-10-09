# Stage 2 progress — mrope injection path (2026-10-09)

## 已验证积木 (4/5)

1. **VisionEmbedBypass** (embed旁路) ✓ — cumsum-gather, 逻辑测试过
2. **processor编码** ✓ — 4680块 → 1170 pads
3. **vision_tower侧载** ✓ — mx.load直接读分片1 + VisionModel.sanitize
   + load_weights(strict=False) 全命中; 前向输出 (1170, 5120) float32
4. **get_rope_index** ✓ — LanguageModel.get_rope_index 可用 FakeSelf(config)
   独立调用; 输出 (3,1,1181): 文本段三行相同 / pad段三行不同(t/h/w) /
   尾部回同步; rope_deltas=-1125 供后续 decode 段接续

## 第5块: rope应用 (当前)

- dspark 路径的 FA attention = mlx-lm qwen3_next.Qwen3NextAttention,
  其 rope = standard nn.RoPE, 调用 `self.rope(q, offset=cache.offset)` 标量
- mlx-lm rope_utils 的 "mrope" 分支实际返回普通 RoPE (纯文本下三行相同,
  退化正确; 图像下错)
- `mx.fast.rope` 不支持 per-token offset (只接受标量/1元素向量) — 实测
- **mlx-vlm 有完整可复用的 MRoPERotaryEmbedding** (rope_utils.py:1158):
  - `__call__(x, position_ids) -> cos, sin`
  - `apply_rotary(q, k, position_ids)` — 含 Metal fused 编译路径
  - style="interleaved" 与 Qwen3_5 匹配

## 注入设计 (最小侵入)

包装 mlx-lm Attention 的 rope 调用:
- 平时(文本/decode): 原 nn.RoPE(offset=cache.offset) — 不动
- 图像 prefill: arm 状态下换成 MRoPE.apply_rotary(q, k, pos_ids_3row)
  + rope_deltas 接续 (decode 段位置 = arange + delta, 三行同步 →
  可继续用原 rope(offset=offset+delta), 零持续开销!)

关键: decode/verify 段位置三行同步 → 不需要动草稿路径 (lossless 保全)

## 下一步

1. 写 AttentionRopeWrapper + 单元测试 (arm 后 mrope, disarm 原样)
2. server._chat 图像分支完整接线
3. 小图端到端 → 与 qfast-vlm(:8190) 输出对比数值一致性
