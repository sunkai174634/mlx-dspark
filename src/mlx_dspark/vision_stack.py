"""Vision stack for mlx-dspark: side-loaded tower + dual-mode rope + embed bypass.

Wires the five verified building blocks onto a loaded Engine so image requests
can flow through the same server as text:

    TowerSidecar  — lazily loads vision_tower weights from the target's shards
    VisionState   — per-request holder: image embeddings + mrope positions
    attach(engine)— replaces FA-layer rope with DualModeRope and wraps
                    embed_tokens with VisionEmbedBypass (both no-ops when
                    disarmed; bit-exact with upstream when idle)
"""
from __future__ import annotations

import base64
import glob
import io
import json
import os
import urllib.request

import mlx.core as mx
import mlx.nn as nn


# --------------------------------------------------------------------------
# building block 1: embed bypass (verified in tests/test_vision_bypass.py)
# --------------------------------------------------------------------------

class VisionEmbedBypass(nn.Module):
    """Wrap embed_tokens: rows whose id == image_pad pull from an armed table."""

    def __init__(self, embed, image_pad_id: int):
        super().__init__()
        self._embed = embed
        self.image_pad_id = image_pad_id
        self._table = None
        self._consumed = 0

    def __call__(self, ids):
        ids = ids if isinstance(ids, mx.array) else mx.array(ids)
        out = self._embed(ids)
        if self._table is not None:
            pad_rows = (ids == self.image_pad_id)          # (...L)
            n_pad = int(pad_rows.sum().item())
            if n_pad:
                take = self._table[self._consumed:self._consumed + n_pad]
                self._consumed += n_pad
                rank = mx.cumsum(pad_rows.astype(mx.int32), axis=-1) - 1
                safe = mx.clip(rank, 0, max(n_pad - 1, 0))
                gathered = take[safe]
                # 广播到out的形状(ids可能2维(1,L)或3维(B,1,L))
                while gathered.ndim < out.ndim:
                    gathered = gathered[None]
                pad_b = pad_rows
                while pad_b.ndim < out.ndim:
                    pad_b = pad_b[..., None] if pad_b.shape[-1] != out.shape[-1] else pad_b[None]
                out = mx.where(pad_rows[..., None], gathered, out)
        return out

    def arm(self, table):
        self._table = table
        self._consumed = 0

    def disarm(self):
        self._table = None
        self._consumed = 0


# --------------------------------------------------------------------------
# building block 2: dual-mode rope (verified in tests/test_dual_rope.py)
# --------------------------------------------------------------------------

class DualModeRope(nn.Module):
    """arm() -> segment-wise mrope over (B,H,L,D); disarm() -> passthrough."""

    def __init__(self, base_rope, mrope_section=(11, 11, 10), base=100000.0):
        super().__init__()
        self._base = base_rope
        self._section = list(mrope_section)
        self._base_hz = base
        self._dim = getattr(base_rope, "dim", 64)
        self._pos = None

    def arm(self, pos_ids_3row):
        self._pos = pos_ids_3row

    def disarm(self):
        self._pos = None

    def __call__(self, x, offset=0):
        if self._pos is None:
            return self._base(x, offset=offset)
        B, H, L, D = x.shape
        half = D // 2
        pos = self._pos.astype(mx.float32)
        inv = self._base_hz ** (-mx.arange(0, half, dtype=mx.float32) / half)
        bounds = [0, self._section[0],
                  self._section[0] + self._section[1], half]
        rows = []
        for si in range(3):
            n = bounds[si + 1] - bounds[si]
            if n <= 0:
                continue
            rows.append(mx.outer(pos[si], inv[bounds[si]:bounds[si + 1]]))
        freqs = mx.concatenate(rows, axis=-1)              # (L, half)
        x32 = x.astype(mx.float32)
        x1, x2 = x32[..., :half], x32[..., half:]
        cos = mx.cos(freqs)[None, None, :, :]
        sin = mx.sin(freqs)[None, None, :, :]
        out = mx.concatenate([x1 * cos - x2 * sin,
                              x1 * sin + x2 * cos], axis=-1)
        return out.astype(x.dtype)


def _extract_image_parts(messages):
    """Return (normalized messages, PIL images). Images may be data URIs or URLs."""
    from PIL import Image
    images = []
    normalized = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            normalized.append(dict(msg))
            continue
        text_parts = []
        for part in content:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    _, encoded = url.split(",", 1)
                    raw = base64.b64decode(encoded + "=" * (-len(encoded) % 4))
                else:
                    with urllib.request.urlopen(url, timeout=60) as response:
                        raw = response.read()
                images.append(Image.open(io.BytesIO(raw)).convert("RGB"))
                text_parts.append("<|vision_start|><|image_pad|><|vision_end|>")
        new_msg = dict(msg)
        new_msg["content"] = "".join(text_parts)
        normalized.append(new_msg)
    return normalized, images


def image_prompt_ids(tokenizer, messages, n_pads):
    """Use the tokenizer's own template; this preserves its exact thinking suffix."""
    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False)
    if isinstance(rendered, dict):
        return list(rendered["input_ids"])
    if isinstance(rendered, str):
        return list(tokenizer.encode(rendered, add_special_tokens=False))
    return list(rendered)


def arm_messages(vision, messages, tokenizer=None):
    normalized, images = _extract_image_parts(messages)
    if not images:
        return None
    if len(images) != 1:
        raise ValueError("vision prototype currently supports exactly one image")
    # Use the same processor path as the sidecar, then replace the single image
    # marker with the exact number of image_pad tokens before tokenization.
    embeds, grid = vision.sidecar.embed(images[0])
    n_pads = int(grid.prod().item()) // 4
    marker = "<|vision_start|><|image_pad|><|vision_end|>"
    normalized[0]["content"] = normalized[0]["content"].replace(
        marker, "<|vision_start|>" + "<|image_pad|>" * n_pads + "<|vision_end|>", 1)
    ids = image_prompt_ids(tokenizer or vision.sidecar.processor.tokenizer,
                           normalized, n_pads)
    # The processor's tokenizer is normally the same instance; fall back to
    # the target tokenizer at the caller if it is not.
    return embeds, grid, normalized, ids


# --------------------------------------------------------------------------
# building block 3: tower sidecar
# --------------------------------------------------------------------------

class TowerSidecar:
    """Lazily instantiate + load the vision tower from the target's shards."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tower = None
        self.processor = None

    def ensure(self):
        if self.tower is not None:
            return
        cfg = json.load(open(os.path.join(self.model_dir, "config.json")))
        from mlx_vlm.models.qwen3_5.vision import VisionModel
        from mlx_vlm.models.qwen3_5.config import VisionConfig
        self.tower = VisionModel(VisionConfig.from_dict(cfg["vision_config"]))
        weights = {}
        for shard in sorted(glob.glob(os.path.join(self.model_dir,
                                                   "*.safetensors"))):
            w = mx.load(shard)
            hit = {k[len("vision_tower."):]: v
                   for k, v in w.items() if k.startswith("vision_tower.")}
            weights.update(hit)
            if hit:
                break  # tower lives in the first shard for our checkpoints
        sanitized = self.tower.sanitize(weights)
        # parameters() is nested at module boundaries; flatten it before
        # comparing against safetensors' dotted keys.
        from mlx.utils import tree_flatten
        expected = dict(tree_flatten(self.tower.parameters()))
        loaded = dict(sanitized)
        missing = [k for k in expected if k not in loaded]
        unexpected = [k for k in loaded if k not in expected]
        if missing or unexpected:
            raise RuntimeError(
                f"vision tower side-load mismatch: missing={len(missing)} "
                f"unexpected={len(unexpected)}; first missing={missing[:3]} "
                f"first unexpected={unexpected[:3]}"
            )
        self.tower.load_weights(list(sanitized.items()), strict=True)
        from mlx_vlm.utils import load_processor
        from pathlib import Path
        self.processor = load_processor(Path(self.model_dir))
        mx.eval(self.tower.parameters())

    def embed(self, image) -> tuple[mx.array, mx.array]:
        """image: PIL.Image -> (embeds (N,hidden), grid_thw)"""
        self.ensure()
        from PIL import Image
        img = image if isinstance(image, Image.Image) else Image.open(image)
        img = img.convert("RGB")
        # Match Qwen3.5's processor/chat template exactly; the closed empty
        # thinking block is part of the prompt even when thinking is disabled.
        formatted = ("<|im_start|>user\n<|vision_start|><|image_pad|>"
                     "<|vision_end|>x<|im_end|>\n<|im_start|>assistant\n"
                     "<think>\n\n</think>\n\n")
        enc = self.processor(text=formatted, images=[img],
                             return_tensors="mlx")
        pv = enc["pixel_values"].astype(
            self.tower.patch_embed.proj.weight.dtype)
        grid = enc["image_grid_thw"]
        hidden, _ = self.tower(pv, grid)
        mx.eval(hidden)
        return hidden, grid


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------

class VisionState:
    """What _chat arms before handing prompt_ids to the engine."""

    def __init__(self):
        self.sidecar: TowerSidecar | None = None
        self.embed_bypass: VisionEmbedBypass | None = None
        self.rope_wrappers: list[DualModeRope] = []
        self.image_pad_id = 248056
        self.mrope_section = (11, 11, 10)
        self.rope_base = 100000.0


_VISION: VisionState | None = None


def get_vision() -> VisionState:
    global _VISION
    if _VISION is None:
        _VISION = VisionState()
    return _VISION


def attach(target, model_dir: str):
    """Replace FA-layer rope + wrap embed_tokens on a loaded Target
    (the object load_target() returns). Both are inert (bit-exact)
    until armed per request."""
    v = get_vision()
    v.sidecar = TowerSidecar(model_dir)

    # read config facts
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    if "image_token_index" in cfg:
        v.image_pad_id = cfg["image_token_index"]
    tc = cfg.get("text_config", cfg)
    rp = tc.get("rope_parameters") or {}
    if "mrope_section" in rp:
        v.mrope_section = tuple(rp["mrope_section"])
    if "rope_theta" in rp:
        v.rope_base = float(rp["rope_theta"])

    inner = target._inner
    mm = inner.model

    # 1) embed bypass
    if not isinstance(mm.embed_tokens, VisionEmbedBypass):
        mm.embed_tokens = VisionEmbedBypass(mm.embed_tokens, v.image_pad_id)
        v.embed_bypass = mm.embed_tokens

    # 2) rope wrappers on every FA (non-linear) layer
    v.rope_wrappers = []
    for layer in mm.layers:
        if getattr(layer, "is_linear", False):
            continue
        attn = layer.self_attn
        if not isinstance(attn.rope, DualModeRope):
            attn.rope = DualModeRope(attn.rope, v.mrope_section, v.rope_base)
        v.rope_wrappers.append(attn.rope)
    return v


def arm_request(v: VisionState, image_embeds, prompt_ids: list[int],
                grid_thw) -> int:
    """Arm all three channels for one image request.
    Returns rope_deltas (int) for decode continuation."""
    # positions
    pos, deltas = _get_rope_index(v, prompt_ids, grid_thw)
    v.embed_bypass.arm(image_embeds)
    for w in v.rope_wrappers:
        w.arm(pos)
    return int(deltas.item()) if hasattr(deltas, "item") else int(deltas)


def disarm_request(v: VisionState):
    if v.embed_bypass is not None:
        v.embed_bypass.disarm()
    for w in v.rope_wrappers:
        w.disarm()


def _get_rope_index(v: VisionState, prompt_ids, grid_thw):
    """Standalone get_rope_index via a fake LanguageModel self."""
    model_dir = v.sidecar.model_dir
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    from mlx_vlm.models.qwen3_5.config import ModelConfig
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    class _FakeSelf:
        pass

    fs = _FakeSelf()
    fs.config = ModelConfig.from_dict(cfg)
    input_ids = mx.array(prompt_ids)[None, :]
    return LanguageModel.get_rope_index(fs, input_ids,
                                        image_grid_thw=mx.array(grid_thw.tolist(),
                                                                dtype=mx.int32))
