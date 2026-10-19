"""Attention capture with exact patch indexing and bounded temporary memory.

Cached tensors are selected-head means in CPU float32 with shape [B, 1, N, N].
``max_heads=None`` includes every head; positive K selects the first K heads.
Probabilities are captured after Q/K normalization and RoPE, before dropout.
Incoming columns are not renormalized: they do not generally sum to one.
"""

from typing import Any, Dict, List, Optional
from uuid import uuid4

import torch

DEFAULT_TARGET_LAYERS = ("global_block_23",)


class AttentionCacheManager:
    """Own one model's attention cache; geometry comes from its actual input."""

    def __init__(self, query_chunk_size: int = 128):
        self.enabled = False
        self.cache = {}
        self.target_layers = []
        self.max_heads = None
        self.current_run_id = None
        self.query_chunk_size = self._positive_int(query_chunk_size, "query_chunk_size")
        self.model_info = {}
        self._pending_run = False

    @staticmethod
    def _positive_int(value, name):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    def enable_caching(self, target_layers: Optional[List[str]] = None,
                       max_heads: Optional[int] = None):
        if max_heads is not None:
            self._positive_int(max_heads, "max_heads")
        layers = list(DEFAULT_TARGET_LAYERS if target_layers is None else target_layers)
        if not all(isinstance(layer, str) and layer for layer in layers):
            raise ValueError("target_layers must contain nonempty layer names")
        self.enabled = True
        self.target_layers = layers
        self.max_heads = max_heads
        self.clear_cache()

    def disable_caching(self):
        self.enabled = False
        self.clear_cache()

    def clear_cache(self):
        """Release tensors and geometry so future inputs cannot reuse it."""
        self.invalidate_for_forward()
        self._pending_run = False

    def invalidate_for_forward(self):
        """Drop previous results before input validation or embedding can fail.

        An explicitly prepared run ID remains pending until valid geometry is
        available, so a failed input does not silently discard a caller's ID.
        """
        self.cache.clear()
        self.model_info = {}

    def start_new_run(self, run_id: Optional[str] = None):
        self.clear_cache()
        self.current_run_id = run_id or f"run_{uuid4().hex}"
        self._pending_run = True

    def begin_forward(self, **model_info):
        """Refresh each forward's geometry, preserving a caller's new run ID."""
        if not self._pending_run:
            self.start_new_run()
        self._pending_run = False
        self.set_model_info(**model_info)

    def set_model_info(self, *, batch_size: int, num_frames: int, patch_h: int,
                       patch_w: int, patch_size: int = 14, patch_start_idx: int = 5):
        """Set geometry from input and embeddings, without guessing from N."""
        for name, value in (("batch_size", batch_size), ("num_frames", num_frames),
                            ("patch_h", patch_h), ("patch_w", patch_w),
                            ("patch_size", patch_size)):
            self._positive_int(value, name)
        if (isinstance(patch_start_idx, bool) or not isinstance(patch_start_idx, int)
                or patch_start_idx < 0):
            raise ValueError("patch_start_idx must be a nonnegative integer")
        self.model_info = {
            "batch_size": batch_size, "num_frames": num_frames,
            "patch_h": patch_h, "patch_w": patch_w, "patch_size": patch_size,
            "patch_start_idx": patch_start_idx,
            "tokens_per_frame": patch_start_idx + patch_h * patch_w,
        }

    def _selected_heads(self, total):
        return total if self.max_heads is None else min(total, self.max_heads)

    def _should_store(self, layer_name):
        return self.enabled and layer_name in self.target_layers

    def _store_mean(self, layer_name, mean_attention, layer_info, original_heads, selected_heads):
        info = dict(layer_info)
        info.update(self.model_info)
        info["attention_type"] = "frame" if layer_name.startswith("frame_block_") else "global"
        self.cache[layer_name] = {
            "attention": mean_attention, "num_heads": 1,
            "selected_num_heads": selected_heads, "original_num_heads": original_heads,
            "head_reduction": "mean", "tokens_shape": layer_info.get("tokens_shape"),
            "layer_info": info, "model_info": self.model_info.copy(),
            "run_id": self.current_run_id,
        }

    @torch.no_grad()
    def capture_qk(self, layer_name: str, query: torch.Tensor, key: torch.Tensor,
                   layer_info: Optional[Dict[str, Any]] = None, scale: Optional[float] = None):
        """Capture probabilities in [B, heads, chunk, N] query blocks.

        Q and K must already include normalization and positional rotation.
        Write the head-reduced output directly to CPU, without an accelerator
        allocation of the complete multihead N x N attention matrix.
        """
        if not self._should_store(layer_name):
            return
        if query.ndim != 4 or query.shape != key.shape:
            raise ValueError("query and key must have matching [B, heads, N, dim] shapes")
        if query.device != key.device:
            raise ValueError("query and key must be on the same device")
        batch, total_heads, tokens, head_dim = query.shape
        if min(batch, total_heads, tokens, head_dim) <= 0:
            raise ValueError("query and key dimensions must be nonzero")
        selected = self._selected_heads(total_heads)
        scale = head_dim ** -0.5 if scale is None else scale
        output = torch.empty(batch, 1, tokens, tokens, dtype=torch.float32, device="cpu")
        # An outer bf16/fp16 autocast must not lower matmul/softmax precision.
        with torch.autocast(device_type=query.device.type, enabled=False):
            key_float = key[:, :selected].detach().float()
            for start in range(0, tokens, self.query_chunk_size):
                end = min(start + self.query_chunk_size, tokens)
                query_float = query[:, :selected, start:end].detach().float()
                scores = torch.matmul(query_float, key_float.transpose(-2, -1)) * scale
                probabilities = torch.softmax(scores, dim=-1)
                output[:, :, start:end].copy_(probabilities.mean(dim=1, keepdim=True).cpu())
        self._store_mean(layer_name, output, layer_info or {}, total_heads, selected)

    @torch.no_grad()
    def store_attention(self, layer_name: str, attention_tensor: torch.Tensor,
                        layer_info: Dict[str, Any]):
        """Compatibility entry point for already normalized weights."""
        if not self._should_store(layer_name):
            return
        if attention_tensor.ndim != 4 or attention_tensor.shape[-2] != attention_tensor.shape[-1]:
            raise ValueError("attention must have shape [B, heads, N, N]")
        total_heads = attention_tensor.shape[1]
        if total_heads < 1:
            raise ValueError("attention must include at least one head")
        selected = self._selected_heads(total_heads)
        # Reduce before transfer; mean owns its storage rather than retaining
        # a slice into the original complete multihead tensor.
        reduced = attention_tensor.detach()[:, :selected].mean(dim=1, keepdim=True,
                                                               dtype=torch.float32).cpu()
        self._store_mean(layer_name, reduced, layer_info, total_heads, selected)

    def get_attention(self, layer_name: str) -> Optional[Dict[str, Any]]:
        return self.cache.get(layer_name)

    def get_all_cached_attention(self) -> Dict[str, Dict[str, Any]]:
        return self.cache.copy()

    @staticmethod
    def _check_index(value, size, name):
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < size:
            raise ValueError(f"{name} must be an integer in [0, {size})")

    def extract_patch_attention(self, source_img_idx: int, patch_y: int, patch_x: int,
                                layer_name: str = "global_block_23", batch_idx: int = 0):
        """Return outgoing rows and incoming columns, excluding special tokens."""
        data = self.get_attention(layer_name)
        if data is None:
            return None
        if data["layer_info"]["attention_type"] != "global":
            raise ValueError("cross-frame patch extraction requires a global attention layer")
        info = data["model_info"]
        required = ("batch_size", "num_frames", "patch_h", "patch_w", "tokens_per_frame",
                    "patch_start_idx", "patch_size")
        if any(name not in info for name in required):
            raise ValueError("patch geometry is missing; call set_model_info before capturing attention")
        self._check_index(batch_idx, info["batch_size"], "batch_idx")
        self._check_index(source_img_idx, info["num_frames"], "source_img_idx")
        self._check_index(patch_y, info["patch_h"], "patch_y")
        self._check_index(patch_x, info["patch_w"], "patch_x")
        attention = data["attention"]
        expected = info["num_frames"] * info["tokens_per_frame"]
        if attention.shape[0] != info["batch_size"] or attention.shape[-2:] != (expected, expected):
            raise ValueError("cached attention shape does not match the captured input geometry")
        token = (source_img_idx * info["tokens_per_frame"] + info["patch_start_idx"]
                 + patch_y * info["patch_w"] + patch_x)
        # Slice first, avoiding another N x N allocation on every click.
        outgoing = attention[batch_idx, :, token, :].mean(dim=0)
        incoming = attention[batch_idx, :, :, token].mean(dim=0)

        def split_maps(values):
            maps = []
            for frame in range(info["num_frames"]):
                start = frame * info["tokens_per_frame"] + info["patch_start_idx"]
                maps.append(values[start:start + info["patch_h"] * info["patch_w"]]
                            .reshape(info["patch_h"], info["patch_w"]))
            return maps, [item.sum().item() for item in maps]

        rows, row_strengths = split_maps(outgoing)
        columns, column_strengths = split_maps(incoming)
        return {
            "source_img_idx": source_img_idx, "source_patch": (patch_y, patch_x),
            "source_token_idx": token, "batch_idx": batch_idx,
            "attention_maps_row": rows, "attention_strengths_row": row_strengths,
            "attention_maps_col": columns, "attention_strengths_col": column_strengths,
            "attention_maps": rows, "attention_strengths": row_strengths,
            "layer_name": layer_name, "patch_size": info["patch_size"],
            "run_id": data["run_id"], "selected_num_heads": data["selected_num_heads"],
            "original_num_heads": data["original_num_heads"],
        }

    def _estimate_memory_usage(self):
        total = sum(item["attention"].numel() * item["attention"].element_size()
                    for item in self.cache.values())
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if total < 1024 or unit == "TB":
                return f"{total:.1f}{unit}"
            total /= 1024

    def get_cache_info(self):
        return {
            "enabled": self.enabled, "target_layers": self.target_layers.copy(),
            "max_heads": self.max_heads, "query_chunk_size": self.query_chunk_size,
            "current_run_id": self.current_run_id, "cached_layers": list(self.cache),
            "model_info": self.model_info.copy(), "memory_usage": self._estimate_memory_usage(),
            "memory_bytes": sum(item["attention"].numel() * item["attention"].element_size()
                                for item in self.cache.values()),
        }

    def __repr__(self):
        return (f"AttentionCacheManager(enabled={self.enabled}, layers={len(self.cache)}, "
                f"memory={self._estimate_memory_usage()})")


# Old standalone scripts may still request this convenience singleton. Each
# VGGT model constructs its own manager and does not share this cache.
global_attention_cache = AttentionCacheManager()


def get_attention_cache() -> AttentionCacheManager:
    return global_attention_cache
