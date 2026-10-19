# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the license in LICENSE.txt in the repository root.

"""VGGT attention with optional, chunked probability capture."""

import torch
from torch import Tensor, nn
import torch.nn.functional as F

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = True,
                 proj_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0, norm_layer: nn.Module = nn.LayerNorm,
                 qk_norm: bool = False, fused_attn: bool = True, rope=None):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self._cache_manager = None
        self._layer_name = None

    def forward(self, x: Tensor, pos=None) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q, k = self.rope(q, pos), self.rope(k, pos)

        if self.fused_attn:
            output = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        else:
            probabilities = ((q * self.scale) @ k.transpose(-2, -1)).softmax(dim=-1)
            output = self.attn_drop(probabilities) @ v

        if self._should_cache_attention():
            # Capture pre-dropout weights from transformed Q/K, with float32
            # softmax and a bounded query block instead of complete head maps.
            self._cache_manager.capture_qk(
                self._layer_name, q, k,
                {"num_heads": self.num_heads, "tokens_shape": (batch, tokens, channels),
                 "head_dim": self.head_dim, "scale": self.scale},
                scale=self.scale,
            )
        output = output.transpose(1, 2).reshape(batch, tokens, channels)
        return self.proj_drop(self.proj(output))

    def _should_cache_attention(self):
        return (self._cache_manager is not None and self._layer_name is not None
                and self._cache_manager.enabled
                and self._layer_name in self._cache_manager.target_layers)

    def _store_attention(self, attention: Tensor, tokens_shape: tuple):
        """Compatibility hook for callers that already have probabilities."""
        if self._should_cache_attention():
            self._cache_manager.store_attention(
                self._layer_name, attention,
                {"num_heads": self.num_heads, "tokens_shape": tokens_shape,
                 "head_dim": self.head_dim, "scale": self.scale},
            )


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        if attn_bias is not None:
            raise ValueError("nested attention biases require xFormers, which is not enabled")
        return super().forward(x, pos=pos)
