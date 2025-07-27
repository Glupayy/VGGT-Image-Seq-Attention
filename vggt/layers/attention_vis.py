# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/models/vision_transformer.py

import logging
import os
import warnings

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

# 导入attention缓存管理器
try:
    from attention_cache import get_attention_cache
except ImportError:
    # 如果导入失败，创建一个虚拟的缓存管理器
    class DummyCache:
        def store_attention(self, *args, **kwargs):
            pass
    def get_attention_cache():
        return DummyCache()

XFORMERS_AVAILABLE = False


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        fused_attn: bool = False,  # use F.scaled_dot_product_attention or not
        rope=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        
        # 用于attention缓存的属性（由aggregator设置）
        self._cache_manager = None
        self._layer_name = None

    def forward(self, x: Tensor, pos=None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
            # 注意：fused attention无法直接获取attention矩阵，需要重新计算用于缓存
            if self._should_cache_attention():
                with torch.no_grad():
                    q_scaled = q * self.scale
                    attn_for_cache = q_scaled @ k.transpose(-2, -1)
                    attn_for_cache = attn_for_cache.softmax(dim=-1)
                    self._store_attention(attn_for_cache, (B, N, C))
        else:
            print("⚠️ 注意：未启用fused attention")
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            
            # 🔥 新增：缓存attention矩阵（如果启用）
            if self._should_cache_attention():
                self._store_attention(attn, (B, N, C))
            
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def _should_cache_attention(self) -> bool:
        """检查是否应该缓存attention"""
        if self._cache_manager is None or self._layer_name is None:
            return False
        return (self._cache_manager.enabled and 
                self._layer_name in self._cache_manager.target_layers)
    
    def _store_attention(self, attn: Tensor, tokens_shape: tuple):
        """存储attention到缓存管理器"""
        if self._cache_manager is None or self._layer_name is None:
            return
        
        # 准备层信息
        B, N, C = tokens_shape
        layer_info = {
            'num_heads': self.num_heads,
            'tokens_shape': (B, N, C),
            'head_dim': self.head_dim,
            'scale': self.scale
        }
        
        # 🔥 关键调试：验证我们是否缓存了正确的attention
        B, N, C = tokens_shape
        print(f"🔍 缓存Attention调试 - {self._layer_name}:")
        print(f"   Attention形状: {attn.shape}")
        print(f"   Token形状: B={B}, N={N}, C={C}")
        
        # 支持多种可能的结构
        if N == 6246:  # 实际观察到的Global Attention (392×518图像)
            print(f"   ✅ 这是实际的Global Attention! (N=6246, 6帧×1041tokens)")
        elif N == 8244:  # 理论Global Attention (518×518图像)
            print(f"   ✅ 这是理论的Global Attention! (N=8244, 6帧×1374tokens)")
        elif N == 1374:  # Frame Attention (518×518单图像)
            print(f"   ⚠️  这是Frame Attention! (N=1374)")
        elif N == 1041:  # Frame Attention (392×518单图像)
            print(f"   ⚠️  这是实际Frame Attention! (N=1041)")
        else:
            print(f"   ❓ 其他Attention类型: N={N} (将尝试动态推断)")
        
        # 存储到缓存管理器
        self._cache_manager.store_attention(
            self._layer_name, 
            attn, 
            layer_info
        )


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None, pos=None) -> Tensor:
        assert pos is None
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
