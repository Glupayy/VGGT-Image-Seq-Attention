"""Compatibility imports; the implementation lives in the installable package."""

from vggt.utils.attention_cache import (
    AttentionCacheManager,
    DEFAULT_TARGET_LAYERS,
    get_attention_cache,
    global_attention_cache,
)

__all__ = ["AttentionCacheManager", "DEFAULT_TARGET_LAYERS", "get_attention_cache",
           "global_attention_cache"]
