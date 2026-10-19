"""Attention parity with upstream math and actual post-RoPE probabilities."""

import pytest
import torch
from torch import nn

from vggt.layers.attention import Attention as UpstreamAttention
from vggt.layers.attention_vis import Attention
from vggt.layers.block_vis import Block
from vggt.layers.rope import RotaryPositionEmbedding2D
from vggt.utils.attention_cache import AttentionCacheManager


@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("qk_norm", [True, False])
def test_capture_preserves_output_and_matches_post_rope_reference(fused, qk_norm):
    torch.manual_seed(23)
    rope = RotaryPositionEmbedding2D()
    attention = Attention(32, num_heads=4, qk_norm=qk_norm, fused_attn=fused, rope=rope).eval()
    upstream = UpstreamAttention(32, num_heads=4, qk_norm=qk_norm,
                                 fused_attn=fused, rope=rope).eval()
    upstream.load_state_dict(attention.state_dict(), strict=True)
    inputs = torch.randn(2, 18, 32)
    positions = torch.randint(0, 5, (2, 18, 2))
    before = attention(inputs, positions)
    torch.testing.assert_close(before, upstream(inputs, positions))
    manager = AttentionCacheManager(query_chunk_size=4)
    manager.enable_caching(max_heads=2)
    attention._cache_manager, attention._layer_name = manager, "global_block_23"
    after = attention(inputs, positions)
    torch.testing.assert_close(after, before, rtol=0, atol=0)

    qkv = attention.qkv(inputs).reshape(2, 18, 3, 4, 8).permute(2, 0, 3, 1, 4)
    query, key, _ = qkv.unbind(0)
    query, key = rope(attention.q_norm(query), positions), rope(attention.k_norm(key), positions)
    reference = ((query[:, :2].float() * attention.scale)
                 @ key[:, :2].float().transpose(-2, -1)).softmax(-1).mean(1, keepdim=True)
    actual = manager.get_attention("global_block_23")["attention"]
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
    assert not actual.requires_grad


def test_sdpa_is_default_even_with_capture_disabled():
    assert Attention(32, 4).fused_attn
    assert Block(dim=32, num_heads=4).attn.fused_attn


def test_vggt_models_own_independent_managers(monkeypatch):
    # Patch expensive heads and embedding construction, keeping the real VGGT
    # constructor and its actual manager wiring under test.
    from vggt.models import vggt_vis

    class SmallAggregator(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def set_attention_cache_manager(self, manager):
            self.manager = manager

    monkeypatch.setattr(vggt_vis, "Aggregator", SmallAggregator)
    for name in ("CameraHead", "DPTHead", "TrackHead"):
        monkeypatch.setattr(vggt_vis, name, lambda **kwargs: nn.Identity())
    first, second = vggt_vis.VGGT(), vggt_vis.VGGT()
    first.enable_attention_caching()
    assert first.attention_cache is not second.attention_cache
    assert first.aggregator.manager is first.attention_cache
    assert second.aggregator.manager is second.attention_cache
    assert first.attention_cache.enabled and not second.attention_cache.enabled
    first.attention_cache.store_attention("global_block_23", torch.ones(1, 1, 3, 3), {})
    second.clear_attention_cache()
    assert first.get_cached_attention()


@pytest.mark.parametrize("failure", ["shape", "channels", "patch_grid", "embedding"])
def test_failed_forward_cannot_expose_previous_attention(failure, monkeypatch):
    from vggt.models.aggregator_vis import Aggregator

    aggregator = Aggregator(img_size=42, embed_dim=32, depth=1, num_heads=4,
                            patch_embed="conv", mlp_ratio=2).eval()
    cache = AttentionCacheManager(query_chunk_size=4)
    cache.enable_caching(["global_block_0"])
    aggregator.set_attention_cache_manager(cache)
    images = torch.rand(1, 2, 3, 28, 42)
    with torch.inference_mode():
        aggregator(images)
    assert cache.get_attention("global_block_0") is not None
    previous_run = cache.current_run_id

    with monkeypatch.context() as scoped:
        invalid = images
        if failure == "shape":
            invalid = images[0]
        elif failure == "channels":
            invalid = images[:, :, :1]
        elif failure == "patch_grid":
            invalid = torch.zeros(1, 2, 3, 29, 42)
        else:
            def failed_embedding(*args, **kwargs):
                raise RuntimeError("injected embedding failure")

            scoped.setattr(aggregator.patch_embed, "forward", failed_embedding)
        with torch.inference_mode(), pytest.raises((ValueError, AssertionError, RuntimeError)):
            aggregator(invalid)
        assert cache.get_attention("global_block_0") is None
        assert cache.get_all_cached_attention() == {}
        assert cache.model_info == {}
        assert cache.extract_patch_attention(0, 0, 0, "global_block_0") is None

    with torch.inference_mode():
        aggregator(images)
    assert cache.current_run_id != previous_run
    restored = cache.extract_patch_attention(1, 1, 2, "global_block_0")
    assert len(restored["attention_maps_row"]) == 2
    assert restored["attention_maps_row"][0].shape == (2, 3)


def test_early_forward_failure_preserves_explicit_pending_run_id():
    from vggt.models.aggregator_vis import Aggregator

    aggregator = Aggregator(img_size=42, embed_dim=32, depth=1, num_heads=4,
                            patch_embed="conv", mlp_ratio=2).eval()
    cache = AttentionCacheManager()
    cache.enable_caching(["global_block_0"])
    aggregator.set_attention_cache_manager(cache)
    cache.start_new_run("caller-selected-run")
    with pytest.raises(ValueError, match="shape"):
        aggregator(torch.rand(2, 3, 28, 42))
    assert cache.current_run_id == "caller-selected-run"
    assert cache.model_info == {} and not cache.cache
    with torch.inference_mode():
        aggregator(torch.rand(1, 2, 3, 28, 42))
    assert cache.get_attention("global_block_0")["run_id"] == "caller-selected-run"
