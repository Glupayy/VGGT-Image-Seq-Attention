"""Numerical and geometry regressions; no checkpoints or network required."""

import pytest
import torch

from vggt.utils.attention_cache import AttentionCacheManager, get_attention_cache


def make_cache(batch=2, frames=3, height=2, width=3, offset=2, max_heads=None):
    cache = AttentionCacheManager(query_chunk_size=5)
    cache.enable_caching(max_heads=max_heads)
    cache.begin_forward(batch_size=batch, num_frames=frames, patch_h=height,
                        patch_w=width, patch_start_idx=offset)
    return cache


@pytest.mark.parametrize("frames,height,width,offset", [(1, 2, 3, 0), (2, 3, 2, 2), (6, 1, 4, 5)])
@pytest.mark.parametrize("max_heads", [None, 1, 99])
def test_exact_outgoing_and_incoming_mapping(frames, height, width, offset, max_heads):
    torch.manual_seed(9)
    cache = make_cache(frames=frames, height=height, width=width, offset=offset,
                       max_heads=max_heads)
    per_frame = offset + height * width
    tokens = frames * per_frame
    probabilities = torch.randn(2, 3, tokens, tokens).softmax(dim=-1)
    cache.store_attention("global_block_23", probabilities, {"tokens_shape": (2, tokens, 12)})
    selected = 3 if max_heads is None else min(3, max_heads)
    reference = probabilities[:, :selected].mean(dim=1)
    result = cache.extract_patch_attention(frames - 1, height - 1, width - 1, batch_idx=1)
    source = (frames - 1) * per_frame + offset + (height - 1) * width + width - 1
    assert result["source_token_idx"] == source
    assert result["selected_num_heads"] == selected
    assert result["original_num_heads"] == 3
    assert result["attention_maps"] is result["attention_maps_row"]
    for frame in range(frames):
        start = frame * per_frame + offset
        end = start + height * width
        torch.testing.assert_close(result["attention_maps_row"][frame],
                                   reference[1, source, start:end].reshape(height, width))
        torch.testing.assert_close(result["attention_maps_col"][frame],
                                   reference[1, start:end, source].reshape(height, width))
    # Values remain probabilities from the original matrix, including the mass
    # excluded for special tokens; no per-image or column normalization occurs.
    assert result["attention_strengths_col"] == pytest.approx(
        [item.sum().item() for item in result["attention_maps_col"]])


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("max_heads", [None, 2, 20])
def test_chunked_capture_matches_float32_reference(dtype, max_heads, monkeypatch):
    torch.manual_seed(14)
    cache = make_cache(max_heads=max_heads)
    tokens = 24
    query = torch.randn(2, 4, tokens, 8, dtype=dtype, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    heads = 4 if max_heads is None else min(4, max_heads)
    reference = ((query.detach()[:, :heads].float() @ key.detach()[:, :heads].float()
                  .transpose(-2, -1)) * (8 ** -0.5)).softmax(-1).mean(1, keepdim=True)
    original_matmul = torch.matmul
    query_sizes = []

    def observed_matmul(left, right):
        query_sizes.append(left.shape[-2])
        return original_matmul(left, right)

    monkeypatch.setattr(torch, "matmul", observed_matmul)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        cache.capture_qk("global_block_23", query, key)
    cached = cache.get_attention("global_block_23")
    actual = cached["attention"]
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-7)
    assert len(query_sizes) == 5 and max(query_sizes) <= 5
    assert actual.dtype == torch.float32 and actual.device.type == "cpu"
    assert not actual.requires_grad and actual.grad_fn is None
    assert cached["num_heads"] == 1
    torch.testing.assert_close(actual.sum(-1), torch.ones(2, 1, tokens))
    assert cache.get_cache_info()["memory_bytes"] == 2 * tokens * tokens * 4


def test_selected_head_mean_owns_only_reduced_storage():
    cache = make_cache(max_heads=2)
    source = torch.rand(2, 8, 24, 24, requires_grad=True)
    cache.store_attention("global_block_23", source, {})
    stored = cache.get_attention("global_block_23")["attention"]
    assert stored.untyped_storage().nbytes() == stored.numel() * stored.element_size()
    assert stored.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    torch.testing.assert_close(stored, source.detach()[:, :2].mean(1, keepdim=True))


@pytest.mark.parametrize("kwargs", [
    {"source_img_idx": -1}, {"source_img_idx": 3}, {"patch_y": -1}, {"patch_y": 2},
    {"patch_x": -1}, {"patch_x": 3}, {"batch_idx": -1}, {"batch_idx": 2},
    {"patch_x": 1.5}, {"patch_y": True},
])
def test_invalid_patch_and_batch_indices_rejected(kwargs):
    cache = make_cache()
    cache.store_attention("global_block_23", torch.ones(2, 1, 24, 24) / 24, {})
    arguments = dict(source_img_idx=0, patch_y=0, patch_x=0, batch_idx=0)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        cache.extract_patch_attention(**arguments)


def test_each_forward_replaces_geometry_and_invalidates_old_cache():
    cache = make_cache()
    cache.store_attention("global_block_23", torch.ones(2, 1, 24, 24) / 24, {})
    previous_run = cache.current_run_id
    cache.begin_forward(batch_size=1, num_frames=2, patch_h=3, patch_w=2,
                        patch_start_idx=1, patch_size=7)
    assert not cache.cache and cache.current_run_id != previous_run
    assert cache.model_info["tokens_per_frame"] == 7
    cache.store_attention("global_block_23", torch.ones(1, 1, 14, 14) / 14, {})
    result = cache.extract_patch_attention(1, 2, 1)
    assert len(result["attention_maps_row"]) == 2
    assert result["attention_maps_row"][0].shape == (3, 2)
    assert result["patch_size"] == 7
    cache.clear_cache()
    assert cache.model_info == {} and cache.get_attention("global_block_23") is None


def test_explicit_run_id_and_cache_controls():
    cache = AttentionCacheManager()
    cache.enable_caching()
    assert cache.target_layers == ["global_block_23"] and cache.max_heads is None
    cache.start_new_run("selected-sequence")
    cache.begin_forward(batch_size=1, num_frames=1, patch_h=1, patch_w=1)
    assert cache.current_run_id == "selected-sequence"
    cache.capture_qk("global_block_0", torch.ones(1, 1, 6, 4), torch.ones(1, 1, 6, 4))
    assert not cache.cache
    cache.disable_caching()
    cache.store_attention("global_block_23", torch.ones(1, 1, 6, 6), {})
    assert not cache.cache
    cache.enable_caching([])
    assert cache.target_layers == []


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_head_limits_rejected(value):
    with pytest.raises(ValueError, match="max_heads"):
        AttentionCacheManager().enable_caching(max_heads=value)


def test_geometry_required_and_mismatches_rejected():
    cache = AttentionCacheManager()
    cache.enable_caching()
    cache.store_attention("global_block_23", torch.ones(1, 1, 4, 4), {})
    with pytest.raises(ValueError, match="geometry is missing"):
        cache.extract_patch_attention(0, 0, 0)
    cache = make_cache()
    cache.store_attention("global_block_23", torch.ones(1, 1, 4, 4), {})
    with pytest.raises(ValueError, match="shape does not match"):
        cache.extract_patch_attention(0, 0, 0)


def test_frame_layer_cannot_be_misinterpreted_as_global():
    cache = make_cache()
    cache.enable_caching(["frame_block_0", "global_block_23"])
    cache.set_model_info(batch_size=2, num_frames=3, patch_h=2, patch_w=3, patch_start_idx=2)
    cache.store_attention("frame_block_0", torch.ones(6, 1, 8, 8) / 8, {})
    with pytest.raises(ValueError, match="global attention layer"):
        cache.extract_patch_attention(0, 0, 0, "frame_block_0")


def test_legacy_import_and_independent_managers():
    from attention_cache import AttentionCacheManager as LegacyManager

    assert LegacyManager is AttentionCacheManager
    first, second = make_cache(), make_cache()
    first.store_attention("global_block_23", torch.ones(2, 1, 24, 24), {})
    second.clear_cache()
    assert first.cache and first is not second
    assert first is not get_attention_cache()
