# Validation record

Release preparation date: 2026-10-19 (Asia/Shanghai).

## Local result

**89 tests passed.** The native desktop check, offline export, wheel build, source
archive build, and installed-wheel export from outside the checkout passed.
The extracted source archive was also tested independently; its tests, scripts,
and six example images are included. Neither distribution contains model weights.

Test environment: macOS arm64, Python 3.11.4, torch 2.11.0, torchvision 0.26.0,
NumPy 1.24.3, Pillow 12.0.0, matplotlib 3.10.8, huggingface_hub 1.2.4,
safetensors 0.5.3, einops 0.8.2. CUDA execution was not tested on this machine.

## Automated checks

The CPU suite exercises actual project components with small tensors and tiny
random models. It does not download weights:

- Cached probabilities against independent full-matrix references, including
  Q/K normalization, 2D RoPE, float32 and bfloat16 autocast.
- Cached versus uncached layer/aggregator output parity.
- Batch selection, 1/2/6 views, rectangular grids and custom special-token offsets.
- Exact outgoing row / incoming column extraction and invalid-coordinate rejection.
- Head averaging, first-K selection, bounded query chunks and per-model isolation.
- Cache invalidation when input shape or run changes, including failed forwards.
- Strict checkpoint loading: matching weights round-trip; wrong weights fail through all maintained entries.
- GUI letterboxing, scaled coordinates, negative/edge clicks and blank grid cells.
- Offline end-to-end PNG / NPZ / JSON export and input validation.
- Actual frame-extraction shell behavior, including paths with whitespace.

Run `python -m pytest -q`. GitHub Actions is configured for CPU tests, offline
exports and package builds on Python 3.10 and 3.12. Remote CI has not run as part
of this local preparation.

## Desktop check

A native macOS Tk session was opened and updated successfully. The check created
the viewer, populated a two-view grid, verified scaled click coordinates, opened
an attention result window, and checked that PNG export retained the original
render size. UI operations stayed on the main thread. A second native end-to-end run used the
verified official weights: background loading/inference, a canvas click at patch
`(14, 18)`, the result window and full-resolution export all passed. Its outgoing
and incoming arrays match the published pretrained sample exactly (maximum
difference 0). The native high-resolution export was 2160 × 1570 pixels.

The installed wheel was also run from `/private/tmp`, outside the checkout, with
the official local checkpoint. Exported images and both probability arrays match
the independent validation path exactly (maximum difference 0).

[Entry-point integration record](assets/integration-validation.json)

## Pretrained model check

The reproducible script is `scripts/validate_pretrained.py`. It runs the official
checkpoint through the real aggregator twice (cache off/on), compares outputs,
checks full row sums and finite values, verifies selected patch probabilities,
and exports the numerical evidence and comparison image.

Checkpoint revision: `860abec7937da0a4c03c41d3c269c366e82abdf9`.
Expected `model.safetensors` SHA256:
`f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`.

**PASS — official checkpoint verified and loaded with `strict=True`.**

Inputs: `temp_images/image_000.jpg` and `temp_images/image_005.jpg`, preprocessed
shape `[2, 3, 392, 518]`; patch grid `28 × 37`; source view 0, patch `(14, 18)`.
The cache averaged all 16 heads of `global_block_23`.

| Check | Observed result |
| --- | --- |
| Weight file integrity | 5,026,367,224 bytes; SHA256 matches above |
| All 24 aggregator outputs, capture off/on | Maximum absolute difference **0** |
| Cached probabilities | Finite, nonnegative; shape `[1, 1, 2082, 2082]` |
| Full outgoing rows sum to 1 | Maximum error `2.384e-07` |
| Independent float64 full row reference | Maximum absolute error `2.024e-10` |
| Independent float64 full column reference | Maximum absolute error `2.137e-10` |
| Extracted spatial row / column | Both match the independent reference |
| Persistent CPU attention cache | 17,338,896 bytes |

The test used CPU float32 with four threads. It loaded all checkpoint parameters
and tested the complete aggregator; camera/depth/point/track prediction heads were
not executed. Timing fields in the JSON are a single local run, not a performance
benchmark. No CUDA result is implied.

[Complete validation JSON](assets/pretrained-validation.json) ·
[Real comparison PNG](assets/bathroom-attention.png) ·
[Raw maps, input tensors and float64 reference vectors](assets/bathroom-attention.npz)


To reproduce the pinned checkpoint locally:

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download("facebook/VGGT-1B", revision="860abec7937da0a4c03c41d3c269c366e82abdf9", allow_patterns=["config.json", "model.safetensors"], local_dir="VGGT-1B")'
python scripts/validate_pretrained.py --model ./VGGT-1B \
  --images temp_images/image_000.jpg temp_images/image_005.jpg \
  --output outputs/pretrained-validation --threads 4
```
