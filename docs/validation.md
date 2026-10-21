# Validation record

## Directory organization — 2026-10-21

**153 tests passed** in both the source checkout and the extracted source archive.
The root now has 17 entries, down from 44. Maintained code is in `vggt/`, inputs
in `examples/`, tools in `scripts/`, and earlier work in `archive/`.

All 152 previously tracked files were accounted for at their new paths. Sixty
files intended to retain their contents were checked byte for byte, including
all 15 input images, historical figures and three notebooks. The GUI implementation
is identical apart from its module location and launch-command docstring.

The rebuilt wheel was installed and checked outside the checkout. The GUI console
entry and all three offline export styles work. Preprocessing the relocated flower
images reproduces the published input tensors exactly; rendering their saved raw
attention reproduces the published PNG pixel for pixel. This checks the new paths
and package layout without presenting cached rendering as a new inference run.
Both distributions exclude model weights.

[Directory and package check record](assets/layout-validation.json)

## Paths after repository reorganization

The numeric JSON records retain the paths and timestamps of their recorded runs.
The same input bytes now live in these locations; moving files does not represent
an additional model-validation run:

| Recorded path | Current repository path |
| --- | --- |
| `sample_images/<scene>/<file>` | `examples/images/<scene>/<file>` |
| `sample_images/SOURCES.json` | `examples/SOURCES.json` |
| `temp_images/<file>` | `examples/images/bathroom/<file>` |
| `interactive_attention_tkinter.py` | `vggt/attention_gui.py` |

The commands below use the current layout. Historical input-path fields in JSON
remain unchanged so the original runs can be audited.

## Gallery revision — 2026-10-21 (Asia/Shanghai)

**152 tests passed.** The suite now also checks percentile selection, spatially
separated raw maxima, exact query/target arrow endpoints, uniform maps, 1/2/6-view
layout, long score captions, per-file sample hashes, display dispatch, and GUI
style switching. PNGs differ across styles while the exported raw arrays remain
identical. Tests require neither pretrained weights nor network access.

The gallery was regenerated from the same verified official checkpoint below,
using `strict=True`, CPU float32, four threads, global block 11 and all 16 heads.
Each scene uses three source images, with four curated queries in total.

| Scene | All 24 outputs, capture off/on | Maximum full-row sum error |
| --- | --- | --- |
| Flower | Exact equality; maximum difference 0 | `4.768e-07` |
| LEGO | Exact equality; maximum difference 0 | `5.960e-07` |
| Fern | Exact equality; maximum difference 0 | `3.576e-07` |

All four queries' full outgoing rows, full incoming columns and extracted
spatial maps pass an independent float64 attention reference (`rtol=1e-4`,
`atol=1e-7`); the largest absolute difference is `2.241e-07`. QKV, Q/K
normalization and RoPE are reapplied in model float32 before float64 QK products,
softmax and full-row denominators. No per-view attention renormalization is used.

A native macOS Tk check ran one real flower inference, clicked patch `(14,25)`,
and switched through Spotlight, Links and Raw. All three result windows and
full-resolution exports passed; raw outgoing/incoming arrays match the gallery
exactly. A forward hook counted **one** aggregator call across all three styles.
Spotlight and Links exported at 2580 × 1480; Raw exported at 2160 × 2150.

The source archive and wheel both build. The extracted source archive passes
the same 152 tests and contains all nine sample PNGs and the generation script.
The installed wheel was exercised from `/private/tmp`, outside the checkout:
all three offline styles pass, and a real flower export has exactly the same
raw arrays and PNG pixels as the published gallery. Neither distribution
contains model weights. CUDA and remote GitHub Actions were not tested here.

[Gallery, selection limits and reproduction](showcase.md) ·
[Complete gallery numeric record](assets/showcase/validation.json) ·
[Desktop and package record](assets/showcase/integration.json)

## Original release preparation — 2026-10-19

The following records preserve the earlier bathroom / block-23 validation.

### Local result

**89 tests passed.** The native desktop check, offline export, wheel build, source
archive build, and installed-wheel export from outside the checkout passed.
The extracted source archive was also tested independently; its tests, scripts,
and six example images are included. Neither distribution contains model weights.

Test environment: macOS arm64, Python 3.11.4, torch 2.11.0, torchvision 0.26.0,
NumPy 1.24.3, Pillow 12.0.0, matplotlib 3.10.8, huggingface_hub 1.2.4,
safetensors 0.5.3, einops 0.8.2. CUDA execution was not tested on this machine.

### Automated checks

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

### Desktop check

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

### Pretrained model check

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
  --images examples/images/bathroom/image_000.jpg examples/images/bathroom/image_005.jpg \
  --output outputs/pretrained-validation --threads 4
```
