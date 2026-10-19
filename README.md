<div align="center">

<img src="docs/assets/hero.svg" alt="VGGT Attention Explorer — follow a patch across views" width="100%">

**Click a patch. Inspect outgoing and incoming attention across views.**

[中文说明](README.zh-CN.md) · [Quick start](#quick-start) · [How it works](#reading-the-maps) · [Validation](docs/validation.md)

[![Tests and package](https://github.com/Glupayy/vggt_attn/actions/workflows/ci.yml/badge.svg)](https://github.com/Glupayy/vggt_attn/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB)
[![VGGT](https://img.shields.io/badge/built_on-VGGT-157f79)](https://github.com/facebookresearch/vggt)
[![License](https://img.shields.io/badge/license-CC_BY--NC_4.0-lightgrey)](LICENSE.txt)

</div>

A research visualization tool built on [VGGT](https://github.com/facebookresearch/vggt) (CVPR 2025). Explore a selected patch's global attention in both directions, with an interactive desktop viewer or a headless export command. This repository is an independent extension of VGGT.

- **Two directions, one selection.** Compare what a patch attends to with which patches attend to it.
- **Comparable maps.** All overlays use the same linear scale and preserve raw probabilities.
- **Capture once, explore repeatedly.** Cache a selected layer's head-mean matrix on CPU; clicks reuse it.
- **Inspect the numbers.** CLI exports a PNG, compressed numeric arrays, and JSON provenance together.
- **Reproducible checks.** Numeric references, model-parity checks, coordinate tests, and offline exports run without pretrained weights.

## Real pretrained example

![Outgoing and incoming VGGT attention across two bathroom views](docs/assets/bathroom-attention.png)

Official VGGT-1B, `global_block_23`, all 16 heads, shared linear scale. The selected patch is outlined in mint. This is an actual CPU inference result, not the synthetic fixture.

[Validation JSON](docs/assets/pretrained-validation.json) · [Raw maps and inputs](docs/assets/bathroom-attention.npz) · [Reproduce the run](docs/validation.md#pretrained-model-check)

## Quick start

Use Python 3.10 or newer. Install a matching PyTorch / torchvision pair for your hardware using the [official selector](https://pytorch.org/get-started/locally/), then:

```bash
git clone https://github.com/Glupayy/vggt_attn.git
cd vggt_attn
python -m venv .venv
source .venv/bin/activate
# Install torch and torchvision in this environment first.
python -m pip install -e '.[dev]'
```

### Try an offline fixture

```bash
vggt-attention --demo --output outputs/demo.png
```

This uses a deterministic, randomly initialized tiny VGGT aggregator and synthetic images. It checks the entire capture → extraction → plot → export path; it is **not** a pretrained VGGT-1B result.

### Explore your images

```bash
vggt-attention-gui
# Equivalent source entry:
python interactive_attention_tkinter.py
```

Select 1–6 related views, run attention capture, then click a patch. The viewer loads the model lazily and keeps the interface responsive while inference runs. Use the export button to save the comparison.

The default model is [`facebook/VGGT-1B`](https://huggingface.co/facebook/VGGT-1B), downloaded on first real inference. CUDA is used when available, otherwise CPU. CPU inference works but the full model is substantial. The GUI needs Tkinter and a graphical session; use the CLI on a headless machine. See [installation and troubleshooting](docs/package.md).

### Export a specific patch

```bash
vggt-attention temp_images/image_000.jpg temp_images/image_005.jpg \
  --source 0 --patch 14 18 --layer 23 --output outputs/bathroom.png
```

All indices are zero-based. `--patch ROW COL` refers to the **preprocessed** image grid, not original image pixels. Omit `--patch` to select its center. To use a local checkpoint:

```bash
vggt-attention temp_images/image_000.jpg --model ./VGGT-1B \
  --device cpu --output outputs/local.png
```

Each export contains:

| File | Contents |
| --- | --- |
| `attention.png` | Original views and paired outgoing / incoming overlays |
| `attention.npz` | `outgoing`, `incoming` arrays `[views, patch_rows, patch_cols]`; preprocessed RGB `images` |
| `attention.json` | Model ID or local path, input paths, patch, layer, head selection, device, and cache geometry |

## Reading the maps

For each head, the tool captures `A = softmax(Q Kᵀ / √d)` **after Q/K normalization and 2D RoPE**, before dropout. It averages the selected heads. The default includes all heads; `--max-heads K` deliberately uses only the first K and records that choice.

For selected token `i`:

| Direction | Slice | Question |
| --- | --- | --- |
| Outgoing | `A[i, :]` | What does this patch attend to? |
| Incoming | `A[:, i]` | Which tokens attend to this patch? |

The full outgoing row sums to one. A column generally does not. Camera/register tokens participate in the softmax, but overlays show only image patches, so visible outgoing mass may sum to less than one. Incoming maps are **not** normalized again. Attention weights describe this layer's routing; they do not establish a causal explanation or a geometric correspondence by themselves.

## Implementation

Unobserved layers use PyTorch scaled dot-product attention. For a selected layer, query chunks reconstruct probabilities in float32, average heads, and transfer one `[B, 1, N, N]` matrix to CPU. This bounds temporary attention allocations while retaining fast repeated patch selection. The persistent CPU cache still costs `4 × B × N²` bytes per layer.

Geometry comes from each actual forward pass: batch, view count, patch dimensions, and special-token offset. Caches belong to individual models and are refreshed on every forward. Default capture is only `global_block_23`; the visualization entries run the aggregator without reconstruction heads.

```python
import torch
from vggt.models.vggt_vis import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

model = VGGT.from_pretrained("facebook/VGGT-1B", strict=True).eval()
images = load_and_preprocess_images(["view_a.jpg", "view_b.jpg"])
model.enable_attention_caching(["global_block_23"])
with torch.inference_mode():
    model.aggregator(images[None])
data = model.extract_patch_attention(0, 13, 17, "global_block_23")
```

## Validation and project map

```bash
python -m pytest -q
python -m build
```

[Validation notes](docs/validation.md) describe the tested environment, numerical checks, and real-model verification. GitHub Actions runs CPU tests and builds on Python 3.10 and 3.12 after publication; the badge reflects the remote workflow state.

| Path | Purpose |
| --- | --- |
| `interactive_attention_tkinter.py` | Maintained desktop viewer |
| `vggt/attention_cli.py` | Reproducible headless exports |
| `vggt/visualization.py` | Shared plotting code |
| `vggt/utils/attention_cache.py` | Model-local, chunked attention capture |
| `show_attention.ipynb` | Historical exploratory notebook; see the guide before use |
| `temp_images/` | Six existing scene views used for examples |
| [`research/`](research/README.md) | Archived RoPE / position experiments, including schematic simulations |
| [`docs/archive/`](docs/archive/README.md) | Original development notes and old demonstrations |

The OpenCV and old-hook viewers remain historical alternatives; current fixes and tests target the Tkinter and CLI entries. The original upstream introduction is preserved in [the upstream notes](docs/upstream-vggt.md).

## Attribution and license

Built on the work of Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. Preserve the upstream notices and [repository license](LICENSE.txt), CC BY-NC 4.0. Model weights retain their [upstream terms](https://huggingface.co/facebook/VGGT-1B). No model weights are committed here.

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
