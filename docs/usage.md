# Attention visualization guide

The maintained paths are the Tkinter viewer (`vggt/attention_gui.py`) and the headless CLI (`vggt/attention_cli.py`). Start with
[README](../README.md) or [中文说明](../README.zh-CN.md).

## Workflow

1. Choose 1–6 views of the same scene.
2. Run `vggt-attention-gui`, select the files, and wait for inference.
3. Click a patch in the preprocessed view grid.
4. Choose `spotlight` for focus regions, `links` for sparse targets, or `raw` for
   outgoing and incoming maps on the same color scale. Style changes reuse capture.
5. Export the original-size PNG.

For numeric exports, use:

```bash
vggt-attention view_a.jpg view_b.jpg --source 0 --patch 13 17 \
  --layer 11 --style spotlight --output outputs/attention.png
```

`ROW COL` coordinates are zero-based in the model's preprocessed patch grid.
The standard preprocessing resizes/crops/pads inputs; original-image pixel
coordinates do not transfer directly. Special tokens are omitted from overlays.
The saved NPZ contains raw probabilities, without display normalization.
The CLI and GUI default to block 11 and average all heads. Spotlight applies a
within-view percentile cutoff; sparse links mark raw attention peaks. See the
[gallery](showcase.md) for real samples and the exact display transforms.

## Capture API

```python
import torch
from vggt.models.vggt_vis import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.visualization import render_attention_view

model = VGGT.from_pretrained("facebook/VGGT-1B", strict=True).eval()
images = load_and_preprocess_images(["view_a.jpg", "view_b.jpg"])
model.enable_attention_caching(["global_block_11"])
with torch.inference_mode():
    model.aggregator(images[None])
data = model.extract_patch_attention(0, 13, 17, "global_block_11", batch_idx=0)
render_attention_view(images, data, style="spotlight").save("attention.png")
model.disable_attention_caching()
```

Capture occurs inside attention after Q/K normalization and RoPE. There are no
forward hooks in the maintained viewer. Each model owns its cache; every forward
refreshes input geometry. Default capture averages all heads. Passing `max_heads=K`
selects the first K; returned metadata records the actual count.
Omitting an explicit layer list in the core API retains block 23 for compatibility.

## Historical notebook

[`archive/notebooks/show_attention.ipynb`](../archive/notebooks/show_attention.ipynb) records the original hook-based exploration. Its outputs
and machine paths were cleared for publication. It is not part of the maintained
validation path; APIs and assumptions in old cells may differ from current ones.
The [old processing report](../archive/notes/VGGT_ATTENTION_PROCESSING_REPORT.md) is retained as a dated implementation record. See the [archive index](../archive/README.md) for historical materials and the current API above for the maintained cache.

Layer-wise entropy conclusions in earlier notes came from individual examples.
They are not general model-level findings established by this repository's tests.
