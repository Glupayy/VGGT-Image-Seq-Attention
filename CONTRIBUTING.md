# Contributing

This is an independent VGGT attention visualization extension. Open an issue with
your Python/PyTorch versions, device, image count and preprocessed shape, selected
layer, and a minimal reproduction. Do not attach model weights or private images.

For changes, branch from the current default branch, keep upstream notices, and
update the relevant documentation. Install `.[dev]` with a matching torch/torchvision
pair, then run:

```bash
python -m pytest -q
python -m build
vggt-attention --demo --output outputs/demo.png
```

Tests use tiny random models and require no download. Add numerical references for
attention changes, especially head selection, special-token offsets and batch
indexing. Desktop changes should keep Tk access on the main thread and preserve
headless imports. Real pretrained checks belong in the validation record with
checkpoint revision and environment; synthetic results must be labelled.

Contributions retain the terms of [LICENSE.txt](LICENSE.txt).
