# Installation and troubleshooting

Use Python 3.10+ and a virtual environment. Install a matching PyTorch / torchvision
pair using the [official PyTorch instructions](https://pytorch.org/get-started/locally/).
Then run `python -m pip install -e '.[dev]'` from the repository.

## Entry points

- `vggt-attention --help`: headless export, no model download.
- `vggt-attention --demo`: offline synthetic tiny-model validation.
- `vggt-attention-gui --help`: desktop launch options, no window or model load.
- `python -m vggt.attention_gui`: equivalent source GUI entry.

CLI and GUI default to `global_block_11` and `spotlight`. Use `--style links`
for sparse outgoing targets or `--style raw` for shared-scale outgoing/incoming
maps. The GUI style selector redraws the selected query from its existing cache.
See the [real sample gallery](showcase.md) for reproducible commands and display
thresholds. The core cache API retains its block-23 compatibility default.

`pip install .` also installs the CLI, GUI and cache modules. Optional extras:
`[notebook]` for JupyterLab, `[demo]` for upstream reconstruction demos, `[research]`
for the archived position-analysis scripts. COLMAP and LightGlue requirements are
listed separately in [`archive/upstream/requirements.txt`](../archive/upstream/requirements.txt) and are not needed for attention. Historical reconstruction scripts and notebooks are collected in [`archive/`](../archive/README.md); the maintained attention entry points live in `vggt/`.

## Desktop dependencies

Tkinter comes from the Python installation rather than PyPI. Test it with
`python -m tkinter`. Linux distributions may provide a `python3-tk` package;
macOS users should use a Python build with working Tk support. A graphical
session is required. On remote/headless hosts use `vggt-attention` instead.

The viewer uses a main-thread event queue for Tk updates. Model inference and
plot construction run in a worker. Errors are shown in the UI and controls are
restored so another input can be selected.

## Weights and hardware

The default checkpoint is `facebook/VGGT-1B`. The first real inference downloads
about 5 GB. `--model /path/to/VGGT-1B` loads an existing directory containing
`config.json` and `model.safetensors`. Use `HF_HUB_OFFLINE=1` when all model files
are local. The maintained entries use strict checkpoint loading: missing or
unexpected parameters are errors, rather than silently retaining random weights.
No weights are needed for tests or `--demo`.

`--device auto` chooses CUDA if available, otherwise CPU; use `--device cpu` to
force CPU. MPS is not selected automatically. CUDA inference uses autocast where
supported, while cached probabilities are reconstructed in float32. The model
remains large even when only attention is requested.

Reduce view count or select a single layer if memory is constrained. Query
chunking reduces temporary allocation; the final CPU mean matrix remains quadratic
in token count. Choosing fewer heads changes the mean and temporary computation,
not the size of that final matrix.

## Development and package verification

```bash
python -m pytest -q
python -m build
python -m pip install --no-deps --force-reinstall dist/*.whl
vggt-attention --demo --output outputs/installed-demo.png
```

For a genuine installed-package smoke test, run the last command outside the
source checkout with the same virtual environment. See [validation](validation.md)
for the verification performed for this release preparation.
