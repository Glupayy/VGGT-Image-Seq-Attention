# Asset provenance

- `hero.svg`: a code-drawn schematic of an attention matrix, not model data.
- `bathroom-attention.png`: rendered directly by `scripts/validate_pretrained.py`
  from the official VGGT-1B checkpoint on the two bundled views 000 and 005.
- `bathroom-attention.npz`: raw outgoing/incoming patch maps, preprocessed RGB
  inputs, and independently computed float64 full-row/full-column references.
- `integration-validation.json`: actual installed CLI and native desktop checks.
- `pretrained-validation.json`: measured numeric checks and environment. Local
  filesystem prefixes were replaced with repository-relative input names; numeric
  results are unchanged.

The checkpoint revision and full SHA256 are recorded in the validation JSON.
Weights are not included. Reproduce the figure using [the validation instructions](../validation.md).
