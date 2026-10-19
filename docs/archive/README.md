# Historical development notes

This folder preserves the original requirements and implementation notes for reference:

- [request_attn.md](request_attn.md): the original Chinese requirements and progress log. Its local checkout path was replaced with `vggt/` for portability.
- [CLAUDE.md](CLAUDE.md): a historical technical summary, including fixed image sizes and old line references. It is documentation, not current project instructions.
- [dual_attention_demo.py](legacy_demos/dual_attention_demo.py) and [tkinter_initialization_demo.py](legacy_demos/tkinter_initialization_demo.py): the original print-based attention demonstration and a manual Tkinter initialization check. They were renamed from `test_dual_attention.py` and `test_tkinter_fix.py` to avoid automated test discovery. They are not correctness tests and may require a desktop or optional packages.

The published implementation has changed since these notes were written. Follow the current [README](../../README.md), [visualization guide](../../ATTENTION_VISUALIZATION_GUIDE.md), and assertion-based tests under `tests/` for supported behavior.
