# Historical development notes

This folder preserves the original requirements and implementation notes for reference:

- [request_attn.md](request_attn.md): the original Chinese requirements and progress log. Its local checkout path was replaced with `vggt/` for portability.
- [VGGT_ATTENTION_PROCESSING_REPORT.md](VGGT_ATTENTION_PROCESSING_REPORT.md): the early architecture and GUI processing report.
- [CLAUDE.md](CLAUDE.md): a historical technical summary, including fixed image sizes and old line references. It is documentation, not current project instructions.
- [dual_attention_demo.py](../legacy/dual_attention_demo.py) and [tkinter_initialization_demo.py](../legacy/tkinter_initialization_demo.py): the original print-based attention demonstration and a manual Tkinter initialization check. They were renamed from `test_dual_attention.py` and `test_tkinter_fix.py` to avoid automated test discovery. They are not correctness tests and may require a desktop or optional packages.

The published implementation has changed since these notes were written. Follow the current [README](../../README.md), [visualization guide](../../docs/usage.md), and assertion-based [tests](../../tests/) for supported behavior.

Historical filenames, commands, line numbers and implementation claims are retained inside the original notes. The current desktop module is `vggt/attention_gui.py`; current capture logic lives in `vggt/utils/attention_cache.py`. See the [archive index](../README.md) for moved scripts and notebooks.
