# Historical material

The maintained desktop and CLI entries live in [`vggt/`](../vggt/). Start with the [project README](../README.md) or [usage guide](../docs/usage.md) for current commands.

This archive preserves earlier experiments and upstream demonstrations:

| Directory | Contents |
| --- | --- |
| [`legacy/`](legacy/) | Old OpenCV and hook-based attention viewers, helper modules, and manual demonstrations |
| [`upstream/`](upstream/) | Upstream reconstruction demos, `visual_util.py`, and their separate `requirements.txt` |
| [`notebooks/`](notebooks/) | Original demo, video-inference, and attention notebooks, with execution outputs cleared |
| [`results/`](results/) | Original attention result PNGs from the earlier development layout |
| [`research/`](research/README.md) | Position encoding / RoPE manuscripts, figure scripts, and explanatory figures |
| [`notes/`](notes/README.md) | Original requirements, progress logs, and early implementation reports |

Historical commands and line references inside these files describe their original layout. These materials may use earlier APIs, machine-specific assumptions, or optional dependencies. The current tests target the maintained code; archiving a script does not establish that it runs with the current package.

The research figure scripts use simulated attention for explanation. The earlier result images and notes are historical records; current pretrained gallery evidence is documented in [`docs/showcase.md`](../docs/showcase.md) and the [validation record](../docs/validation.md).

Input images remain outside this archive in [`examples/`](../examples/README.md). Numerical validation JSON files retain their recorded input paths; the [validation notes](../docs/validation.md#paths-after-repository-reorganization) map those paths to the current layout.
