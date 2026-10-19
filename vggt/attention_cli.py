"""Export patch attention as a PNG, numeric NPZ, and provenance JSON."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Optional, Sequence


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="*", type=Path, help="One to six image files, in view order")
    parser.add_argument("--demo", action="store_true", help="Offline synthetic tiny-model fixture; no pretrained weights")
    parser.add_argument("--model", default="facebook/VGGT-1B", help="Hugging Face model ID or local checkpoint directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--layer", type=int, default=None, help="Global block index (default: 23; demo: 1)")
    parser.add_argument("--max-heads", type=int, default=None, help="Average the first K heads (default: all)")
    parser.add_argument("--source", type=int, default=0, help="Zero-based source view index")
    parser.add_argument("--patch", type=int, nargs=2, metavar=("ROW", "COL"), default=None,
                        help="Zero-based coordinates in the preprocessed patch grid")
    parser.add_argument("--output", type=Path, default=Path("outputs/attention.png"))
    return parser


def create_demo(source=0, patch=(2, 3), layer=1, max_heads=None):
    """Run the real alternating-attention implementation with random tiny weights."""
    import torch
    from vggt.models.aggregator_vis import Aggregator
    from vggt.utils.attention_cache import AttentionCacheManager

    y, x = torch.meshgrid(torch.linspace(0, 1, 84), torch.linspace(0, 1, 112), indexing="ij")
    images = []
    for shift in (0.0, 0.16, -0.12):
        circle = ((x - 0.48 - shift) ** 2 + (y - 0.52) ** 2 < 0.055).float()
        images.append(torch.stack((0.15 + 0.7 * circle, 0.18 + 0.65 * x, 0.22 + 0.65 * y)))
    images = torch.stack(images)
    cache = AttentionCacheManager()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        model = Aggregator(img_size=112, embed_dim=32, depth=2, num_heads=4,
                           patch_embed="conv", mlp_ratio=2).eval()
    cache.enable_caching([f"global_block_{layer}"], max_heads=max_heads)
    model.set_attention_cache_manager(cache)
    with torch.inference_mode():
        model(images[None])
    data = cache.extract_patch_attention(source, *patch, f"global_block_{layer}")
    data["fixture"] = "SYNTHETIC FIXTURE / random tiny weights / not pretrained VGGT-1B"
    return images, data, cache.get_cache_info()


def main(argv: Optional[Sequence[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_heads is not None and args.max_heads < 1:
        parser.error("--max-heads must be a positive integer")
    layer = args.layer if args.layer is not None else (1 if args.demo else 23)
    if not 0 <= layer < (2 if args.demo else 24):
        parser.error("--layer is outside the model's global block range")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must end in .png")
    if args.demo and args.images:
        parser.error("--demo uses synthetic inputs; omit image paths")
    if args.demo and args.device == "cuda":
        parser.error("--demo is a fixed CPU fixture; use --device auto or cpu")
    if not args.demo and not 1 <= len(args.images) <= 6:
        parser.error("Provide one to six image files, or use --demo")
    count = 3 if args.demo else len(args.images)
    if not 0 <= args.source < count:
        parser.error("--source is outside the input view range")
    if args.patch is not None and min(args.patch) < 0:
        parser.error("--patch coordinates must be nonnegative")
    if args.demo and args.patch is not None and not (args.patch[0] < 6 and args.patch[1] < 8):
        parser.error("--demo patch must lie inside its 6 x 8 grid")
    for path in args.images:
        if not path.is_file():
            parser.error(f"Image does not exist: {path}")

    import numpy as np
    import torch
    from vggt.visualization import render_attention_comparison

    try:
        if args.demo:
            images, data, info = create_demo(args.source, tuple(args.patch or (2, 3)), layer, args.max_heads)
            provenance = {"kind": "synthetic-tiny-model", "seed": 7,
                          "model": "random 2-block VGGT aggregator; not VGGT-1B",
                          "device": "cpu", "model_dtype": "float32"}
        else:
            from vggt.models.vggt_vis import VGGT
            from vggt.utils.load_fn import load_and_preprocess_images
            device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
            if device == "auto":
                device = "cpu"
            if device == "cuda" and not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable")
            images = load_and_preprocess_images([str(p) for p in args.images])
            patch = tuple(args.patch or (images.shape[-2] // 28, images.shape[-1] // 28))
            grid = (images.shape[-2] // 14, images.shape[-1] // 14)
            if not (0 <= patch[0] < grid[0] and 0 <= patch[1] < grid[1]):
                raise ValueError(f"--patch must lie inside the {grid[0]} x {grid[1]} preprocessed grid")
            model = VGGT.from_pretrained(args.model, strict=True).to(device).eval()
            model.enable_attention_caching([f"global_block_{layer}"], max_heads=args.max_heads)
            dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
            context = torch.autocast("cuda", dtype=dtype) if device == "cuda" else nullcontext()
            with torch.inference_mode(), context:
                model.aggregator(images[None].to(device))
            data = model.extract_patch_attention(args.source, *patch, f"global_block_{layer}")
            info = model.get_attention_cache_info()
            provenance = {"kind": "pretrained-vggt", "model": args.model, "device": device,
                          "model_dtype": str(next(model.parameters()).dtype),
                          "autocast_dtype": str(dtype) if device == "cuda" else None,
                          "images": [str(p) for p in args.images]}
        if data is None:
            raise ValueError("No attention was captured for the requested layer")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        image = render_attention_comparison(images, data)
        image.save(args.output)
        np.savez_compressed(args.output.with_suffix(".npz"),
                            outgoing=np.stack([m.float().numpy() for m in data["attention_maps_row"]]),
                            incoming=np.stack([m.float().numpy() for m in data["attention_maps_col"]]),
                            images=images.cpu().float().numpy())
        metadata = {**provenance, "layer": data["layer_name"], "source_view": args.source,
                    "source_patch_row_col": list(data["source_patch"]),
                    "input_shape": list(images.shape), "torch_version": torch.__version__,
                    "selected_num_heads": data.get("selected_num_heads"),
                    "original_num_heads": data.get("original_num_heads"),
                    "head_reduction": "mean",
                    "cache": info, "normalization": "raw softmax(QK^T / sqrt(d)), head mean; no column normalization"}
        args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {args.output} (+ .npz raw maps and .json provenance)")
        if args.demo:
            print("Synthetic validation fixture with random tiny weights; not pretrained VGGT-1B results.")
    except (OSError, RuntimeError, ValueError, IndexError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
