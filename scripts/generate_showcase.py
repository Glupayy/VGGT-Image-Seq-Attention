#!/usr/bin/env python3
"""Reproduce the curated real-weight gallery and verify its raw attention.

Run from a source checkout, for example::

    python scripts/generate_showcase.py --model ./VGGT-1B --output outputs/showcase

The fixed queries were selected by visual inspection, not a matching benchmark.
All 16 heads are averaged. Each scene runs with capture off and on, and every
published query is checked against an independent float64 row/column reference.
No prediction heads, training, or synthetic attention are used.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCENES = {
    "flower": {"files": ["000.png", "006.png", "012.png"],
               "queries": [("flower-center", (14, 25))]},
    "lego": {"files": ["00.png", "03.png", "06.png"],
             "queries": [("lego-cabin", (9, 17)), ("lego-bucket", (14, 29))]},
    "fern": {"files": ["000.png", "005.png", "010.png"],
             "queries": [("fern-trunk", (19, 19))]},
}
CHECKPOINT_SHA256 = "f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e"
CHECKPOINT_REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "VGGT-1B")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/showcase")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    checkpoint = args.model / "model.safetensors"
    if not checkpoint.is_file():
        parser.error("--model requires a local directory with model.safetensors")

    import numpy as np
    import torch
    from PIL import Image
    from scripts.validate_pretrained import independent_patch_reference, tensor_comparison
    from vggt.models.vggt_vis import VGGT
    from vggt.showcase import select_attention_peaks, select_top_attention
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.visualization import render_attention_view

    torch.set_num_threads(args.threads)
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != CHECKPOINT_SHA256:
        parser.error("checkpoint SHA256 differs from the official pinned gallery weights")
    args.output.mkdir(parents=True, exist_ok=True)
    layer = "global_block_11"
    report = {
        "kind": "curated-pretrained-attention-showcase",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": False, "device": "cpu", "dtype": "float32",
        "torch_version": str(torch.__version__), "threads": args.threads,
        "scope": "full aggregator; geometric prediction heads not executed",
        "checkpoint": {"filename": checkpoint.name, "bytes": checkpoint.stat().st_size,
                       "sha256": checkpoint_sha256, "strict_loading": True,
                       "model_id": "facebook/VGGT-1B", "revision": CHECKPOINT_REVISION},
        "layer": layer, "selected_heads": "all 16 heads; arithmetic mean",
        "sample_sources": "examples/SOURCES.json",
        "curation": "Visually chosen interpretable queries; not a matching-accuracy benchmark. "
                    "Other queries can attend to boundaries or unrelated regions.",
        "display": {
            "spotlight": "Original RGB retained strictly above each target view's 90th percentile; "
                         "other pixels dimmed to 34%. No attention renormalization.",
            "links": "Rank 1 raw outgoing maximum in each target view; patch-center arrows. "
                     "Not a verified geometric match.",
            "animation": "Two real LEGO queries, same model forward/layer/heads; 1800 ms per query.",
        }, "scenes": [],
    }
    print("Loading strict official checkpoint once on CPU…", flush=True)
    model = VGGT.from_pretrained(str(args.model), strict=True).cpu().float().eval()
    attention = model.aggregator.global_blocks[11].attn
    frames = []
    generated_paths = []
    all_passed = True
    for scene, spec in SCENES.items():
        paths = [ROOT / "examples/images" / scene / name for name in spec["files"]]
        print(f"{scene}: real aggregator, capture off/on…", flush=True)
        started = perf_counter()
        images = load_and_preprocess_images([str(path) for path in paths]).cpu().float()
        model.disable_attention_caching()
        with torch.inference_mode():
            baseline, baseline_offset = model.aggregator(images[None])
        captured = {}

        def capture_input(module, positional, keywords):
            captured["tokens"] = positional[0].detach().clone()
            positions = keywords.get("pos", positional[1] if len(positional) > 1 else None)
            captured["positions"] = None if positions is None else positions.detach().clone()

        handle = attention.register_forward_pre_hook(capture_input, with_kwargs=True)
        model.enable_attention_caching([layer], max_heads=None)
        try:
            with torch.inference_mode():
                observed, observed_offset = model.aggregator(images[None])
        finally:
            handle.remove()
        comparisons = [tensor_comparison(actual, expected, rtol=1e-5, atol=1e-6)
                       for actual, expected in zip(observed, baseline)]
        parity = (len(comparisons) == len(baseline) == len(observed) == 24
                  and baseline_offset == observed_offset and all(item["allclose"] for item in comparisons))
        del baseline, observed
        entry = model.attention_cache.get_attention(layer)
        matrix = entry["attention"]
        finite = bool(torch.isfinite(matrix).all())
        nonnegative = bool((matrix >= 0).all())
        row_error = float((matrix.sum(-1) - 1).abs().max())
        probability_ok = finite and nonnegative and row_error < 2e-6
        info = model.get_attention_cache_info()
        geometry = info["model_info"]
        indices = torch.cat([
            torch.arange(view * geometry["tokens_per_frame"] + geometry["patch_start_idx"],
                         (view + 1) * geometry["tokens_per_frame"])
            for view in range(len(images))])
        scene_report = {
            "scene": scene, "inputs": [str(path.relative_to(ROOT)) for path in paths],
            "input_shape": list(images.shape), "cache_shape": list(matrix.shape),
            "cache_bytes": info["memory_bytes"],
            "capture_parity": {"passed": parity, "outputs": len(comparisons),
                               "max_abs_diff": max(item["max_abs_diff"] for item in comparisons)},
            "probabilities": {"finite": finite, "nonnegative": nonnegative,
                              "full_rows_sum_to_one": probability_ok, "max_row_sum_error": row_error},
            "queries": [],
        }
        all_passed &= parity and probability_ok and entry["selected_num_heads"] == 16
        for name, patch in spec["queries"]:
            print(f"  {name} {patch}: independent float64 reference and render…", flush=True)
            data = model.extract_patch_attention(0, *patch, layer)
            token = data["source_token_idx"]
            ref_row, ref_col = independent_patch_reference(
                attention, captured["tokens"], captured["positions"], token)
            outgoing = torch.stack(data["attention_maps_row"])
            incoming = torch.stack(data["attention_maps_col"])
            checks = {
                "full_row": tensor_comparison(matrix[:, 0, token].double(), ref_row, rtol=1e-4, atol=1e-7),
                "full_column": tensor_comparison(matrix[:, 0, :, token].double(), ref_col, rtol=1e-4, atol=1e-7),
                "spatial_outgoing": tensor_comparison(outgoing.flatten().double(), ref_row[0, indices], rtol=1e-4, atol=1e-7),
                "spatial_incoming": tensor_comparison(incoming.flatten().double(), ref_col[0, indices], rtol=1e-4, atol=1e-7),
            }
            all_passed &= all(item["allclose"] for item in checks.values())
            arrays_path = args.output / f"{name}.npz"
            np.savez_compressed(arrays_path, images=images.numpy(),
                                outgoing=outgoing.numpy(), incoming=incoming.numpy(),
                                reference_full_row=ref_row.numpy(), reference_full_column=ref_col.numpy())
            generated_paths.append(arrays_path)
            targets = []
            for target in range(1, len(images)):
                raw = outgoing[target].numpy()
                mask = select_top_attention(raw)
                targets.append({"view": target, "raw_visible_mass": float(raw.sum()),
                                "spotlight_selected_patches": int(mask.sum()),
                                "spotlight_selected_mass": float(raw[mask].sum()),
                                "top3_separated_peaks": select_attention_peaks(raw)})
            for style in ("links", "spotlight"):
                figure = render_attention_view(images, data, style=style, top_k=1)
                image_path = args.output / f"{name}-{style}.png"
                figure.save(image_path)
                generated_paths.append(image_path)
                if scene == "lego" and style == "spotlight":
                    frames.append(figure.convert("RGB"))
            scene_report["queries"].append({"name": name, "source_view": 0,
                                            "patch_row_col": list(patch), "source_token_idx": token,
                                            "independent_float64_reference": checks, "targets": targets})
        scene_report["elapsed_seconds"] = perf_counter() - started
        report["scenes"].append(scene_report)
        del captured, matrix, entry
    if frames:
        # A shared palette keeps colors stable as the selected real query changes.
        palette = frames[0].quantize(colors=256)
        animation = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
        animation_path = args.output / "lego-query-change.gif"
        animation[0].save(animation_path, save_all=True,
                          append_images=animation[1:], duration=1800, loop=0, disposal=2)
        generated_paths.append(animation_path)
    report["passed"] = bool(all_passed)
    report["artifacts"] = {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                           for path in sorted(generated_paths)}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{'PASS' if all_passed else 'FAIL'}: {args.output / 'validation.json'}", flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
