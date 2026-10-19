#!/usr/bin/env python3
"""Validate a real VGGT checkpoint on CPU and export attention evidence.

Example (from the repository root)::

    python scripts/validate_pretrained.py --model /path/to/VGGT-1B \
        --images temp_images/image_000.jpg temp_images/image_001.jpg \
        --output outputs/pretrained_validation --threads 4

This runs the complete aggregator twice, with and without capture. It skips
geometric prediction heads. A separate float64 QK reference checks a complete
outgoing row and incoming column, including every softmax denominator and all
special tokens, before checking the extracted spatial maps.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/VGGT-1B",
                        help="Hugging Face model ID or local pretrained checkpoint directory")
    parser.add_argument("--images", nargs="+", type=Path,
                        default=[REPO_ROOT / "temp_images/image_000.jpg",
                                 REPO_ROOT / "temp_images/image_001.jpg"],
                        help="One to six images; default: the first two bundled temp_images views")
    parser.add_argument("--output", type=Path, default=Path("outputs/pretrained_validation"),
                        help="Directory for validation.json, attention.png and attention.npz")
    parser.add_argument("--threads", type=int, default=4, help="CPU PyTorch threads (default: 4)")
    parser.add_argument("--reference-chunk-size", type=int, default=64,
                        help="Query block size for the independent float64 reference")
    return parser


def tensor_comparison(actual, expected, *, rtol, atol):
    """Return JSON-safe comparison metrics, including nonfinite failures."""
    import torch

    same_shape = actual.shape == expected.shape
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    difference = float((actual - expected).abs().max()) if same_shape and finite else None
    return {
        "allclose": bool(same_shape and finite and torch.allclose(actual, expected, rtol=rtol, atol=atol)),
        "max_abs_diff": difference, "finite": finite,
        "actual_shape": list(actual.shape), "expected_shape": list(expected.shape),
        "rtol": rtol, "atol": atol,
    }


def finite_number(value):
    """Keep failure reports valid JSON even when a tensor contains NaNs."""
    value = float(value)
    return value if math.isfinite(value) else None


def independent_patch_reference(attention, tokens, positions, source_token, query_chunk_size=64):
    """Recompute one full row/column without using any cache implementation.

    The hook input is already transformed by Block.norm1. Reapply the actual
    learned QKV linear map, Q/K normalization and RoPE in the model's float32
    dtype. QK products, softmax and column denominators then use float64 einsum
    and logsumexp, independently of the cache's float32 matmul/softmax path.
    """
    import torch
    import torch.nn.functional as F

    if query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    batch, token_count, channels = tokens.shape
    if not 0 <= source_token < token_count:
        raise ValueError("source_token is outside the attention sequence")
    heads = attention.num_heads
    head_dim = channels // heads
    with torch.inference_mode():
        qkv = F.linear(tokens, attention.qkv.weight, attention.qkv.bias)
        qkv = qkv.reshape(batch, token_count, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
        query, key, _ = qkv.unbind(0)
        query, key = attention.q_norm(query), attention.k_norm(key)
        if attention.rope is not None:
            query, key = attention.rope(query, positions), attention.rope(key, positions)
        query, key = query.double(), key.double()
        scale = head_dim ** -0.5
        row_logits = torch.einsum("bhd,bhkd->bhk", query[:, :, source_token], key) * scale
        outgoing = row_logits.softmax(-1).mean(dim=1)
        incoming = torch.empty(batch, token_count, dtype=torch.float64)
        for start in range(0, token_count, query_chunk_size):
            end = min(start + query_chunk_size, token_count)
            logits = torch.einsum("bhqd,bhkd->bhqk", query[:, :, start:end], key) * scale
            # Every key (including camera/register tokens) contributes to each
            # row's denominator. Softmax over the column would be incorrect.
            log_denominator = torch.logsumexp(logits, dim=-1)
            probability = torch.exp(logits[..., source_token] - log_denominator)
            incoming[:, start:end] = probability.mean(dim=1)
    return outgoing, incoming


def run_validation(args):
    import numpy as np
    import torch

    from vggt.models.vggt_vis import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.visualization import render_attention_comparison

    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "validation.json"
    report = {
        "kind": "pretrained-vggt-validation",
        "model": args.model, "device": "cpu", "dtype": "float32",
        "scope": "full pretrained aggregator, geometric prediction heads skipped",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "images": [str(path.resolve()) for path in args.images],
        "torch_version": str(torch.__version__), "threads": torch.get_num_threads(),
        "passed": False,
    }
    total_started = perf_counter()

    def save_report():
        report["elapsed_seconds"] = perf_counter() - total_started
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    try:
        local_checkpoint = Path(args.model) / "model.safetensors"
        if local_checkpoint.is_file():
            digest = hashlib.sha256()
            with local_checkpoint.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            report["checkpoint"] = {"filename": "model.safetensors",
                                    "size_bytes": local_checkpoint.stat().st_size,
                                    "sha256": digest.hexdigest(), "strict_loading": True}
        print("Preprocessing input views…", flush=True)
        images = load_and_preprocess_images([str(path) for path in args.images]).cpu().float()
        report["preprocessing"] = "VGGT load_and_preprocess_images(mode='crop'); width 518, patch-aligned height"
        report["input_shape"] = list(images.shape)
        print(f"Loading {args.model} on CPU…", flush=True)
        started = perf_counter()
        model = VGGT.from_pretrained(args.model, strict=True).cpu().float().eval()
        report["model_load_seconds"] = perf_counter() - started
        if len(model.aggregator.global_blocks) <= 23:
            raise ValueError("this validation requires a VGGT checkpoint with global_block_23")
        layer = "global_block_23"
        attention = model.aggregator.global_blocks[23].attn
        report["layer"] = layer
        report["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
        inputs = images.unsqueeze(0)

        print("Forward 1/2: attention capture disabled…", flush=True)
        model.disable_attention_caching()
        started = perf_counter()
        with torch.inference_mode():
            baseline, baseline_offset = model.aggregator(inputs)
        report["uncached_forward_seconds"] = perf_counter() - started

        captured = {"calls": 0}

        def capture_input(module, positional, keywords):
            captured["calls"] += 1
            captured["tokens"] = positional[0].detach().cpu().clone()
            positions = keywords.get("pos", positional[1] if len(positional) > 1 else None)
            captured["positions"] = None if positions is None else positions.detach().cpu().clone()

        handle = attention.register_forward_pre_hook(capture_input, with_kwargs=True)
        print("Forward 2/2: capture global_block_23, averaging all heads…", flush=True)
        model.enable_attention_caching([layer], max_heads=None)
        started = perf_counter()
        try:
            with torch.inference_mode():
                observed, observed_offset = model.aggregator(inputs)
        finally:
            handle.remove()
        report["cached_forward_seconds"] = perf_counter() - started
        outputs = [tensor_comparison(actual, expected, rtol=1e-5, atol=1e-6)
                   for actual, expected in zip(observed, baseline)]
        for index, item in enumerate(outputs):
            item["aggregator_output_index"] = index
        output_match = (len(observed) == len(baseline) and observed_offset == baseline_offset
                        and bool(outputs) and all(item["allclose"] for item in outputs))
        report["aggregator_outputs"] = {
            "allclose": output_match, "baseline_count": len(baseline), "cached_count": len(observed),
            "patch_start_idx_unchanged": observed_offset == baseline_offset,
            "max_abs_diff": (max(item["max_abs_diff"] for item in outputs)
                             if outputs and all(item["finite"] for item in outputs) else None),
            "per_output": outputs,
        }
        del baseline, observed

        cache = model.attention_cache.get_attention(layer)
        if cache is None:
            raise RuntimeError("global_block_23 did not produce cached attention")
        probabilities = cache["attention"]
        info = model.get_attention_cache_info()
        finite = bool(torch.isfinite(probabilities).all())
        nonnegative = bool((probabilities >= 0).all())
        row_error = float((probabilities.sum(-1) - 1).abs().max()) if finite else None
        row_normalized = bool(finite and row_error <= 2e-6)
        report["cache"] = {
            **info, "attention_shape": list(probabilities.shape), "dtype": str(probabilities.dtype),
            "selected_num_heads": cache["selected_num_heads"],
            "original_num_heads": cache["original_num_heads"],
            "cached_num_heads": cache["num_heads"], "head_reduction": cache["head_reduction"],
            "finite": finite, "nonnegative": nonnegative,
            "full_rows_sum_to_one": row_normalized, "max_row_sum_error": row_error,
            "row_sum_atol": 2e-6,
        }
        geometry = info["model_info"]
        patch = (geometry["patch_h"] // 2, geometry["patch_w"] // 2)
        data = model.extract_patch_attention(0, *patch, layer)
        if data is None:
            raise RuntimeError("no spatial maps were extracted for the selected patch")
        token = data["source_token_idx"]
        report["source"] = {"view": 0, "patch_row_col": list(patch), "global_token_idx": token}

        print("Independent reference: recomputing QKV/QK norm/RoPE and float64 row/column…", flush=True)
        if captured["calls"] != 1:
            raise RuntimeError(f"expected one last-layer attention call, observed {captured['calls']}")
        started = perf_counter()
        reference_row, reference_column = independent_patch_reference(
            attention, captured["tokens"], captured["positions"], token, args.reference_chunk_size)
        report["reference_seconds"] = perf_counter() - started
        del captured
        comparison_options = {"rtol": 1e-4, "atol": 1e-7}
        row_check = tensor_comparison(probabilities[:, 0, token].double(), reference_row,
                                      **comparison_options)
        column_check = tensor_comparison(probabilities[:, 0, :, token].double(), reference_column,
                                         **comparison_options)
        spatial_indices = torch.cat([
            torch.arange(frame * geometry["tokens_per_frame"] + geometry["patch_start_idx"],
                         (frame + 1) * geometry["tokens_per_frame"])
            for frame in range(geometry["num_frames"])
        ])
        extracted_row = torch.cat([item.flatten() for item in data["attention_maps_row"]]).double()
        extracted_column = torch.cat([item.flatten() for item in data["attention_maps_col"]]).double()
        spatial_row_check = tensor_comparison(extracted_row, reference_row[0, spatial_indices],
                                              **comparison_options)
        spatial_column_check = tensor_comparison(extracted_column, reference_column[0, spatial_indices],
                                                 **comparison_options)
        reference_ok = all(item["allclose"] for item in
                           (row_check, column_check, spatial_row_check, spatial_column_check))
        report["independent_reference"] = {
            "allclose": reference_ok,
            "input": "actual global_block_23.attn pre-hook input, after Block.norm1",
            "calculation": "F.linear + learned Q/K norm + RoPE; float64 einsum, softmax and logsumexp",
            "query_chunk_size": args.reference_chunk_size,
            "all_heads": attention.num_heads, "full_row": row_check, "full_column": column_check,
            "extracted_spatial_row": spatial_row_check, "extracted_spatial_column": spatial_column_check,
            "full_row_sum": finite_number(reference_row.sum()),
            "full_column_sum": finite_number(reference_column.sum()),
            "column_normalization": "none; a column is not a probability distribution over queries",
        }
        report["normalization"] = (
            "softmax over all keys after Q/K normalization and RoPE, then all-head mean; "
            "special tokens excluded from exported spatial maps; incoming columns are not renormalized")
        report["passed"] = bool(output_match and finite and nonnegative and row_normalized and reference_ok)
        save_report()

        if finite and nonnegative:
            print("Exporting PNG and raw probability maps…", flush=True)
            render_attention_comparison(images, data, patch_size=geometry["patch_size"]).save(
                args.output / "attention.png")
            np.savez_compressed(
                args.output / "attention.npz",
                outgoing=np.stack([item.numpy() for item in data["attention_maps_row"]]),
                incoming=np.stack([item.numpy() for item in data["attention_maps_col"]]),
                images=images.numpy(), full_row_reference=reference_row.numpy(),
                full_column_reference=reference_column.numpy(),
            )
            report["artifacts"] = {"png": "attention.png", "raw_maps": "attention.npz",
                                   "provenance_and_validation": "validation.json"}
        save_report()
        print(f"{'PASS' if report['passed'] else 'FAIL'} · {report_path}", flush=True)
        return 0 if report["passed"] else 1
    except Exception as exc:
        report["passed"] = False
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save_report()
        print(f"FAIL · {type(exc).__name__}: {exc}\nDiagnostics: {report_path}", file=sys.stderr, flush=True)
        return 1


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 1 <= len(args.images) <= 6:
        parser.error("--images requires one to six image paths")
    if args.threads < 1 or args.reference_chunk_size < 1:
        parser.error("--threads and --reference-chunk-size must be positive")
    for path in args.images:
        if not path.is_file():
            parser.error(f"Image does not exist: {path}")
    return run_validation(args)


if __name__ == "__main__":
    raise SystemExit(main())
