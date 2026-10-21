"""Presentation options must not change raw attention or cross Tk threads."""

import json
import threading
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from interactive_attention_tkinter import TkinterAttentionVisualizer, build_parser as gui_parser
from vggt.attention_cli import create_demo, main
from vggt.visualization import DISPLAY_STYLES, render_attention_view


@pytest.fixture(scope="module")
def demo_inputs():
    return create_demo()


@pytest.mark.parametrize("style", ["unknown", "", None, "incoming"])
def test_dispatcher_rejects_unknown_style_before_rendering(style):
    with pytest.raises(ValueError, match="style"):
        render_attention_view(None, None, style=style)


def test_dispatcher_passes_style_specific_options(monkeypatch):
    import vggt.showcase as showcase
    import vggt.visualization as visualization

    calls = []
    image, data, sentinel = object(), object(), object()

    def renderer(name):
        def capture(*args, **kwargs):
            calls.append((name, args, kwargs))
            return sentinel
        return capture

    monkeypatch.setattr(showcase, "render_attention_spotlight", renderer("spotlight"))
    monkeypatch.setattr(showcase, "render_attention_links", renderer("links"))
    monkeypatch.setattr(visualization, "render_attention_comparison", renderer("raw"))
    assert render_attention_view(image, data, patch_size=7, dpi=45, top_percent=12.5) is sentinel
    assert render_attention_view(image, data, style="links", patch_size=7, dpi=45, top_k=5) is sentinel
    assert render_attention_view(image, data, style="raw", patch_size=7, dpi=45) is sentinel
    assert calls == [
        ("spotlight", (image, data), {"patch_size": 7, "dpi": 45, "top_percent": 12.5}),
        ("links", (image, data), {"patch_size": 7, "dpi": 45, "top_k": 5}),
        ("raw", (image, data), {"patch_size": 7, "dpi": 45, "shared_scale": True}),
    ]


@pytest.mark.parametrize("options", [
    ["--style", "unknown"], ["--top-percent", "0"], ["--top-percent", "100"],
    ["--top-percent", "-1"], ["--top-percent", "nan"], ["--top-percent", "inf"],
    ["--top-percent", "not-a-number"], ["--top-k", "0"], ["--top-k", "-1"],
    ["--top-k", "1.5"],
])
def test_cli_invalid_display_options_fail_before_inference(options, monkeypatch):
    def unexpected_inference(*args, **kwargs):
        pytest.fail("invalid display options reached model inference")

    monkeypatch.setattr("vggt.attention_cli.create_demo", unexpected_inference)
    with pytest.raises(SystemExit) as error:
        main(["--demo", *options])
    assert error.value.code == 2


def test_three_cli_styles_export_identical_raw_arrays_and_describe_display(
    demo_inputs, tmp_path, monkeypatch,
):
    images, data, info = demo_inputs
    monkeypatch.setattr("vggt.attention_cli.create_demo", lambda *args: (images, data, info))
    images_before = images.clone()
    rows_before = [item.clone() for item in data["attention_maps_row"]]
    columns_before = [item.clone() for item in data["attention_maps_col"]]
    exports, pngs = [], []
    for style in DISPLAY_STYLES:
        output = tmp_path / f"{style}.png"
        assert main(["--demo", "--style", style, "--top-percent", "12.5", "--top-k", "5",
                     "--output", str(output)]) == 0
        metadata = json.loads(output.with_suffix(".json").read_text())
        display = metadata["display"]
        assert display["style"] == style
        assert display["direction"] == ("outgoing + incoming" if style == "raw" else "outgoing")
        assert display["spotlight_above_percentile"] == (87.5 if style == "spotlight" else None)
        assert display["separated_target_peaks"] == (5 if style == "links" else None)
        with np.load(output.with_suffix(".npz")) as archive:
            exports.append({key: archive[key].copy() for key in archive.files})
        with Image.open(output) as rendered:
            assert rendered.format == "PNG" and rendered.width > 1000
        pngs.append(output.read_bytes())
    assert len(set(pngs)) == 3
    for export in exports:
        assert export.keys() == {"outgoing", "incoming", "images"}
        np.testing.assert_array_equal(export["outgoing"], np.stack(rows_before))
        np.testing.assert_array_equal(export["incoming"], np.stack(columns_before))
        np.testing.assert_array_equal(export["images"], images_before.numpy())
    torch.testing.assert_close(images, images_before, rtol=0, atol=0)
    for before, after in zip(rows_before + columns_before,
                             data["attention_maps_row"] + data["attention_maps_col"]):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


def test_real_cli_defaults_to_layer_11_and_spotlight_without_downloading(tmp_path, monkeypatch):
    from vggt.models.vggt_vis import VGGT

    images = torch.zeros(1, 3, 28, 42)
    calls = []

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def enable_attention_caching(self, layers, max_heads):
            calls.append(("cache", layers, max_heads))

        def aggregator(self, inputs):
            assert tuple(inputs.shape) == (1, 1, 3, 28, 42)

        def extract_patch_attention(self, source, row, col, layer):
            calls.append(("extract", source, row, col, layer))
            return {"source_img_idx": source, "source_patch": (row, col), "layer_name": layer,
                    "attention_maps_row": [torch.full((2, 3), 0.05)],
                    "attention_maps_col": [torch.full((2, 3), 0.03)],
                    "selected_num_heads": 16, "original_num_heads": 16}

        def get_attention_cache_info(self):
            return {"cached_layers": ["global_block_11"]}

    def load_checkpoint(model, **kwargs):
        calls.append(("checkpoint", model, kwargs))
        return Model()

    monkeypatch.setattr(VGGT, "from_pretrained", load_checkpoint)
    monkeypatch.setattr("vggt.utils.load_fn.load_and_preprocess_images", lambda paths: images)
    path = tmp_path / "scene.png"
    Image.new("RGB", (42, 28)).save(path)
    output = tmp_path / "attention.png"
    assert main([str(path), "--device", "cpu", "--output", str(output)]) == 0
    assert calls == [("checkpoint", "facebook/VGGT-1B", {"strict": True}),
                     ("cache", ["global_block_11"], None),
                     ("extract", 0, 1, 1, "global_block_11")]
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["layer"] == "global_block_11"
    assert metadata["display"]["style"] == "spotlight"


@pytest.mark.parametrize("style", DISPLAY_STYLES)
def test_gui_parser_and_constructor_accept_display_styles(style):
    args = gui_parser().parse_args(["--style", style])
    viewer = TkinterAttentionVisualizer(style=args.style)
    assert viewer.style == style and viewer.layer == "global_block_11"
    assert viewer.model is None and viewer.root is None


def test_gui_rejects_invalid_style_without_opening_window():
    with pytest.raises(ValueError, match="style"):
        TkinterAttentionVisualizer(style="unknown")
    with pytest.raises(SystemExit) as error:
        gui_parser().parse_args(["--style", "unknown"])
    assert error.value.code == 2


def test_gui_style_change_reads_tk_only_on_main_thread_and_reuses_cache(monkeypatch):
    import vggt.visualization as visualization

    viewer = TkinterAttentionVisualizer()
    viewer.create_image_grid(torch.zeros(2, 3, 28, 42))
    coordinate = viewer.get_coordinate_info(14, 14)
    viewer._selected_patch = coordinate
    main_thread = threading.get_ident()
    accesses, renders, extractions, results = [], [], [], []

    class MainThreadObject:
        def __init__(self, name):
            self.name = name

        def check(self, method, *args):
            assert threading.get_ident() == main_thread, f"worker touched Tk {self.name}.{method}"
            accesses.append((self.name, method, args))

        def get(self):
            self.check("get")
            return "links"

        def configure(self, **kwargs):
            self.check("configure", kwargs)

        def start(self, value):
            self.check("start", value)

        def stop(self):
            self.check("stop")

        def after(self, *args):
            self.check("after", *args)

    data = {"source_img_idx": 0, "source_patch": (1, 1), "layer_name": "global_block_11"}

    class CachedModel:
        def extract_patch_attention(self, *args):
            assert threading.get_ident() != main_thread
            extractions.append(args)
            return data

        def aggregator(self, *args):
            pytest.fail("style changes must reuse the existing attention cache")

    def render(images, attention, **kwargs):
        assert threading.get_ident() != main_thread
        assert attention is data
        renders.append(kwargs)
        return Image.new("RGB", (20, 20))

    def show_result(*args):
        assert threading.get_ident() == main_thread
        results.append(args)

    viewer.style_choice = MainThreadObject("style")
    viewer.high_quality_mode = MainThreadObject("quality")
    viewer.root = MainThreadObject("root")
    viewer.load_button = MainThreadObject("load")
    viewer.style_selector = MainThreadObject("selector")
    viewer.progress = MainThreadObject("progress")
    viewer.info_label = MainThreadObject("info")
    viewer.model = CachedModel()
    monkeypatch.setattr(visualization, "render_attention_view", render)
    monkeypatch.setattr(viewer, "show_result_window", show_result)
    viewer.on_style_changed()
    viewer._worker.join(timeout=3)
    assert not viewer._worker.is_alive()
    assert viewer._busy and viewer.style == "links"
    assert extractions == [(0, 1, 1, "global_block_11")]
    assert renders == [{"patch_size": 14, "dpi": 200, "style": "links"}]
    assert viewer._events.queue[0][0] == "attention"
    viewer._drain_events()
    assert not viewer._busy and len(results) == 1
    assert sum(name == "style" and method == "get" for name, method, _ in accesses) == 1
    assert not any(name == "quality" and method == "get" for name, method, _ in accesses)
    assert ("selector", "configure", ({"state": "disabled"},)) in accesses
    assert ("selector", "configure", ({"state": "readonly"},)) in accesses


@pytest.mark.parametrize("selected,busy", [(False, False), (True, True)])
def test_gui_style_change_does_not_start_unnecessary_worker(selected, busy, monkeypatch):
    viewer = TkinterAttentionVisualizer()
    viewer.style_choice = SimpleNamespace(get=lambda: "raw")
    viewer._selected_patch = {"image_idx": 0} if selected else None
    viewer._busy = busy
    monkeypatch.setattr(viewer, "_start_worker", lambda *args: pytest.fail("unexpected worker"))
    viewer.on_style_changed()
    assert viewer.style == "raw"
