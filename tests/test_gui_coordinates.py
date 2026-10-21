"""GUI coordinate and launch tests that need neither a display nor weights."""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from vggt.attention_gui import TkinterAttentionVisualizer


@pytest.fixture
def visualizer():
    viewer = TkinterAttentionVisualizer()
    viewer.create_image_grid(torch.zeros(4, 3, 28, 42))
    return viewer


def test_import_and_constructor_do_not_load_model_or_create_window():
    viewer = TkinterAttentionVisualizer()
    assert viewer.root is None
    assert viewer.model is None
    assert viewer.model_loaded is False
    assert viewer._get_high_quality_mode() is True


def test_coordinate_in_second_view(visualizer):
    info = visualizer.get_coordinate_info(42 + 20, 16)
    assert info["image_idx"] == 1
    assert info["patch_coords"] == (1, 1)
    assert info["patch_grid_size"] == (2, 3)
    assert info["local_coords"] == (20, 16)


def test_centered_scaled_image_coordinates(visualizer):
    visualizer.display_offset_x = 40
    visualizer.display_offset_y = 25
    visualizer.display_scale_x = 0.5
    visualizer.display_scale_y = 0.75
    info = visualizer.get_coordinate_info(40 + 20 * 0.5, 25 + 16 * 0.75)
    assert info["image_idx"] == 0
    assert info["patch_coords"] == (1, 1)
    assert info["original_coords"] == (20, 16)
    assert visualizer.get_coordinate_info(39, 25) is None
    assert visualizer.get_coordinate_info(40, 24) is None


@pytest.mark.parametrize("point", [(-1, 0), (0, -1), (126, 0), (0, 56), (45, 30), (float("nan"), 0), (0, float("inf"))])
def test_invalid_region_is_not_a_patch(visualizer, point):
    assert visualizer.get_coordinate_info(*point) is None


def test_exact_patch_and_image_boundaries(visualizer):
    assert visualizer.get_coordinate_info(14, 14)["patch_coords"] == (1, 1)
    assert visualizer.get_coordinate_info(42, 0)["image_idx"] == 1
    assert visualizer.get_coordinate_info(0, 28)["image_idx"] == 3
    assert visualizer.get_coordinate_info(125.999, 27.999)["patch_coords"] == (2, 1)


def test_incomplete_edge_patch_is_excluded():
    viewer = TkinterAttentionVisualizer()
    viewer.create_image_grid(torch.zeros(1, 3, 29, 43))
    assert viewer.get_coordinate_info(42, 2) is None
    assert viewer.get_coordinate_info(2, 28) is None


def test_grid_accepts_bfloat16_without_tk():
    viewer = TkinterAttentionVisualizer()
    viewer.create_image_grid(torch.ones(1, 3, 28, 42, dtype=torch.bfloat16))
    assert viewer.grid_image_pil.size == (42, 28)
    assert viewer.grid_image_pil.getpixel((0, 0)) == (255, 255, 255)
    assert viewer.current_images.dtype == torch.float32


@pytest.mark.parametrize("count", [0, 7])
def test_image_count_is_explicitly_validated(count):
    viewer = TkinterAttentionVisualizer()
    assert viewer.load_images(["unused.png"] * count) is False
    assert "between 1 and 6" in viewer.last_error
    assert viewer.model is None


def test_help_needs_no_tk_window_or_model():
    result = subprocess.run([sys.executable, "-m", "vggt.attention_gui", "--help"],
                            capture_output=True, text=True, check=True)
    assert "--max-heads" in result.stdout
    assert "--device" in result.stdout
    assert "facebook/VGGT-1B" not in result.stderr


def test_import_does_not_import_torch_or_tkinter():
    code = (
        "import sys; import vggt.attention_gui; "
        "assert 'torch' not in sys.modules; assert 'tkinter' not in sys.modules; "
        "assert 'vggt.models.vggt_vis' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_module_help_does_not_import_torch_or_tkinter():
    code = """import runpy
import sys
sys.argv = ['vggt.attention_gui', '--help']
try:
    runpy.run_module('vggt.attention_gui', run_name='__main__')
except SystemExit as error:
    assert error.code == 0
assert 'torch' not in sys.modules
assert 'tkinter' not in sys.modules
assert 'vggt.models.vggt_vis' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)


def test_load_worker_only_queues_results_and_caches_selected_layer(tmp_path, monkeypatch):
    viewer = TkinterAttentionVisualizer(layer="global_block_2")
    input_path = tmp_path / "image.png"
    input_path.touch()
    images = torch.zeros(1, 3, 28, 42)
    calls = []

    class Model:
        def enable_attention_caching(self, layers, max_heads):
            calls.append(("cache", layers, max_heads))

        def start_new_attention_run(self):
            calls.append(("start",))

        def aggregator(self, inputs):
            calls.append(("forward", tuple(inputs.shape)))
            return []

        def get_attention_cache_info(self):
            return {"cached_layers": ["global_block_2"]}

    class NoTk:
        def __getattr__(self, name):
            raise AssertionError(f"Worker tried to access Tk: {name}")

    monkeypatch.setitem(sys.modules, "vggt.utils.load_fn", SimpleNamespace(load_and_preprocess_images=lambda paths: images))
    viewer.root = NoTk()
    viewer.model = Model()
    viewer.model_loaded = True
    viewer.device = "cpu"
    viewer.load_images_thread([str(input_path)])
    assert calls == [("cache", ["global_block_2"], None), ("start",), ("forward", (1, 1, 3, 28, 42))]
    assert viewer.current_images.shape == images.shape
    events = [viewer._events.get_nowait() for _ in range(viewer._events.qsize())]
    assert events[-1] == ("loaded", 1)


def test_failed_load_clears_stale_grid_and_can_be_retried(visualizer, tmp_path):
    missing_path = tmp_path / "missing.png"
    assert visualizer.load_images([str(missing_path)]) is False
    assert visualizer.current_images is None
    assert visualizer.get_coordinate_info(0, 0) is None
    visualizer.create_image_grid(torch.zeros(1, 3, 28, 42))
    assert visualizer.get_coordinate_info(0, 0)["valid"] is True
