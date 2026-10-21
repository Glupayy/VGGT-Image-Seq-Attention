"""Exercise the real Hub loader without allocating VGGT's large architecture."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from PIL import Image
from safetensors.torch import save_file


@pytest.fixture
def tiny_vggt(monkeypatch):
    """Keep VGGT + PyTorchModelHubMixin intact; replace expensive submodules."""
    from vggt.models import vggt_vis

    class SmallAggregator(nn.Module):
        def __init__(self, patch_size=14, **kwargs):
            super().__init__()
            self.patch_size = patch_size
            self.projection = nn.Linear(2, 2)

        def set_attention_cache_manager(self, manager):
            self.manager = manager

    monkeypatch.setattr(vggt_vis, "Aggregator", SmallAggregator)
    for name in ("CameraHead", "DPTHead", "TrackHead"):
        monkeypatch.setattr(vggt_vis, name, lambda **kwargs: nn.Linear(2, 2))
    return vggt_vis.VGGT


@pytest.fixture
def checkpoint_directory(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(
        json.dumps({"img_size": 28, "patch_size": 14, "embed_dim": 8}),
        encoding="utf-8",
    )
    return checkpoint


def write_incompatible_checkpoint(checkpoint):
    save_file({"unrelated_key": torch.ones(1)}, str(checkpoint / "model.safetensors"))


def test_matching_local_checkpoint_round_trips_through_real_hub_loader(tiny_vggt, checkpoint_directory):
    original = tiny_vggt(img_size=28, patch_size=14, embed_dim=8)
    save_file(original.state_dict(), str(checkpoint_directory / "model.safetensors"))
    loaded = tiny_vggt.from_pretrained(str(checkpoint_directory), strict=True)
    assert loaded.state_dict().keys() == original.state_dict().keys()
    for name, parameter in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[name], parameter, rtol=0, atol=0)


def test_incompatible_local_checkpoint_is_rejected(tiny_vggt, checkpoint_directory):
    write_incompatible_checkpoint(checkpoint_directory)
    with pytest.raises(RuntimeError, match="Missing key"):
        tiny_vggt.from_pretrained(str(checkpoint_directory), strict=True)


def test_gui_rejects_incompatible_checkpoint(tiny_vggt, checkpoint_directory):
    from vggt.attention_gui import TkinterAttentionVisualizer

    write_incompatible_checkpoint(checkpoint_directory)
    viewer = TkinterAttentionVisualizer(model=str(checkpoint_directory), device="cpu")
    with pytest.raises(RuntimeError, match="Missing key"):
        viewer.load_model()
    assert viewer.model_loaded is False


def test_gui_accepts_matching_checkpoint_without_opening_window(tiny_vggt, checkpoint_directory):
    from vggt.attention_gui import TkinterAttentionVisualizer

    original = tiny_vggt(img_size=28, patch_size=14, embed_dim=8)
    save_file(original.state_dict(), str(checkpoint_directory / "model.safetensors"))
    viewer = TkinterAttentionVisualizer(model=str(checkpoint_directory), device="cpu")
    viewer.load_model()
    assert viewer.model_loaded is True and viewer.root is None
    for name, parameter in original.state_dict().items():
        torch.testing.assert_close(viewer.model.state_dict()[name], parameter, rtol=0, atol=0)


def test_cli_rejects_incompatible_checkpoint(tiny_vggt, checkpoint_directory, tmp_path, capsys):
    from vggt.attention_cli import main

    write_incompatible_checkpoint(checkpoint_directory)
    image = tmp_path / "image.png"
    Image.new("RGB", (28, 28)).save(image)
    with pytest.raises(SystemExit) as error:
        main([str(image), "--model", str(checkpoint_directory), "--device", "cpu"])
    assert error.value.code == 1
    assert "Missing key" in capsys.readouterr().err


def test_validation_script_records_incompatible_checkpoint_as_failed(
    tiny_vggt, checkpoint_directory, tmp_path, monkeypatch,
):
    import sys

    write_incompatible_checkpoint(checkpoint_directory)
    image = tmp_path / "image.png"
    Image.new("RGB", (28, 28)).save(image)
    script = Path(__file__).resolve().parents[1] / "scripts" / "validate_pretrained.py"
    monkeypatch.setattr(sys, "path", sys.path.copy())
    spec = importlib.util.spec_from_file_location("pretrained_validation_smoke", script)
    validation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validation)
    output = tmp_path / "validation"
    threads = torch.get_num_threads()
    try:
        status = validation.run_validation(SimpleNamespace(
            model=str(checkpoint_directory), images=[image], output=output,
            threads=1, reference_chunk_size=8,
        ))
    finally:
        torch.set_num_threads(threads)
    report = json.loads((output / "validation.json").read_text())
    assert status == 1 and report["passed"] is False
    assert report["error"]["type"] == "RuntimeError"
    assert "Missing key" in report["error"]["message"]
