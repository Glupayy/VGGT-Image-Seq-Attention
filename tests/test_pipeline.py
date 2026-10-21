"""Exercise attention capture through the real aggregator and export path."""

import json
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from vggt.attention_cli import create_demo, main
from vggt.visualization import render_attention_comparison


def test_demo_is_repeatable_and_exports_raw_maps(tmp_path):
    output = tmp_path / "attention.png"
    assert main(["--demo", "--output", str(output)]) == 0
    metadata = json.loads(output.with_suffix(".json").read_text())
    assert metadata["kind"] == "synthetic-tiny-model"
    assert metadata["source_patch_row_col"] == [2, 3]
    assert metadata["selected_num_heads"] == metadata["original_num_heads"] == 4
    with np.load(output.with_suffix(".npz")) as stored:
        outgoing = stored["outgoing"].copy()
        assert outgoing.shape == (3, 6, 8)
        assert np.isfinite(outgoing).all()
        assert stored["incoming"].shape == outgoing.shape
    images, data, _ = create_demo()
    np.testing.assert_array_equal(outgoing, np.stack(data["attention_maps_row"]))
    assert Image.open(output).width > 1000
    before = data["attention_maps_row"][0].clone()
    rendered = render_attention_comparison(images.bfloat16(), data, dpi=50)
    assert isinstance(rendered, Image.Image)
    torch.testing.assert_close(before, data["attention_maps_row"][0])


def test_help_is_lightweight():
    code = "from vggt.attention_cli import main; main(['--help'])"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0
    assert "--demo" in result.stdout
    assert not result.stderr


@pytest.mark.parametrize("args", [["--demo", "--source", "3"], ["--demo", "--layer", "23"],
                                 ["--demo", "--max-heads", "0"], ["--demo", "--patch", "-1", "0"],
                                 ["--demo", "--output", "bad.pdf"],
                                 ["--demo", "--device", "cuda"], ["--demo", "--patch", "6", "0"], []])
def test_cli_rejects_invalid_input(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_visualization_rejects_mismatched_views():
    images, data, _ = create_demo()
    with pytest.raises(ValueError, match="one attention map"):
        render_attention_comparison(images[:2], data)
