"""Verify bundled sample bytes and pinned source mappings without a network."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import re

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = ROOT / "examples/images"
MANIFEST = json.loads((ROOT / "examples/SOURCES.json").read_text(encoding="utf-8"))
REPOSITORY = "https://github.com/facebookresearch/vggt"
SCENES = {
    "lego": ("examples/kitchen/images", ["00.png", "03.png", "06.png"], (779, 520)),
    "flower": ("examples/llff_flower/images", ["000.png", "006.png", "012.png"], (706, 529)),
    "fern": ("examples/llff_fern/images", ["000.png", "005.png", "010.png"], (706, 529)),
}


def test_manifest_lists_exactly_the_nine_bundled_pngs():
    entries = MANIFEST["files"]
    paths = [entry["path"] for entry in entries]
    expected = {f"examples/images/{scene}/{name}"
                for scene, (_, names, _) in SCENES.items() for name in names}
    actual = {path.relative_to(ROOT).as_posix() for path in SAMPLE_ROOT.glob("*/*.png")}
    assert len(entries) == len(paths) == len(set(paths)) == 9
    assert set(paths) == expected == actual
    assert {entry["scene"] for entry in entries} == set(SCENES)
    assert MANIFEST["source_repository"] == REPOSITORY
    for field in ("source_revision", "verified_unchanged_at_revision"):
        assert re.fullmatch(r"[0-9a-f]{40}", MANIFEST[field]), f"{field} must pin a full commit"


@pytest.mark.parametrize("scene", SCENES)
def test_each_scene_preserves_three_views_in_source_frame_order(scene):
    _, expected_names, _ = SCENES[scene]
    entries = [entry for entry in MANIFEST["files"] if entry["scene"] == scene]
    names = [PurePosixPath(entry["path"]).name for entry in entries]
    assert len(entries) == 3
    assert names == expected_names
    assert names == sorted(names, key=lambda name: int(PurePosixPath(name).stem))


@pytest.mark.parametrize("entry", MANIFEST["files"], ids=lambda entry: entry["path"])
def test_sample_content_and_upstream_mapping(entry):
    scene = entry["scene"]
    upstream_directory, expected_names, expected_dimensions = SCENES[scene]
    relative = PurePosixPath(entry["path"])
    assert relative.parts[:3] == ("examples", "images", scene)
    assert len(relative.parts) == 4 and relative.name in expected_names
    assert entry["upstream_path"] == f"{upstream_directory}/{relative.name}"
    assert entry["source_url"] == (
        f"{REPOSITORY}/blob/{MANIFEST['source_revision']}/{entry['upstream_path']}"
    )

    path = ROOT / relative
    contents = path.read_bytes()
    assert isinstance(entry["size_bytes"], int) and entry["size_bytes"] > 0
    assert len(contents) == path.stat().st_size == entry["size_bytes"]
    assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
    assert hashlib.sha256(contents).hexdigest() == entry["sha256"]
    assert re.fullmatch(r"[0-9a-f]{40}", entry["git_blob_sha1"])
    git_blob = f"blob {len(contents)}\0".encode("ascii") + contents
    assert hashlib.sha1(git_blob).hexdigest() == entry["git_blob_sha1"]

    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.size == tuple(entry["image_width_height"]) == expected_dimensions
        image.load()  # Decode the complete image, not just its header.
