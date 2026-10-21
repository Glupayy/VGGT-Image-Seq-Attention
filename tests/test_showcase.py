"""Readable showcase selection/rendering without Tk or pretrained weights."""

import numpy as np
from PIL import Image
import pytest
import torch

from vggt.showcase import (
    render_attention_links,
    render_attention_spotlight,
    select_attention_peaks,
    select_top_attention,
)


def test_top_percent_uses_rank_and_strict_percentile_cutoff():
    values = np.arange(20, dtype=np.float32).reshape(4, 5) / 100
    mask = select_top_attention(values, 10)
    np.testing.assert_array_equal(np.flatnonzero(mask), [18, 19])
    tied = np.array([[0.0, 0.1], [0.3, 0.3]])
    assert not select_top_attention(tied, 10).any()
    assert not select_top_attention(np.zeros((2, 3)), 10).any()
    assert not select_top_attention(np.full((3, 3), 0.1), 10).any()


def test_nms_keeps_highest_raw_value_and_suppresses_neighboring_peaks():
    values = np.zeros((8, 9), dtype=np.float32)
    values[1, 1], values[1, 2], values[6, 7], values[4, 1] = 0.9, 0.8, 0.7, 0.6
    peaks = select_attention_peaks(values, top_k=3)
    assert [(item["row"], item["col"]) for item in peaks] == [(1, 1), (6, 7), (4, 1)]
    assert [item["rank"] for item in peaks] == [1, 2, 3]
    assert [item["score"] for item in peaks] == pytest.approx([0.9, 0.7, 0.6])
    for left, right in zip(peaks, peaks[1:]):
        assert max(abs(left["row"] - right["row"]), abs(left["col"] - right["col"])) > 2


def test_nms_ties_and_exhausted_grid_are_deterministic():
    values = np.array([[0.25, 0.25], [0.25, 0.1]])
    peaks = select_attention_peaks(values, top_k=3)
    assert len(peaks) == 1 and (peaks[0]["row"], peaks[0]["col"]) == (0, 0)
    assert select_attention_peaks(np.zeros((3, 3))) == []
    assert select_attention_peaks(np.full((3, 3), 0.25)) == []
    assert len(select_attention_peaks(values, top_k=3, suppression_radius=0)) == 3


@pytest.mark.parametrize("renderer", [render_attention_spotlight, render_attention_links])
@pytest.mark.parametrize("count", [1, 2, 6])
def test_headless_render_keeps_images_and_raw_values_unchanged(renderer, count):
    images = torch.linspace(0, 1, count * 3 * 42 * 56).reshape(count, 3, 42, 56).bfloat16()
    outgoing = [torch.arange(1, 13).reshape(3, 4).float() / 100 for _ in range(count)]
    data = {"source_img_idx": 0, "source_patch": (0, 0), "attention_maps_row": outgoing,
            "layer_name": "global_block_23", "selected_num_heads": 16}
    before_images = images.clone()
    before_maps = [item.clone() for item in outgoing]
    result = renderer(images, data, dpi=35)
    assert isinstance(result, Image.Image) and min(result.size) > 200
    assert data["source_patch"] == (0, 0)
    assert "attention_maps_col" not in data
    torch.testing.assert_close(images, before_images, rtol=0, atol=0)
    for before, after in zip(before_maps, outgoing):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize("percent", [0, -1, 101, float("nan"), True])
def test_invalid_top_percent_rejected(percent):
    with pytest.raises(ValueError, match="top_percent"):
        select_top_attention(np.ones((3, 3)), percent)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_peak_count_rejected(top_k):
    with pytest.raises(ValueError, match="top_k"):
        select_attention_peaks(np.ones((3, 3)), top_k)


def test_selection_helpers_preserve_original_raw_values():
    values = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.bfloat16)
    before = values.clone()
    select_top_attention(values)
    select_attention_peaks(values)
    torch.testing.assert_close(values, before, rtol=0, atol=0)


def test_links_end_on_actual_highest_patch_and_do_not_repeat_source(monkeypatch):
    import vggt.showcase as showcase

    images = torch.zeros(2, 3, 84, 112)
    source = torch.ones(6, 8) / 100
    target = torch.zeros(6, 8)
    target[4, 5] = 0.2
    data = {"source_img_idx": 0, "source_patch": (2, 3),
            "attention_maps_row": [source, target]}
    monkeypatch.setattr(showcase, "_finish", lambda fig: fig)
    figure = showcase.render_attention_links(images, data)
    assert len(figure.artists) == 1
    line = figure.artists[0]
    assert line.xy1 == (35, 35)  # selected patch center within the 5 x 5 crop
    assert line.xy2 == (77, 63)  # actual (row 4, col 5) target patch center
    titles = [axes.get_title(loc="left") for axes in figure.axes]
    assert sum("RAW ATTENTION PEAKS" in title for title in titles) == 1
    assert any("VIEW 1 / RAW ATTENTION PEAKS" in title for title in titles)


@pytest.mark.parametrize("renderer", [render_attention_spotlight, render_attention_links])
def test_single_view_is_explicitly_labelled_same_view(renderer, monkeypatch):
    import vggt.showcase as showcase

    images = torch.zeros(1, 3, 42, 56)
    data = {"source_img_idx": 0, "source_patch": (0, 0),
            "attention_maps_row": [torch.zeros(3, 4)]}
    monkeypatch.setattr(showcase, "_finish", lambda fig: fig)
    figure = renderer(images, data)
    assert any("SAME-VIEW" in axes.get_title(loc="left") for axes in figure.axes)
    assert not figure.artists  # a uniform map has no distinct peaks or links


@pytest.mark.parametrize("top_k", [3, 10, 30])
def test_many_link_scores_fit_below_images_and_above_footer(top_k, monkeypatch):
    import vggt.showcase as showcase

    images = torch.zeros(6, 3, 280, 420)
    values = torch.arange(1, 601).reshape(20, 30).float() / 100_000
    data = {"source_img_idx": 0, "source_patch": (10, 15),
            "attention_maps_row": [values.clone() for _ in range(6)]}
    monkeypatch.setattr(showcase, "_finish", lambda fig: fig)
    figure = showcase.render_attention_links(images, data, dpi=65, top_k=top_k)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    captions = [text for text in figure.texts if "raw A =" in text.get_text()]
    footers = [text for text in figure.texts if text.get_text().startswith(
        ("Lines point", "These are attention targets"))]
    footer_top = max(text.get_window_extent(renderer).y1 for text in footers)
    target_axes = [axes for axes in figure.axes if "RAW ATTENTION PEAKS" in axes.get_title(loc="left")]
    assert len(captions) == len(target_axes) == 5
    for caption, axes in zip(captions, target_axes):
        assert len(caption.get_text().splitlines()) == top_k
        bounds = caption.get_window_extent(renderer)
        assert bounds.y0 > footer_top
        assert 0 <= bounds.x0 < bounds.x1 < figure.bbox.width
        assert bounds.y1 < axes.get_window_extent(renderer).y0
        # Requested extra scores never shrink the default three-peak images.
        assert axes.get_position().height * figure.get_figheight() == pytest.approx(2.1)
        image_bounds = axes.get_window_extent(renderer)
        for label in axes.texts:
            label_bounds = label.get_window_extent(renderer)
            assert label_bounds.x0 >= image_bounds.x0
            assert label_bounds.x1 <= image_bounds.x1
            assert label_bounds.y0 >= image_bounds.y0
            assert label_bounds.y1 <= image_bounds.y1
    # Upper-row captions must also clear the next row's title, not only the footer.
    for index, caption in enumerate(captions[:-2]):
        lower_title = target_axes[index + 2]._left_title.get_window_extent(renderer)
        assert caption.get_window_extent(renderer).y0 > lower_title.y1


def test_spotlight_uses_large_single_target_panels(monkeypatch):
    import vggt.showcase as showcase

    images = torch.zeros(3, 3, 392, 518)
    data = {"source_img_idx": 0, "source_patch": (14, 18),
            "attention_maps_row": [torch.arange(28 * 37).reshape(28, 37).float() / 10_000
                                   for _ in range(3)]}
    monkeypatch.setattr(showcase, "_finish", lambda fig: fig)
    figure = showcase.render_attention_spotlight(images, data, dpi=45)
    figure.canvas.draw()
    target_axes = [axes for axes in figure.axes if "/ SPOTLIGHT" in axes.get_title(loc="left")]
    assert figure.get_figwidth() == pytest.approx(12.9)
    assert len(figure.axes) == 4  # source + query detail + two targets
    assert len(target_axes) == 2
    assert all(axes.get_position().width * figure.get_figwidth() > 3.8 for axes in target_axes)
    assert not any("ORIGINAL" in axes.get_title(loc="left") for axes in figure.axes)
    assert sum("Visible outgoing mass" in text.get_text() for text in figure.texts) == 2
