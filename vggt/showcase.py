"""Readable, outgoing-only attention figures with an explicit query patch.

The spotlight is a within-view rank mask, not a normalized probability map.
Sparse markers are high-attention patches after greedy spatial suppression;
they are not assertions of geometric correspondence. Raw inputs are never changed.
"""

from io import BytesIO
import math
from numbers import Integral
from typing import Mapping

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import ConnectionPatch, Rectangle
from PIL import Image


_INK = "#142b3b"
_MUTED = "#52697a"
_QUERY = "#f4aa28"
_TARGET = "#03a6be"
_PAPER = "#f7f9fc"


def _attention_array(attention):
    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().float().numpy()
    values = np.array(attention, dtype=np.float32, copy=True)
    if values.ndim != 2 or not values.size:
        raise ValueError("attention must be a nonempty two-dimensional patch grid")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("attention must contain finite nonnegative raw values")
    return values


def select_top_attention(attention, top_percent=10):
    """Return patches strictly above a within-view percentile cutoff.

    Ranking is independent for each target view. Zero-weight patches are never
    selected, including an all-zero view. Values equal to the cutoff are not
    selected; constant maps have no distinct region rather than an arbitrary mask.
    """
    if (isinstance(top_percent, bool) or not np.isfinite(top_percent)
            or not 0 < top_percent <= 100):
        raise ValueError("top_percent must be in (0, 100]")
    values = _attention_array(attention)
    if float(values.max()) == float(values.min()):
        return np.zeros_like(values, dtype=bool)
    cutoff = np.percentile(values, 100 - top_percent)
    return (values > cutoff) & (values > 0)


def select_attention_peaks(attention, top_k=3, suppression_radius=2):
    """Greedily select raw maxima, suppressing a square patch neighborhood.

    Scores and order are the original raw attention values. Equal scores use
    row-major order to make selection deterministic. With radius two, a selected
    patch suppresses its surrounding 5 x 5 patch region before the next maximum.
    Constant maps have no distinct targets. No positive scores means no markers,
    and fewer than K markers may be returned.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, Integral) or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if (isinstance(suppression_radius, bool) or not isinstance(suppression_radius, Integral)
            or suppression_radius < 0):
        raise ValueError("suppression_radius must be a nonnegative integer")
    values = _attention_array(attention)
    if float(values.max()) == float(values.min()):
        return []
    remaining = values.copy()
    peaks = []
    height, width = values.shape
    for rank in range(1, top_k + 1):
        flat_index = int(np.argmax(remaining))
        row, col = divmod(flat_index, width)
        score = float(values[row, col])
        if float(remaining[row, col]) <= 0:
            break
        peaks.append({"rank": rank, "row": row, "col": col, "score": score})
        remaining[max(0, row - suppression_radius):min(height, row + suppression_radius + 1),
                  max(0, col - suppression_radius):min(width, col + suppression_radius + 1)] = -1
    return peaks


def _validate(images, data, patch_size, dpi):
    if isinstance(patch_size, bool) or not isinstance(patch_size, Integral) or patch_size < 1:
        raise ValueError("patch_size must be a positive integer")
    if isinstance(dpi, bool) or not isinstance(dpi, Integral) or dpi < 1:
        raise ValueError("dpi must be a positive integer")
    images = torch.as_tensor(images).detach().cpu().float()
    if images.ndim != 4 or images.shape[1] != 3 or not 1 <= len(images) <= 6:
        raise ValueError("images must have shape [views, 3, height, width], with 1 to 6 views")
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("images must contain finite RGB values in [0, 1]")
    _, _, height, width = images.shape
    expected = (height // patch_size, width // patch_size)
    if min(expected) < 1:
        raise ValueError("images must contain at least one complete patch")
    rows = data["attention_maps_row"]
    if len(rows) != len(images):
        raise ValueError("one outgoing attention grid is required for every input view")
    maps = [_attention_array(item) for item in rows]
    if any(item.shape != expected for item in maps):
        raise ValueError(f"outgoing attention grids must have shape {expected}")
    source = data["source_img_idx"]
    patch_y, patch_x = data["source_patch"]
    for value, bound, name in ((source, len(images), "source view"),
                               (patch_y, expected[0], "patch row"),
                               (patch_x, expected[1], "patch column")):
        if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < bound:
            raise ValueError(f"{name} is outside the input grid")
    rgb = images.permute(0, 2, 3, 1).numpy()
    return rgb, maps, int(source), int(patch_y), int(patch_x)


def _axes(fig, x, y, width, height):
    fw, fh = fig.get_size_inches()
    ax = fig.add_axes([x / fw, y / fh, width / fw, height / fh])
    ax.set_axis_off()
    return ax


def _image(ax, rgb):
    height, width = rgb.shape[:2]
    ax.imshow(rgb, extent=(0, width, height, 0), interpolation="nearest")
    return height, width


def _patch_box(ax, row, col, patch_size, color, width=2):
    ax.add_patch(Rectangle((col * patch_size, row * patch_size), patch_size, patch_size,
                           edgecolor=color, facecolor="none", linewidth=width))


def _context(rgb, patch_y, patch_x, patch_size):
    """A five-patch crop, shifted at image edges without moving the query."""
    ph, pw = rgb.shape[0] // patch_size, rgb.shape[1] // patch_size
    x0 = min(max(0, patch_x - 2), max(0, pw - 5))
    y0 = min(max(0, patch_y - 2), max(0, ph - 5))
    x1, y1 = min(pw, x0 + 5), min(ph, y0 + 5)
    return (rgb[y0 * patch_size:y1 * patch_size, x0 * patch_size:x1 * patch_size],
            patch_y - y0, patch_x - x0)


def _source_rail(fig, rgb, source, py, px, patch_size, body_top, body_height):
    x, width = 0.45, 3.15
    image_height = min(2.15, body_height * 0.39)
    image_bottom = body_top - image_height - 0.20
    original = _axes(fig, x, image_bottom, width, image_height)
    _image(original, rgb)
    original.set_title(f"SOURCE VIEW {source} / QUERY", size=11, color=_INK,
                       loc="left", pad=9, weight="bold")
    _patch_box(original, py, px, patch_size, _QUERY, width=2.3)
    original.annotate("QUERY", xy=((px + 0.5) * patch_size, (py + 0.5) * patch_size),
                      xytext=(0.04, 0.92), textcoords="axes fraction", color=_QUERY,
                      fontsize=9, weight="bold",
                      bbox={"facecolor": "#142b3b", "edgecolor": "none", "alpha": 0.85, "pad": 4},
                      arrowprops={"arrowstyle": "->", "color": _QUERY, "linewidth": 1.2})
    crop, local_y, local_x = _context(rgb, py, px, patch_size)
    crop_height = min(1.72, body_height * 0.32)
    crop_bottom = image_bottom - 0.75 - crop_height
    crop_width = crop_height * crop.shape[1] / crop.shape[0]
    zoom = _axes(fig, x + (width - crop_width) / 2, crop_bottom, crop_width, crop_height)
    _image(zoom, crop)
    _patch_box(zoom, local_y, local_x, patch_size, _QUERY, width=2.5)
    for position in range(patch_size, crop.shape[1], patch_size):
        zoom.axvline(position, color="white", alpha=0.45, linewidth=0.6)
    for position in range(patch_size, crop.shape[0], patch_size):
        zoom.axhline(position, color="white", alpha=0.45, linewidth=0.6)
    zoom.set_title(f"QUERY DETAIL / row {py}, col {px}", size=10, color=_INK,
                   pad=10, weight="bold")
    zoom.text(0.5, -0.14,
              f"{crop.shape[0] // patch_size} × {crop.shape[1] // patch_size} patch context · model input",
              transform=zoom.transAxes, ha="center", va="top", size=8, color=_MUTED)
    return zoom, ((local_x + 0.5) * patch_size, (local_y + 0.5) * patch_size)


def _caption_heights(captions, dpi, font_size):
    """Measure multiline captions with the same Agg/font settings as the plot."""
    probe = Figure(figsize=(1, 1), dpi=dpi)
    FigureCanvasAgg(probe)
    renderer = probe.canvas.get_renderer()
    heights = {}
    for target, caption in captions.items():
        text = probe.text(0, 0, caption, fontsize=font_size, linespacing=1.4)
        heights[target] = text.get_window_extent(renderer).height / dpi
        text.remove()
    return heights


def _target_layout(rgb, targets, captions, dpi, kind):
    """Reserve image and measured caption space independently in each row.

    Links retain their original image sizes when more peaks are requested. Only
    the space below those images grows, based on actual NMS output counts.
    """
    groups = min(2, len(targets))
    rows = math.ceil(len(targets) / groups)
    panel_width = (12.9 - 4.60) / groups - 0.28
    natural_height = panel_width * rgb[0].shape[0] / rgb[0].shape[1]
    baseline_row = max(5.0, rows * 2.85) / rows
    image_height = (min(baseline_row - 0.75, natural_height)
                    if kind == "links" else natural_height)
    caption_heights = _caption_heights(captions, dpi, 7.8 if kind == "links" else 7.5)
    row_heights = []
    for row in range(rows):
        row_targets = targets[row * groups:(row + 1) * groups]
        caption_height = max(caption_heights[target] for target in row_targets)
        # Title gap + unchanged image + caption gap + actual text + next-row gap.
        required = 0.28 + image_height + 0.14 + caption_height + 0.22
        row_heights.append(max(baseline_row, required))
    return row_heights, image_height


def _new_figure(data, source, py, px, target_count, dpi, kind, row_heights=None):
    groups = min(2, max(1, target_count))
    rows = max(1, math.ceil(target_count / groups))
    body_height = sum(row_heights) if row_heights is not None else max(5.0, rows * 2.85)
    width = 12.9
    # Reserve real space below axes for raw-score captions and the interpretation
    # footer; captions must not be squeezed into the image itself.
    fixture = data.get("fixture")
    extra_header = 0.30 if fixture else 0.0
    height = body_height + 2.4 + extra_header
    fig = Figure(figsize=(width, height), dpi=dpi, facecolor=_PAPER)
    FigureCanvasAgg(fig)
    title = "ONE QUERY. WHERE DOES IT ATTEND?" if kind == "spotlight" else "ONE QUERY. A FEW ATTENTION TARGETS."
    fig.text(0.45 / width, (height - 0.43) / height, title, color=_INK,
             weight="bold", size=20, va="center")
    layer = data.get("layer_name", "global attention")
    heads = data.get("selected_num_heads")
    head_text = f" · mean of {heads} heads" if heads else " · head-mean attention"
    fig.text(0.45 / width, (height - 0.84) / height,
             f"{layer}{head_text} · outgoing only · source view {source}, patch ({py}, {px})",
             color=_MUTED, size=10, va="center")
    if fixture:
        fig.text(0.45 / width, (height - 1.1) / height, fixture, color="#985126", size=8)
    fig.text(0.45 / width, 0.24 / height,
             "These are attention targets, not verified geometric correspondences. Camera/register tokens are excluded from the displayed patches.",
             size=8, color=_MUTED, va="bottom")
    return fig, groups, rows, height - 1.40 - extra_header, body_height


def _finish(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor(), dpi=fig.dpi)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.copy()


def _boundary_segments(mask, patch_size):
    """Trace selected patch edges, never interpolating a false boundary."""
    height, width = mask.shape
    segments = []
    for row, col in np.argwhere(mask):
        x, y = col * patch_size, row * patch_size
        if row == 0 or not mask[row - 1, col]:
            segments.append([(x, y), (x + patch_size, y)])
        if row + 1 == height or not mask[row + 1, col]:
            segments.append([(x, y + patch_size), (x + patch_size, y + patch_size)])
        if col == 0 or not mask[row, col - 1]:
            segments.append([(x, y), (x, y + patch_size)])
        if col + 1 == width or not mask[row, col + 1]:
            segments.append([(x + patch_size, y), (x + patch_size, y + patch_size)])
    return segments


def render_attention_spotlight(images, data: Mapping, patch_size=14, dpi=160, top_percent=10):
    """Show large rank-based target spotlights from raw outgoing maps.

    Selected patches keep their original RGB values; all other pixels are dimmed
    to 34% brightness. A cyan boundary traces the top-percent cutoff separately
    within each target view, strictly above the percentile. Uniform maps are not
    highlighted. Selection does not renormalize raw
    attention. Captions retain raw selected ranges and visible attention mass.
    The source view is shown once, with its query and enlarged local context.
    """
    rgb, maps, source, py, px = _validate(images, data, patch_size, dpi)
    masks = [select_top_attention(item, top_percent) for item in maps]
    targets = [source] if len(rgb) == 1 else [index for index in range(len(rgb)) if index != source]
    captions = {}
    for target in targets:
        mask = masks[target]
        selected = maps[target][mask]
        score_range = f"{selected.min():.2e}–{selected.max():.2e}" if selected.size else "none"
        selection_text = (f"Selected {int(mask.sum())}/{mask.size} patches · raw A {score_range}"
                          if mask.any() else "No distinct regions above this percentile cutoff")
        captions[target] = (f"{selection_text}\n"
                            f"Visible outgoing mass in this view: {maps[target].sum():.4f}")
    row_heights, image_height = _target_layout(rgb, targets, captions, dpi, "spotlight")
    fig, groups, rows, body_top, body_height = _new_figure(
        data, source, py, px, len(targets), dpi, "spotlight", row_heights)
    _source_rail(fig, rgb[source], source, py, px, patch_size, body_top, body_height)
    fw, fh = fig.get_size_inches()
    right_x, right_width = 4.15, fw - 4.60
    group_width = right_width / groups
    fig.text(right_x / fw, (body_top + 0.22) / fh,
             f"Above the {100 - top_percent:g}th percentile within each view · rank threshold, not probability color",
             size=10, color=_INK, weight="bold")
    if not targets:
        _empty_targets(fig, right_x, body_top)
    for index, target in enumerate(targets):
        row, group = divmod(index, groups)
        x = right_x + group * group_width
        top = body_top - sum(row_heights[:row])
        panel_width = group_width - 0.28
        bottom = top - image_height - 0.28
        view_label = f"VIEW {target}" + (" / SAME-VIEW" if len(rgb) == 1 else "")
        attention = _axes(fig, x, bottom, panel_width, image_height)
        mask = masks[target]
        pixel_mask = np.repeat(np.repeat(mask, patch_size, axis=0), patch_size, axis=1)
        visible = np.zeros(rgb[target].shape[:2], dtype=bool)
        visible[:pixel_mask.shape[0], :pixel_mask.shape[1]] = pixel_mask
        spotlight = (rgb[target].copy() * np.where(visible[..., None], 1.0, 0.34)
                     if mask.any() else rgb[target].copy())
        _image(attention, spotlight)
        attention.add_collection(LineCollection(_boundary_segments(mask, patch_size), colors=_TARGET, linewidths=0.9))
        attention.set_title(f"{view_label} / SPOTLIGHT", loc="left", size=10, color=_INK, pad=8, weight="bold")
        fig.text(x / fw, (bottom - 0.14) / fh, captions[target],
                 va="top", size=7.5, color=_MUTED, linespacing=1.4)
    return _finish(fig)


def _empty_targets(fig, x, top):
    fw, fh = fig.get_size_inches()
    fig.text(x / fw, (top - 1.5) / fh,
             "Add another view to inspect cross-view attention.\nThe query is not repeated as a target map.",
             size=13, color=_MUTED, linespacing=1.7)


def render_attention_links(images, data: Mapping, patch_size=14, dpi=160, top_k=3):
    """Mark spaced raw-attention maxima and link each target view's rank one.

    Greedy NMS suppresses a 5 x 5 patch neighborhood after every selection. Labels
    rank the remaining peaks *within each target view*, with raw A values printed
    beneath the image. Lines join actual query/target patch centers; they do not
    encode a geometric match or interpolation. Original image colors are retained.
    """
    rgb, maps, source, py, px = _validate(images, data, patch_size, dpi)
    peaks = [select_attention_peaks(item, top_k) for item in maps]
    targets = [source] if len(rgb) == 1 else [index for index in range(len(rgb)) if index != source]
    captions = {
        target: ("\n".join(
            f"{item['rank']}  (row {item['row']}, col {item['col']})  raw A = {item['score']:.3e}"
            for item in peaks[target]
        ) or "No distinct outgoing-attention peaks in this view.")
        for target in targets
    }
    row_heights, image_height = _target_layout(rgb, targets, captions, dpi, "links")
    fig, groups, rows, body_top, body_height = _new_figure(
        data, source, py, px, len(targets), dpi, "links", row_heights)
    query_axes, query_center = _source_rail(fig, rgb[source], source, py, px, patch_size, body_top, body_height)
    fw, fh = fig.get_size_inches()
    right_x, right_width = 4.15, fw - 4.60
    group_width = right_width / groups
    fig.text(right_x / fw, (body_top + 0.22) / fh,
             f"Up to {top_k} spaced peaks per view · marker numbers are within-view ranks",
             size=9.5, color=_INK, weight="bold")
    if not targets:
        _empty_targets(fig, right_x, body_top)
    for index, target in enumerate(targets):
        row, group = divmod(index, groups)
        x = right_x + group * group_width
        top = body_top - sum(row_heights[:row])
        panel_width = group_width - 0.28
        bottom = top - image_height - 0.28
        target_axes = _axes(fig, x, bottom, panel_width, image_height)
        _image(target_axes, rgb[target])
        view_label = f"VIEW {target}" + (" / SAME-VIEW" if len(rgb) == 1 else "")
        target_axes.set_title(f"{view_label} / RAW ATTENTION PEAKS", loc="left", size=10, color=_INK, pad=8, weight="bold")
        for peak in peaks[target]:
            center = ((peak["col"] + 0.5) * patch_size, (peak["row"] + 0.5) * patch_size)
            _patch_box(target_axes, peak["row"], peak["col"], patch_size, _TARGET, width=1.4)
            # Keep edge labels inside the image; the numbered badge may move,
            # while the underlying box and rank-one link keep exact patch centers.
            offset_x = -9 if peak["col"] >= maps[target].shape[1] - 2 else 9
            offset_y = -9 if peak["row"] < 2 else 9
            target_axes.annotate(str(peak["rank"]), xy=center, xytext=(offset_x, offset_y), textcoords="offset points",
                                 fontsize=8, weight="bold", color="white", ha="center", va="center",
                                 bbox={"boxstyle": "circle,pad=0.25", "facecolor": _TARGET, "edgecolor": "white", "linewidth": 0.8})
            if peak["rank"] == 1:
                fig.add_artist(ConnectionPatch(
                    xyA=query_center, coordsA="data", axesA=query_axes,
                    xyB=center, coordsB="data", axesB=target_axes,
                    arrowstyle="->", mutation_scale=10, linewidth=0.9,
                    color=_TARGET, alpha=0.6, shrinkA=4, shrinkB=2,
                ))
        fig.text(x / fw, (bottom - 0.14) / fh, captions[target],
                 va="top", size=7.8, color=_MUTED, linespacing=1.4)
    fig.text(right_x / fw, 0.57 / fh,
             "Lines point to rank 1 only. Scores stay raw; no per-view renormalization. NMS radius: 2 patches.",
             size=8, color=_MUTED)
    return _finish(fig)
