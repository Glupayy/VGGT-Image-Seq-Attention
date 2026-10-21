"""Headless attention display dispatch and shared-scale bidirectional plots."""

from io import BytesIO
from typing import Mapping

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image


DISPLAY_STYLES = ("spotlight", "links", "raw")


def render_attention_view(images, patch_attention_data, *, style="spotlight", patch_size=14,
                          dpi=160, top_percent=10, top_k=3):
    """Choose a readable query-focused view or the raw bidirectional analysis."""
    if style == "raw":
        return render_attention_comparison(images, patch_attention_data, patch_size=patch_size,
                                           dpi=dpi, shared_scale=True)
    if style == "spotlight":
        from vggt.showcase import render_attention_spotlight
        return render_attention_spotlight(images, patch_attention_data, patch_size=patch_size,
                                          dpi=dpi, top_percent=top_percent)
    if style == "links":
        from vggt.showcase import render_attention_links
        return render_attention_links(images, patch_attention_data, patch_size=patch_size,
                                      dpi=dpi, top_k=top_k)
    raise ValueError(f"style must be one of {DISPLAY_STYLES}")


def render_attention_comparison(
    images: torch.Tensor,
    patch_attention_data: Mapping,
    patch_size: int = 14,
    dpi: int = 160,
    shared_scale: bool = True,
) -> Image.Image:
    """Render original / outgoing / incoming panels without a GUI backend.

    Values remain raw attention probabilities. A shared linear color scale is
    used across every overlay by default; columns are never re-normalized.
    """
    images = images.detach().to(device="cpu", dtype=torch.float32)
    if images.ndim != 4 or images.shape[1] != 3 or not 1 <= len(images) <= 6:
        raise ValueError("images must have shape [S, 3, H, W], with 1 <= S <= 6")
    if patch_size <= 0 or dpi <= 0:
        raise ValueError("patch_size and dpi must be positive")
    count, _, height, width = images.shape
    if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
        raise ValueError("images must contain finite RGB values in [0, 1]")
    rows = patch_attention_data["attention_maps_row"]
    cols = patch_attention_data["attention_maps_col"]
    if len(rows) != count or len(cols) != count:
        raise ValueError("Each direction must contain one attention map per image")
    maps = [torch.as_tensor(m).detach().cpu().float() for m in [*rows, *cols]]
    expected = (height // patch_size, width // patch_size)
    for attn in maps:
        if tuple(attn.shape) != expected:
            raise ValueError(f"Attention grid must be {expected}, got {tuple(attn.shape)}")
        if not torch.isfinite(attn).all() or (attn < 0).any():
            raise ValueError("Attention maps must be finite and nonnegative")
    source = int(patch_attention_data["source_img_idx"])
    py, px = patch_attention_data["source_patch"]
    if not (0 <= source < count and 0 <= py < expected[0] and 0 <= px < expected[1]):
        raise ValueError("Source image or patch is outside the input grid")

    # Keep related panels together. Six views occupy three rows of two groups.
    groups = 1 if count <= 3 else 2
    nrows = (count + groups - 1) // groups
    fig = Figure(figsize=(10.8 * groups, 2.9 * nrows + 2.05), dpi=dpi, facecolor="#f7f9fc")
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(nrows, 3 * groups, left=0.025, right=0.975,
                           top=1 - 1.55 / fig.get_figheight(), bottom=0.09,
                           wspace=0.045, hspace=0.28)
    fig.text(0.03, 1 - 0.35 / fig.get_figheight(), "VGGT / ATTENTION EXPLORER",
             color="#112b40", weight="bold", size=18, va="center")
    layer = patch_attention_data.get("layer_name", "attention")
    selected = patch_attention_data.get("selected_num_heads")
    heads = f" · {selected} heads averaged" if selected is not None else ""
    fig.text(0.03, 1 - 0.72 / fig.get_figheight(),
             f"{layer} · source view {source} · patch (row {py}, col {px}){heads}",
             color="#52697a", size=10, va="center")
    if patch_attention_data.get("fixture"):
        fig.text(0.03, 1 - 1.0 / fig.get_figheight(), patch_attention_data["fixture"],
                 color="#a04b22", size=9, va="center")
    vmax = max(float(m.max()) for m in maps) or 1.0
    norm = Normalize(vmin=0, vmax=vmax)
    axes = []
    im = None
    for i in range(count):
        group_row, group_col = divmod(i, groups)
        rgb = images[i].permute(1, 2, 0).numpy()
        for j in range(3):
            ax = fig.add_subplot(grid[group_row, group_col * 3 + j])
            axes.append(ax)
            ax.imshow(rgb)
            if j:
                attn = maps[i if j == 1 else count + i]
                upsampled = torch.nn.functional.interpolate(
                    attn[None, None], size=(height, width), mode="bilinear", align_corners=False,
                )[0, 0].numpy()
                panel_norm = norm if shared_scale else Normalize(0, float(attn.max()) or 1.0)
                im = ax.imshow(upsampled, cmap="magma", alpha=0.72, norm=panel_norm)
                direction = "Outgoing · this patch → view" if j == 1 else "Incoming · view → this patch"
                ax.set_title(f"{direction}\nview {i} · patch mass {float(attn.sum()):.4f}",
                             size=9, color="#253e50", loc="left", pad=8)
            else:
                ax.set_title(f"VIEW {i}" + (" / SELECTED PATCH" if i == source else " / INPUT"),
                             size=10, weight="bold", color="#007f85" if i == source else "#253e50",
                             loc="left", pad=8)
            if i == source:
                ax.add_patch(Rectangle((px * patch_size, py * patch_size), patch_size, patch_size,
                                       linewidth=1.8, edgecolor="#46ffdd", facecolor="none"))
            ax.set_axis_off()
    if shared_scale:
        bar_ax = fig.add_axes([0.69, 0.035, 0.27, 0.012])
        bar = fig.colorbar(im, cax=bar_ax, orientation="horizontal", format="%.1e")
        bar.set_ticks(np.linspace(0, vmax, 4))
        bar.ax.tick_params(labelsize=8, colors="#52697a", length=2)
        bar.outline.set_visible(False)
        fig.text(0.69, 0.06, "Raw attention · shared linear scale", size=8, color="#52697a")
    fig.text(0.03, 0.035, "Rows attend out. Columns receive attention.\nSpecial tokens excluded from overlays.",
             size=8, color="#52697a", va="center")
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    buffer.seek(0)
    with Image.open(buffer) as rendered:
        return rendered.copy()
