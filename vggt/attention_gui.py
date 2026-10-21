#!/usr/bin/env python3
"""Click image patches to explore VGGT's outgoing and incoming attention.

Importing this module does not create a window, load a model, or download weights.
Run ``python -m vggt.attention_gui --help`` for launch options.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
from pathlib import Path
import queue
import re
import sys
import threading
from typing import Dict, List, Optional

import numpy as np
from PIL import Image


class TkinterAttentionVisualizer:
    """A Tk front end whose worker threads never call the Tk interpreter.

    Coordinate methods and image rendering can also be used without creating a
    window. Coordinates refer to the preprocessed model images, not the original
    files before resizing and cropping.
    """

    def __init__(
        self,
        model: str = "facebook/VGGT-1B",
        device: str = "auto",
        layer: str = "global_block_11",
        max_heads: Optional[int] = None,
        style: str = "spotlight",
    ):
        if not re.fullmatch(r"global_block_(?:[0-9]|1[0-9]|2[0-3])", layer):
            raise ValueError("layer must be global_block_0 through global_block_23")
        if max_heads is not None and max_heads < 1:
            raise ValueError("max_heads must be a positive integer or None")
        if style not in {"spotlight", "links", "raw"}:
            raise ValueError("style must be spotlight, links, or raw")
        self.model_id = model
        self.requested_device = device
        self.device = device
        self.layer = layer
        self.max_heads = max_heads
        self.style = style
        self.style_choice = None
        self.style_selector = None
        self.dtype = None
        self.model = None
        self.model_loaded = False
        self.current_images = None
        self.patch_size = 14
        self.grid_config = None
        self.grid_image_pil = None
        self.grid_image_tk = None
        self.display_scale_x = 1.0
        self.display_scale_y = 1.0
        self.display_offset_x = 0
        self.display_offset_y = 0
        self.high_quality_mode = None
        self._high_quality = True
        self.root = None
        self.canvas = None
        self.info_label = None
        self.load_button = None
        self.result_window = None
        self._selected_patch = None
        self._events = queue.Queue()
        self._worker = None
        self._busy = False
        self._closing = threading.Event()
        self.last_error = None
        self._export_directory = Path.cwd()

    def _get_high_quality_mode(self):
        """Read a Python value so rendering workers never access a Tk variable."""
        return self._high_quality

    def on_quality_mode_changed(self):
        self._high_quality = bool(self.high_quality_mode.get())

    def on_style_changed(self, event=None):
        # Copy the Tk value on the main thread; workers only read plain state.
        self.style = self.style_choice.get()
        if self._selected_patch is not None and not self._busy:
            self._set_busy(True, f"Rendering {self.style} view…")
            self._start_worker(self.show_attention_for_click, (self._selected_patch,))

    def _select_device(self):
        import torch

        device = self.requested_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        resolved = torch.device(device)
        if resolved.type not in {"cpu", "cuda"}:
            raise ValueError("VGGT visualization supports CPU and CUDA devices")
        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA was requested but is unavailable; use --device cpu")
            major, _ = torch.cuda.get_device_capability(resolved)
            self.dtype = torch.bfloat16 if major >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        self.device = str(resolved)
        return resolved

    def load_model(self):
        """Load weights only after the user explicitly selects input images."""
        if self.model_loaded:
            return
        device = self._select_device()
        from vggt.models.vggt_vis import VGGT

        self.model = VGGT.from_pretrained(self.model_id, strict=True).to(device).eval()
        self.patch_size = self.model.aggregator.patch_size
        self.model_loaded = True

    def load_images(self, image_paths: List[str]) -> bool:
        """Preprocess one to six images and cache only the selected global block.

        This method is also usable without a GUI. It updates Python state only;
        the main thread is responsible for displaying the resulting image grid.
        On failure the error is available as ``last_error`` and a new load may
        be attempted without restarting the application.
        """
        try:
            if not 1 <= len(image_paths) <= 6:
                raise ValueError("Select between 1 and 6 images")
            for path in image_paths:
                if not Path(path).is_file():
                    raise FileNotFoundError(f"Image does not exist: {path}")
            import torch
            from vggt.utils.load_fn import load_and_preprocess_images

            # Validate/decode inputs before downloading a potentially large model.
            images = load_and_preprocess_images(image_paths)
            self._events.put(("status", "Loading model weights (first use may download them)…"))
            self.load_model()
            self.model.enable_attention_caching([self.layer], max_heads=self.max_heads)
            self.model.start_new_attention_run()
            self._events.put(("status", f"Computing {self.layer} attention on {self.device}…"))
            device = torch.device(self.device)
            autocast = (
                torch.autocast(device_type="cuda", dtype=self.dtype)
                if device.type == "cuda"
                else nullcontext()
            )
            with torch.inference_mode(), autocast:
                # The geometric prediction heads are unnecessary for attention.
                features = self.model.aggregator(images.to(device).unsqueeze(0))
                del features
            cached = self.model.get_attention_cache_info().get("cached_layers", [])
            if self.layer not in cached:
                raise RuntimeError(f"No attention was cached for {self.layer}")
            self.create_image_grid(images)
            self.last_error = None
            return True
        except Exception as exc:
            self.last_error = str(exc)
            # A failed run must not leave an old grid paired with a new cache.
            self.current_images = None
            self.grid_config = None
            self.grid_image_pil = None
            if self.model is not None:
                self.model.clear_attention_cache()
            return False

    def create_image_grid(self, images):
        """Create a display grid without interacting with Tk (safe in a worker)."""
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("Expected images with shape [views, 3, height, width]")
        count, _, height, width = images.shape
        if not 1 <= count <= 6:
            raise ValueError("Select between 1 and 6 images")
        if height < self.patch_size or width < self.patch_size:
            raise ValueError("Images must contain at least one full patch")
        # NumPy cannot convert bfloat16 tensors directly.
        cpu_images = images.detach().float().cpu()
        cols = min(3, count)
        rows = math.ceil(count / cols)
        grid = np.full((rows * height, cols * width, 3), 26, dtype=np.uint8)
        for index, image in enumerate(cpu_images):
            row, col = divmod(index, cols)
            pixels = image.permute(1, 2, 0).clamp(0, 1).numpy()
            grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = (
                pixels * 255
            ).astype(np.uint8)
        self.current_images = cpu_images
        self.grid_config = {
            "num_images": count,
            "cols": cols,
            "rows": rows,
            "image_height": height,
            "image_width": width,
            "patch_size": self.patch_size,
        }
        self.grid_image_pil = Image.fromarray(grid)

    def get_coordinate_info(self, x: float, y: float) -> Optional[Dict]:
        """Map canvas coordinates to a model patch, including centered margins."""
        if self.grid_config is None:
            return None
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        if self.display_scale_x <= 0 or self.display_scale_y <= 0:
            return None
        original_x = (x - self.display_offset_x) / self.display_scale_x
        original_y = (y - self.display_offset_y) / self.display_scale_y
        config = self.grid_config
        if not (
            0 <= original_x < config["cols"] * config["image_width"]
            and 0 <= original_y < config["rows"] * config["image_height"]
        ):
            return None
        col = int(original_x // config["image_width"])
        row = int(original_y // config["image_height"])
        image_index = row * config["cols"] + col
        if image_index >= config["num_images"]:
            return None
        local_x = original_x - col * config["image_width"]
        local_y = original_y - row * config["image_height"]
        patch_x = int(local_x // config["patch_size"])
        patch_y = int(local_y // config["patch_size"])
        patch_h = config["image_height"] // config["patch_size"]
        patch_w = config["image_width"] // config["patch_size"]
        if patch_x >= patch_w or patch_y >= patch_h:
            return None
        return {
            "global_coords": (x, y),
            "original_coords": (original_x, original_y),
            "image_idx": image_index,
            "local_coords": (local_x, local_y),
            "patch_coords": (patch_x, patch_y),
            "patch_grid_size": (patch_h, patch_w),
            "valid": True,
        }

    def extract_patch_attention(self, img_idx, patch_y, patch_x, layer_name=None):
        if self.model is None:
            raise RuntimeError("Load images before extracting attention")
        return self.model.extract_patch_attention(
            img_idx, patch_y, patch_x, layer_name or self.layer
        )

    def visualize_patch_attention_dual(self, patch_attention_data: Dict) -> Optional[Image.Image]:
        """Render the selected presentation; preserve the old callable name."""
        if patch_attention_data is None:
            return None
        if self.current_images is None:
            raise RuntimeError("Load images before rendering attention")
        from vggt.visualization import render_attention_view

        return render_attention_view(
            self.current_images,
            patch_attention_data,
            patch_size=self.patch_size,
            dpi=200 if self._get_high_quality_mode() else 100,
            style=self.style,
        )

    def visualize_patch_attention(self, patch_attention_data: Dict, attention_type="row"):
        """Compatibility alias for the selected presentation style."""
        return self.visualize_patch_attention_dual(patch_attention_data)

    def _set_busy(self, busy, message=None):
        self._busy = busy
        self.load_button.configure(state="disabled" if busy else "normal")
        if self.style_selector is not None:
            self.style_selector.configure(state="disabled" if busy else "readonly")
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()
        if message:
            self.info_label.configure(text=message)

    def _start_worker(self, target, args=()):
        self._worker = threading.Thread(target=self._worker_task, args=(target, args), daemon=True)
        self._worker.start()

    def _worker_task(self, target, args):
        try:
            target(*args)
        except Exception as exc:
            self._events.put(("error", str(exc)))
        finally:
            if self._closing.is_set():
                self._release_cache()

    def load_images_thread(self, file_paths):
        success = self.load_images(file_paths)
        self._events.put(("loaded", len(file_paths)) if success else ("error", self.last_error))

    def show_attention_for_click(self, coord_info: Dict):
        """Extract and render in a worker; communicate only through a Queue."""
        patch_x, patch_y = coord_info["patch_coords"]
        data = self.extract_patch_attention(coord_info["image_idx"], patch_y, patch_x)
        if data is None:
            raise RuntimeError("The selected patch has no cached attention; load the images again")
        result = self.visualize_patch_attention_dual(data)
        self._events.put(("attention", (result, coord_info, data)))

    def _drain_events(self):
        """Only the main thread calls this function or schedules Tk callbacks."""
        if self._closing.is_set():
            return
        try:
            while True:
                event, payload = self._events.get_nowait()
                if event == "status":
                    self.info_label.configure(text=payload)
                elif event == "loaded":
                    self._selected_patch = None
                    self.update_canvas_display()
                    self._set_busy(False, f"{payload} view(s) ready · Click a patch to compare attention")
                elif event == "attention":
                    result, coordinate, data = payload
                    self.show_result_window(result, coordinate, data, "dual")
                    self._set_busy(False, "Attention ready · Save the full-resolution PNG from the result window")
                elif event == "error":
                    if self.grid_image_pil is None:
                        self.canvas.delete("all")
                    self._set_busy(False, f"Could not complete the operation: {payload}")
                    self.messagebox.showerror("Attention visualization", payload, parent=self.root)
        except queue.Empty:
            pass
        except Exception as exc:
            self._set_busy(False, f"Could not display the result: {exc}")
            self.messagebox.showerror("Attention visualization", str(exc), parent=self.root)
        self.root.after(60, self._drain_events)

    def on_canvas_click(self, event):
        if self._busy:
            return
        coordinate = self.get_coordinate_info(event.x, event.y)
        if coordinate is None:
            return
        self._selected_patch = coordinate
        self._draw_selection()
        patch_x, patch_y = coordinate["patch_coords"]
        self._set_busy(
            True,
            f"Rendering {self.layer} · View {coordinate['image_idx']} · Patch (row {patch_y}, col {patch_x})…",
        )
        self._start_worker(self.show_attention_for_click, (coordinate,))

    def on_canvas_motion(self, event):
        if self._busy:
            return
        coordinate = self.get_coordinate_info(event.x, event.y)
        if coordinate:
            patch_x, patch_y = coordinate["patch_coords"]
            self.info_label.configure(
                text=f"View {coordinate['image_idx']} · Patch (row {patch_y}, col {patch_x}) · Click to inspect"
            )

    def _draw_selection(self):
        self.canvas.delete("selection")
        if self._selected_patch is None or self.grid_config is None:
            return
        row, col = divmod(self._selected_patch["image_idx"], self.grid_config["cols"])
        patch_x, patch_y = self._selected_patch["patch_coords"]
        x = col * self.grid_config["image_width"] + patch_x * self.patch_size
        y = row * self.grid_config["image_height"] + patch_y * self.patch_size
        self.canvas.create_rectangle(
            self.display_offset_x + x * self.display_scale_x,
            self.display_offset_y + y * self.display_scale_y,
            self.display_offset_x + (x + self.patch_size) * self.display_scale_x,
            self.display_offset_y + (y + self.patch_size) * self.display_scale_y,
            outline="#38bdf8", width=2, tags="selection",
        )

    def update_canvas_display(self):
        if self.canvas is None or self.grid_image_pil is None:
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if min(width, height) <= 1:
            return
        scale = min(width / self.grid_image_pil.width, height / self.grid_image_pil.height)
        new_width = max(1, round(self.grid_image_pil.width * scale))
        new_height = max(1, round(self.grid_image_pil.height * scale))
        self.display_scale_x = new_width / self.grid_image_pil.width
        self.display_scale_y = new_height / self.grid_image_pil.height
        self.display_offset_x = (width - new_width) // 2
        self.display_offset_y = (height - new_height) // 2
        preview = self.grid_image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.grid_image_tk = self.ImageTk.PhotoImage(preview, master=self.root)
        self.canvas.delete("all")
        self.canvas.create_image(
            self.display_offset_x, self.display_offset_y,
            image=self.grid_image_tk, anchor="nw",
        )
        config = self.grid_config
        for index in range(config["num_images"]):
            row, col = divmod(index, config["cols"])
            x = self.display_offset_x + col * config["image_width"] * self.display_scale_x + 10
            y = self.display_offset_y + row * config["image_height"] * self.display_scale_y + 10
            tag = f"view-label-{index}"
            self.canvas.create_text(x, y, text=f"View {index}", anchor="nw", fill="white", font=("Arial", 11, "bold"), tags=tag)
            bounds = self.canvas.bbox(tag)
            background = self.canvas.create_rectangle(bounds[0] - 4, bounds[1] - 2, bounds[2] + 4, bounds[3] + 2, fill="#172033", outline="")
            self.canvas.tag_lower(background, tag)
        self._draw_selection()

    def load_images_dialog(self):
        paths = self.filedialog.askopenfilenames(
            parent=self.root, title="Select 1–6 views of a scene",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("All files", "*")],
        )
        if not paths:
            return
        if len(paths) > 6:
            self.messagebox.showwarning("Select up to 6 views", "Choose between 1 and 6 images for one attention run.", parent=self.root)
            return
        self._set_busy(True, "Preparing images…")
        self._start_worker(self.load_images_thread, (list(paths),))

    def show_result_window(self, result_image, coord_info, patch_attention_data, attention_type="dual"):
        if result_image is None:
            raise ValueError("The renderer did not return an image")
        if self.result_window is not None and self.result_window.winfo_exists():
            self.result_window.destroy()
        window = self.tk.Toplevel(self.root)
        self.result_window = window
        window.title(f"VGGT · {self.style.title()} attention")
        window.geometry("1200x850")
        frame = self.ttk.Frame(window, padding=18)
        frame.pack(fill="both", expand=True)
        patch_x, patch_y = coord_info["patch_coords"]
        self.ttk.Label(frame, text=f"View {coord_info['image_idx']} · Patch (row {patch_y}, col {patch_x}) · {patch_attention_data['layer_name']}", font=("Arial", 13, "bold")).pack(anchor="w")
        description = {
            "spotlight": "Selected patch → stronger attention regions. Each target uses its own percentile cutoff.",
            "links": "Selected patch → highest-weight separated targets. Lines show attention, not geometric matches.",
            "raw": "Outgoing: what this patch attends to. Incoming: which patches attend to it. Shared raw scale.",
        }[self.style]
        self.ttk.Label(frame, text=description).pack(anchor="w", pady=(6, 12))
        preview_label = self.ttk.Label(frame)
        preview_label.pack(fill="both", expand=True)
        # Keep the renderer's full-resolution image intact for export.
        def resize_preview(event):
            if event.width <= 1 or event.height <= 1:
                return
            preview = result_image.copy()
            preview.thumbnail((event.width, event.height), Image.Resampling.LANCZOS)
            photo = self.ImageTk.PhotoImage(preview, master=window)
            preview_label.configure(image=photo)
            preview_label.image = photo

        preview_label.bind("<Configure>", resize_preview)

        def save_result():
            filename = f"attention_{self.style}_view{coord_info['image_idx']}_patch{patch_y}_{patch_x}_{self.layer}.png"
            path = self.filedialog.asksaveasfilename(
                parent=window, title="Export full-resolution attention", initialdir=str(self._export_directory),
                initialfile=filename, defaultextension=".png", filetypes=[("PNG image", "*.png")],
            )
            if not path:
                return
            try:
                result_image.save(path, format="PNG")
                self._export_directory = Path(path).parent
                self.info_label.configure(text=f"Saved full-resolution attention to {path}")
            except (OSError, ValueError) as exc:
                self.messagebox.showerror("Export failed", str(exc), parent=window)

        self.ttk.Button(frame, text="Save full-resolution PNG…", command=save_result).pack(anchor="e", pady=(12, 0))

    def _release_cache(self):
        if self.model is not None:
            self.model.disable_attention_caching()
            self.model.clear_attention_cache()

    def _close(self):
        self._closing.set()
        self.root.destroy()

    def create_gui(self):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
        from PIL import ImageTk

        self.tk, self.ttk, self.filedialog, self.messagebox, self.ImageTk = tk, ttk, filedialog, messagebox, ImageTk
        self.root = tk.Tk()
        self.root.title("VGGT Attention Explorer")
        self.root.geometry("1280x880")
        self.root.minsize(760, 560)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.high_quality_mode = tk.BooleanVar(master=self.root, value=self._high_quality)
        self.style_choice = tk.StringVar(master=self.root, value=self.style)
        main = ttk.Frame(self.root, padding=20)
        main.pack(fill="both", expand=True)
        toolbar = ttk.Frame(main)
        toolbar.pack(fill="x")
        heading = ttk.Frame(toolbar)
        heading.pack(side="left")
        ttk.Label(heading, text="VGGT Attention Explorer", font=("Arial", 20, "bold")).pack(anchor="w")
        heads = "all heads" if self.max_heads is None else f"first {self.max_heads} heads"
        ttk.Label(heading, text=f"{self.model_id} · {self.layer} · {heads} · {self.requested_device}").pack(anchor="w", pady=(4, 0))
        controls = ttk.Frame(toolbar)
        controls.pack(side="right")
        self.load_button = ttk.Button(controls, text="Open 1–6 images…", command=self.load_images_dialog)
        self.load_button.pack(anchor="e")
        self.style_selector = ttk.Combobox(controls, textvariable=self.style_choice,
                                          values=("spotlight", "links", "raw"), state="readonly", width=18)
        self.style_selector.pack(anchor="e", pady=(6, 0))
        self.style_selector.bind("<<ComboboxSelected>>", self.on_style_changed)
        ttk.Checkbutton(controls, text="High-resolution export", variable=self.high_quality_mode, command=self.on_quality_mode_changed).pack(anchor="e", pady=(6, 0))
        ttk.Separator(main).pack(fill="x", pady=14)
        self.info_label = ttk.Label(main, text="Open scene views, then click an image patch to compare attention.", wraplength=1100)
        self.info_label.pack(fill="x", pady=(0, 8))
        self.progress = ttk.Progressbar(main, mode="indeterminate")
        self.progress.pack(fill="x", pady=(0, 12))
        self.canvas = tk.Canvas(main, background="#1a1f2b", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Configure>", lambda event: self.update_canvas_display())
        ttk.Label(main, text="Click a patch; switch between spotlight, sparse targets and raw analysis.\nCoordinates use the preprocessed model input. Clicks and style changes reuse the cached attention.", foreground="#687386", wraplength=1100).pack(anchor="w", pady=(12, 0))
        self.root.after(60, self._drain_events)

    def run(self):
        try:
            self.create_gui()
            self.root.mainloop()
        finally:
            self._closing.set()
            if self._worker is None or not self._worker.is_alive():
                self._release_cache()


def build_parser():
    parser = argparse.ArgumentParser(description="Explore VGGT patch attention in a desktop window. Model weights are loaded only after selecting 1–6 images.")
    parser.add_argument("--model", default="facebook/VGGT-1B", help="Hugging Face model ID or local pretrained model directory")
    parser.add_argument("--device", default="auto", help="auto (CUDA when available), cpu, cuda, or cuda:N")
    parser.add_argument("--layer", default="global_block_11", help="global_block_0 through global_block_23 (default: %(default)s)")
    parser.add_argument("--style", choices=("spotlight", "links", "raw"), default="spotlight")
    parser.add_argument("--max-heads", type=int, default=None, help="cache only the first K heads (default: all heads)")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        visualizer = TkinterAttentionVisualizer(model=args.model, device=args.device, layer=args.layer,
                                               max_heads=args.max_heads, style=args.style)
        visualizer.run()
    except (ImportError, ValueError) as exc:
        parser.exit(1, f"Unable to start the attention explorer: {exc}\n")
    except Exception as exc:
        print(f"Unable to start the attention explorer: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
