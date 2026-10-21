# Multi-view sample images

These three scenes provide recognizable objects for exploring attention: a yellow LEGO bulldozer, red flowers, and a large indoor plant. Each group contains three views of the same scene.

| Local group | Included files | Resolution | Upstream example |
| --- | --- | --- | --- |
| `lego/` | `00.png`, `03.png`, `06.png` | 779 × 520 | `examples/kitchen/images/` |
| `flower/` | `000.png`, `006.png`, `012.png` | 706 × 529 | `examples/llff_flower/images/` |
| `fern/` | `000.png`, `005.png`, `010.png` | 706 × 529 | `examples/llff_fern/images/` |

All nine files retain their original upstream bytes, filenames, and dimensions. Only the containing scene directories have shorter local names. Resizing and cropping happen when the model loads the images; the PNG files here remain unchanged.

The source is [facebookresearch/vggt](https://github.com/facebookresearch/vggt), pinned to revision [`b529f922f3b05fada4374ab95cd4f2b1d06f6469`](https://github.com/facebookresearch/vggt/tree/b529f922f3b05fada4374ab95cd4f2b1d06f6469/examples). [SOURCES.json](SOURCES.json) records each image's upstream path and URL, Git blob SHA-1, SHA-256, byte size, and dimensions. It also records an additional upstream revision where these image blobs were verified unchanged.

At the pinned revision, the upstream repository includes a [CC BY-NC 4.0 license](https://github.com/facebookresearch/vggt/blob/b529f922f3b05fada4374ab95cd4f2b1d06f6469/LICENSE.txt), and its README describes the code as available under that license. The example image directories contain no separate image license notice. This provenance record does not establish additional permissions for these images; retain upstream attribution and applicable source terms when reusing them.

Use a single group per run so all inputs depict one scene. For example, from the repository root:

```bash
vggt-attention sample_images/lego/00.png sample_images/lego/03.png sample_images/lego/06.png --style spotlight --output outputs/lego.png
```

See the project [README](../README.md) for model setup and the meanings of the display styles.
