# VGGT Attention Explorer

[English](README.md) · [使用指南](docs/usage.md) · [验证记录](docs/validation.md) · [安装与排障](docs/package.md)

点击一个图像 patch，查看它在其他视角中**关注哪里**。支持重点区域、候选点连线和原始双向热图三种展示。本项目是在上游 [VGGT](https://github.com/facebookresearch/vggt) 基础上开发的独立研究可视化工具。

![选择花心，查看另两个视角中的最高注意力位置](docs/assets/showcase/flower-center-links.png)

**花心跨视角展示。** 橙色框是选中的 patch，青色框是每个目标视角中 attention 最高的位置。左下角放大选中区域。这是官方 VGGT-1B 完整权重的真实 CPU 推理，使用 `global_block_11`（索引从 0 开始），平均全部 16 个 heads。连线表示注意力指向，不代表经过独立验证的几何匹配。

![切换 LEGO 驾驶舱和铲斗，观察关注区域变化](docs/assets/showcase/lego-query-change.gif)

**切换查询位置。** 动图在驾驶舱和铲斗两个 patch 之间切换，复用同一次模型推理。每个目标视角中超过第 90 百分位的 patch 保留原色，其他区域变暗；attention 原始数值保持不变。

[更多样本与展示说明](docs/showcase.md) · [原始数值与检查记录](docs/assets/showcase/validation.json) · [图片来源](examples/README.md)

## 安装

需要 Python 3.10+。在虚拟环境中先按照 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装适合硬件的 torch / torchvision 组合，再安装本项目：

```bash
git clone https://github.com/Glupayy/VGGT-Image-Seq-Attention.git
cd VGGT-Image-Seq-Attention
python -m venv .venv
source .venv/bin/activate
# 先在此环境安装匹配的 torch 和 torchvision
python -m pip install -e '.[dev]'
```

## 使用

无需下载权重的端到端演示：

```bash
vggt-attention --demo --output outputs/demo.png
```

它使用固定随机种子的微型 VGGT aggregator 和合成图像，只用于验证程序流程，**不是预训练 VGGT-1B 的实验结果**。

桌面交互：

```bash
vggt-attention-gui
```

选择 1–6 张相关图像，运行一次模型，再点击 patch 探索。默认缓存 `global_block_11`，平均全部 attention heads。下拉框可以切换 `spotlight`、`links`、`raw`，点击和切换样式都会复用缓存。模型首次推理会从官方 `facebook/VGGT-1B` 下载；有 CUDA 时自动使用 CUDA，否则使用 CPU。桌面入口需要 Tkinter 和图形会话。

命令行导出：

```bash
vggt-attention examples/images/flower/000.png examples/images/flower/006.png \
  examples/images/flower/012.png --source 0 --patch 14 25 --layer 11 \
  --style links --top-k 1 --output outputs/flower.png
```

`--source` 与 `--patch 行 列` 均从 0 开始，坐标指向**预处理后**的 patch 网格。省略 `--patch` 默认选择中心。可用 `--model ./VGGT-1B` 载入本地目录，`--max-heads K` 只平均前 K 个 heads。

输出包含配套的 `.png` 图、`.npz` 原始出向/入向数值与预处理 RGB 图像、`.json` 模型/输入/层/坐标/heads/缓存与展示变换信息。三种样式导出的 NPZ 原始数值一致。

## 如何理解图

对归一化、2D RoPE 后的 Q/K，取 `A = softmax(QKᵀ / √d)`，再平均各 head。

| 方向 | 读取方式 | 含义 |
| --- | --- | --- |
| 出向 | `A[i, :]` | 选中 patch 对其他 token 的关注 |
| 入向 | `A[:, i]` | 其他 token 对选中 patch 的关注 |

完整行的和为 1；列不要求和为 1。图上仅显示图像 patch，camera/register tokens 仍参与 softmax，因此可见的出向权重之和可能小于 1。程序不会二次归一化入向权重。

| 样式 | 展示方式 |
| --- | --- |
| `spotlight`（默认） | 每视角按 attention 排名，保留超过第 90 百分位区域的原色；`--top-percent` 可调阈值 |
| `links` | 用少量标记展示原始 attention 的高值位置，连到每视角的第一名；`--top-k` 调整候选点数 |
| `raw` | 原图、出向和入向热图，所有热图共享线性色标 |

Attention 可用于观察该层的信息路由，不单独构成因果解释或几何对应的证明。首页是人工选择的易读示例，不是匹配准确率评测；其他位置也可能关注边缘或不相关区域。CLI/GUI 默认使用第 11 层，底层缓存 API 保留第 23 层的兼容默认值。

缓存每个模型独立，每次推理更新真实 batch、帧数、网格和特殊 token 偏移。计算按 query 分块，CPU 保存 head 均值；持久缓存仍为每层 `4 × B × N²` 字节，大量图像会增加内存。

## 验证与历史资料

```bash
python -m pytest -q
python -m build
```

详见 [验证记录](docs/validation.md)。当前维护 `vggt/` 下的桌面与 CLI 入口；历史材料统一收在 [archive](archive/README.md)，其中研究脚本的模拟 attention 不作为真实模型证据。

| 目录 | 用途 |
| --- | --- |
| [`vggt/`](vggt/) | 模型、attention 缓存、渲染、CLI 与桌面入口 |
| [`examples/`](examples/README.md) | 多视图输入与逐文件来源记录 |
| [`scripts/`](scripts/) | 样例生成、数值验证与抽帧工具 |
| [`tests/`](tests/) | 无需下载权重的数值、界面与导出检查 |
| [`docs/`](docs/usage.md) | 使用、安装、展示说明与验证记录 |
| [`archive/`](archive/README.md) | 历史演示、notebook、结果图与研究记录 |

保留上游贡献者署名与 [CC BY-NC 4.0 许可证](LICENSE.txt)。权重遵循官方仓库条款，不提交到本仓库。
