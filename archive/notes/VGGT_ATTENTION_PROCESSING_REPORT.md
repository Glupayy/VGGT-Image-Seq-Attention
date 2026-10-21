# VGGT Interactive Attention Visualizer 技术报告

> 历史实现说明：本报告记录早期可视化方案，包含当时固定的图像尺寸、token 数量、缓存方式和界面行为。发布版的实现与这些描述已有变化；当前使用方法请以 [README](../../README.md) 和 [可视化指南](../../docs/usage.md) 为准，正确性以当前 [tests](../../tests/) 中的断言测试及 [验证记录](../../docs/validation.md) 为准。

## 概述

VGGT Interactive Attention Visualizer 是一个基于深度学习的多视图几何理解模型的交互式注意力可视化工具。该系统通过分析VGGT-1B模型的attention maps，实现对图像中任意patch的多视图attention分布的实时可视化。

## 系统架构

```ascii
┌─────────────────────────────────────────────────────────────────┐
│                   VGGT Attention Visualizer                    │
├─────────────────────────────────────────────────────────────────┤
│  GUI Layer (Tkinter)                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │   Canvas    │  │  Controls   │  │   Results   │            │
│  │  Display    │  │   Panel     │  │   Window    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  Processing Layer                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Coordinate  │  │  Attention  │  │    Plot     │            │
│  │  Mapping    │  │ Extraction  │  │ Generation  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│  Model Layer                                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │    VGGT     │  │  Attention  │  │    Hook     │            │
│  │    Model    │  │    Maps     │  │   System    │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

## 详细工作流程

### Phase 1: 系统初始化与模型加载

```ascii
系统启动
    │
    ├── 设备检测 (CUDA/CPU)
    │   ┌─────────────────────────────────┐
    │   │ if torch.cuda.is_available():   │
    │   │   device = "cuda"               │
    │   │   dtype = bfloat16/float16      │
    │   │ else:                           │
    │   │   device = "cpu"                │
    │   │   dtype = float32               │
    │   └─────────────────────────────────┘
    │
    ├── GUI界面创建 (Tkinter)
    │   ┌─────────────────────────────────┐
    │   │ ┌─────────┐ ┌─────────┐        │
    │   │ │ 控制面板 │ │ 状态栏  │        │
    │   │ └─────────┘ └─────────┘        │
    │   │ ┌─────────────────────────────┐ │
    │   │ │         Canvas              │ │
    │   │ │      (图像显示区域)          │ │
    │   │ └─────────────────────────────┘ │
    │   │ ┌─────────────────────────────┐ │
    │   │ │        Help Text            │ │
    │   │ └─────────────────────────────┘ │
    │   └─────────────────────────────────┘
    │
    └── 延迟模型加载策略
        (用户触发时才加载VGGT-1B模型)
```

**可调参数:**
- `device`: 计算设备选择 ("cuda"/"cpu")
- `dtype`: 数据类型 (bfloat16/float16/float32)
- GUI窗口大小: 默认1000x700像素

### Phase 2: 图像加载与预处理

```ascii
用户选择图像文件 (2-6张)
    │
    ├── 图像验证
    │   ┌─────────────────────────────────┐
    │   │ if len(files) < 2:              │
    │   │   return "需要至少2张图像"       │
    │   │ if len(files) > 6:              │
    │   │   files = files[:6]             │
    │   └─────────────────────────────────┘
    │
    ├── VGGT模型加载 (延迟加载)
    │   ┌─────────────────────────────────┐
    │   │ VGGT.from_pretrained("VGGT-1B") │
    │   │        ↓                        │
    │   │   model.to(device)              │
    │   │   model.eval()                  │
    │   └─────────────────────────────────┘
    │
    ├── 图像预处理
    │   ┌─────────────────────────────────┐
    │   │ load_and_preprocess_images()    │
    │   │   ├── 读取图像文件               │
    │   │   ├── 尺寸标准化                │
    │   │   ├── 像素值归一化 [0,1]        │
    │   │   └── 转换为Tensor              │
    │   └─────────────────────────────────┘
    │
    └── 创建图像网格
        ┌─────────────────────────────────┐
        │  S张图像 → cols=min(3,S)        │
        │           → rows=(S+cols-1)//cols│
        │                                 │
        │  Grid Layout:                   │
        │  ┌─────┐ ┌─────┐ ┌─────┐        │
        │  │IMG 0│ │IMG 1│ │IMG 2│        │
        │  └─────┘ └─────┘ └─────┘        │
        │  ┌─────┐ ┌─────┐ ┌─────┐        │
        │  │IMG 3│ │IMG 4│ │IMG 5│        │
        │  └─────┘ └─────┘ └─────┘        │
        └─────────────────────────────────┘
```

**可调参数:**
- 最大图像数量: 6张 (可修改)
- 网格列数: 最多3列 (可修改)
- 图像预处理参数: 在`load_and_preprocess_images()`中配置

### Phase 3: Attention Hook系统设置

```ascii
设置Attention Hooks
    │
    ├── 目标层选择
    │   ┌─────────────────────────────────┐
    │   │ target_layers=[0,5,11,17,23]    │
    │   │   ├── frame_blocks (局部特征)   │
    │   │   └── global_blocks (全局特征)  │
    │   └─────────────────────────────────┘
    │
    ├── Hook函数创建
    │   ┌─────────────────────────────────┐
    │   │ def create_hook(name):          │
    │   │   def hook(module, input, out): │
    │   │     # 提取QKV矩阵              │
    │   │     qkv = module.attn.qkv(x)    │
    │   │     # 计算attention权重        │
    │   │     attn = (q @ k.T) * scale    │
    │   │     attn = softmax(attn)        │
    │   │     # 存储attention maps        │
    │   │     store_attention(name, attn) │
    │   └─────────────────────────────────┘
    │
    └── Hook注册
        ┌─────────────────────────────────┐
        │ for layer_idx in target_layers: │
        │   if frame_blocks[i] exists:    │
        │     register_hook(frame_block_i)│
        │   if global_blocks[i] exists:   │
        │     register_hook(global_block_i│
        └─────────────────────────────────┘
```

**可调参数:**
- `target_layers`: 捕获attention的层索引 [0,5,11,17,23]
- `num_heads_to_keep`: 保留的attention head数量 (默认8)
- Hook作用域: frame_blocks + global_blocks

### Phase 4: 模型推理与Attention捕获

```ascii
模型推理过程
    │
    ├── 前向传播
    │   ┌─────────────────────────────────┐
    │   │ with torch.no_grad():           │
    │   │   if CUDA available:            │
    │   │     with autocast():            │
    │   │       output = model(images)    │
    │   │   else:                         │
    │   │       output = model(images)    │
    │   └─────────────────────────────────┘
    │
    ├── Attention Maps捕获
    │   ┌─────────────────────────────────┐
    │   │ Attention Flow:                 │
    │   │                                 │
    │   │ Input Images (B,3,H,W)          │
    │   │        ↓                        │
    │   │ Patch Embedding                 │
    │   │        ↓                        │
    │   │ ┌─────────────────────────────┐ │
    │   │ │    Frame Blocks             │ │
    │   │ │  ┌───┐ ┌───┐ ┌───┐ ┌───┐   │ │
    │   │ │  │ 0 │→│ 5 │→│11 │→│17 │   │ │ 
    │   │ │  └───┘ └───┘ └───┘ └───┘   │ │
    │   │ │     ↓Hook ↓Hook ↓Hook ↓Hook│ │
    │   │ └─────────────────────────────┘ │
    │   │        ↓                        │
    │   │ ┌─────────────────────────────┐ │
    │   │ │    Global Blocks            │ │
    │   │ │  ┌───┐ ┌───┐ ┌───┐ ┌───┐   │ │
    │   │ │  │ 0 │→│ 5 │→│11 │→│17 │→│23││ │
    │   │ │  └───┘ └───┘ └───┘ └───┘ └──┘│ │
    │   │ │     ↓Hook ↓Hook ↓Hook ↓Hook ↓│ │
    │   │ └─────────────────────────────┘ │
    │   └─────────────────────────────────┘
    │
    └── Attention数据结构
        ┌─────────────────────────────────┐
        │ attention_maps = {              │
        │   'frame_block_0': {            │
        │     'attention': tensor(B,H,N,N)│
        │     'tokens_shape': (B,N,C)     │
        │     'num_heads': 8              │
        │   },                            │
        │   'global_block_23': {          │
        │     'attention': tensor(B,H,N,N)│
        │     'tokens_shape': (B,N,C)     │
        │     'num_heads': 8              │
        │   },                            │
        │   ...                           │
        │ }                               │
        └─────────────────────────────────┘
```

**可调参数:**
- 推理精度: autocast启用/禁用
- 内存管理: `torch.cuda.empty_cache()`频率
- Batch size限制: 最多6张图像

### Phase 5: 交互式坐标映射

```ascii
鼠标交互系统
    │
    ├── 坐标捕获
    │   ┌─────────────────────────────────┐
    │   │ Canvas Event: <Motion>, <Click> │
    │   │   ├── event.x, event.y          │
    │   │   └── 实时坐标显示               │
    │   └─────────────────────────────────┘
    │
    ├── 坐标变换链
    │   ┌─────────────────────────────────┐
    │   │ GUI坐标 → 图像网格坐标           │
    │   │    (x,y) → (grid_x, grid_y)     │
    │   │                                 │
    │   │ 网格坐标 → 图像索引              │
    │   │ col = x // image_width          │
    │   │ row = y // image_height         │
    │   │ img_idx = row * cols + col      │
    │   │                                 │
    │   │ 图像坐标 → Patch坐标             │
    │   │ local_x = x % image_width       │
    │   │ local_y = y % image_height      │
    │   │ patch_x = local_x // patch_size │
    │   │ patch_y = local_y // patch_size │
    │   └─────────────────────────────────┘
    │
    └── 坐标信息结构
        ┌─────────────────────────────────┐
        │ coordinate_info = {             │
        │   'global_coords': (x, y),      │
        │   'image_idx': img_idx,         │
        │   'local_coords': (lx, ly),     │
        │   'patch_coords': (px, py),     │
        │   'patch_grid_size': (ph, pw),  │
        │   'valid': True                 │
        │ }                               │
        └─────────────────────────────────┘
```

**可调参数:**
- `patch_size`: Patch尺寸 (默认14x14像素)
- 网格配置: `cols`, `rows`可动态调整
- 坐标验证范围: 可设置有效区域边界

### Phase 6: Patch Attention提取

```ascii
单Patch Attention提取流程
    │
    ├── 参数验证
    │   ┌─────────────────────────────────┐
    │   │ extract_patch_attention(        │
    │   │   img_idx,    # 源图像索引      │
    │   │   patch_y,    # Patch Y坐标     │
    │   │   patch_x,    # Patch X坐标     │
    │   │   layer_name  # 目标层名称      │
    │   │ )                               │
    │   └─────────────────────────────────┘
    │
    ├── Attention Map解析
    │   ┌─────────────────────────────────┐
    │   │ # 从captured attention maps中    │
    │   │ # 提取指定层的attention矩阵      │
    │   │                                 │
    │   │ attn_data = attention_maps[     │
    │   │   layer_name]['attention']      │
    │   │                                 │
    │   │ # Shape: (B, num_heads, N, N)   │
    │   │ # B=1, N=patch_count            │
    │   │                                 │
    │   │ Patch Grid Layout:              │
    │   │ ┌───┬───┬───┬───┬───┬───┬───┐   │
    │   │ │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │...│   │
    │   │ ├───┼───┼───┼───┼───┼───┼───┤   │
    │   │ │16 │17 │18 │19 │20 │21 │...│   │
    │   │ ├───┼───┼───┼───┼───┼───┼───┤   │
    │   │ │32 │33 │34 │35 │36 │37 │...│   │
    │   │ └───┴───┴───┴───┴───┴───┴───┘   │
    │   │                                 │
    │   │ source_patch_idx = py*pw + px   │
    │   └─────────────────────────────────┘
    │
    ├── 多头Attention聚合
    │   ┌─────────────────────────────────┐
    │   │ # 对多个attention heads求平均    │
    │   │ attn_aggregated = attn_data.    │
    │   │   mean(dim=1)  # 头维度平均     │
    │   │                                 │
    │   │ # Shape: (B, N, N) → (1, N, N)  │
    │   │                                 │
    │   │ Source Patch Attention Vector:  │
    │   │ src_attn = attn_aggregated[0,   │
    │   │   source_patch_idx, :]          │
    │   │                                 │
    │   │ # Shape: (N,) - 对所有patch的   │
    │   │ # attention权重                 │
    │   └─────────────────────────────────┘
    │
    └── 多图像Attention Maps生成
        ┌─────────────────────────────────┐
        │ for each image in batch:        │
        │   # 重构为2D attention map      │
        │   attn_2d = src_attn.reshape(   │
        │     patch_height, patch_width)  │
        │                                 │
        │   # 上采样到原图尺寸            │
        │   attn_upsampled = F.           │
        │     interpolate(attn_2d,        │
        │     size=(H, W), mode='bilinear'│
        │   )                             │
        │                                 │
        │   attention_maps[i] = attn_     │
        │     upsampled                   │
        └─────────────────────────────────┘
```

**可调参数:**
- `layer_name`: 分析的attention层 (默认'global_block_23')
- Attention聚合方式: mean/max/weighted average
- 上采样方法: 'bilinear'/'nearest'/'bicubic'
- Patch网格尺寸: 根据图像尺寸和patch_size计算

### Phase 7: 可视化结果生成

```ascii
Attention可视化渲染
    │
    ├── 布局计算
    │   ┌─────────────────────────────────┐
    │   │ 图像数量 ≤ 3:                   │
    │   │ ┌─────┐ ┌─────┐ ┌─────┐         │
    │   │ │原图0│ │原图1│ │原图2│         │
    │   │ └─────┘ └─────┘ └─────┘         │
    │   │ ┌─────┐ ┌─────┐ ┌─────┐         │
    │   │ │热图0│ │热图1│ │热图2│         │
    │   │ └─────┘ └─────┘ └─────┘         │
    │   │                                 │
    │   │ 图像数量 > 3:                   │
    │   │ 第一部分 (图像0-2):             │
    │   │ ┌─────┐ ┌─────┐ ┌─────┐         │
    │   │ │原图0│ │原图1│ │原图2│         │
    │   │ └─────┘ └─────┘ └─────┘         │
    │   │ ┌─────┐ ┌─────┐ ┌─────┐         │
    │   │ │热图0│ │热图1│ │热图2│         │
    │   │ └─────┘ └─────┘ └─────┘         │
    │   │                                 │
    │   │ 第二部分 (图像3-5):             │
    │   │ ┌─────┐ ┌─────┐ ┌─────┐         │
    │   │ │原图3│ │原图4│ │原图5│         │
    │   │ └─────┘ └─────┘ └─────┘         │
    │   │ ┌─────┐ ┌─────┐ ┌─────┐         │
    │   │ │热图3│ │热图4│ │热图5│         │
    │   │ └─────┘ └─────┘ └─────┘         │
    │   │                                 │
    │   │ 最终: 垂直拼接两部分            │
    │   └─────────────────────────────────┘
    │
    ├── 渲染管道
    │   ┌─────────────────────────────────┐
    │   │ for each image:                 │
    │   │                                 │
    │   │ 1. 原图渲染                     │
    │   │    ├── 图像数据标准化 [0,1]     │
    │   │    ├── matplotlib imshow()      │
    │   │    └── 添加标题和坐标轴         │
    │   │                                 │
    │   │ 2. Source Patch标记             │
    │   │    if image == source:          │
    │   │      ├── 绘制红色矩形框         │
    │   │      ├── 框位置: (px*14, py*14) │
    │   │      └── 框尺寸: 14x14像素      │
    │   │                                 │
    │   │ 3. Attention热图叠加            │
    │   │    ├── 原图作为背景 (alpha=0.7) │
    │   │    ├── 热图叠加 (alpha=0.8)     │
    │   │    ├── 颜色映射: 'hot' colormap │
    │   │    └── 添加colorbar             │
    │   │                                 │
    │   │ 4. 图像拼接 (如需要)            │
    │   │    ├── 第一组图像 → PIL Image1  │
    │   │    ├── 第二组图像 → PIL Image2  │
    │   │    └── 垂直拼接 → Final Result  │
    │   └─────────────────────────────────┘
    │
    └── 结果输出
        ┌─────────────────────────────────┐
        │ 输出格式: PIL Image             │
        │ 分辨率: DPI=150                 │
        │ 颜色空间: RGB                   │
        │ 保存格式: PNG (无损)            │
        │                                 │
        │ 元数据包含:                     │
        │ ├── Source image索引            │
        │ ├── Source patch坐标            │
        │ ├── Attention layer名称         │
        │ ├── 最大attention值             │
        │ └── 处理时间戳                  │
        └─────────────────────────────────┘
```

**可调参数:**
- 图像分辨率: `dpi=150` (可调整)
- 颜色映射: `cmap='hot'` (可选择其他colormap)
- 透明度设置: `alpha=0.7/0.8` (背景/前景)
- 拼接策略: 垂直/水平拼接可选
- 输出格式: PNG/JPEG/SVG等

## 核心算法详解

### Attention权重计算

```ascii
Multi-Head Self-Attention 计算流程:

Input: X ∈ R^(B×N×C)
├── B = batch_size (图像数量)
├── N = sequence_length (patch数量)  
└── C = hidden_dimension (特征维度)

Step 1: 线性变换生成Q, K, V
┌─────────────────────────────────┐
│ Q = X @ W_q  # Query矩阵        │
│ K = X @ W_k  # Key矩阵          │  
│ V = X @ W_v  # Value矩阵        │
│                                 │
│ Shape: (B×N×C) → (B×N×d_head×h) │
│ 其中: d_head = C//num_heads     │
│       h = num_heads             │
└─────────────────────────────────┘

Step 2: 多头重构
┌─────────────────────────────────┐
│ Q = Q.reshape(B, N, h, d_head)  │
│     .permute(0, 2, 1, 3)        │
│ K = K.reshape(B, N, h, d_head)  │
│     .permute(0, 2, 1, 3)        │
│                                 │  
│ Result Shape: (B, h, N, d_head) │
└─────────────────────────────────┘

Step 3: Attention Score计算
┌─────────────────────────────────┐
│ scores = Q @ K.transpose(-2,-1) │
│ scores = scores / √d_head       │
│                                 │
│ Attention Matrix:               │
│      P0   P1   P2  ...  PN      │
│ P0 [ a00  a01  a02 ... a0N ]    │
│ P1 [ a10  a11  a12 ... a1N ]    │
│ P2 [ a20  a21  a22 ... a2N ]    │
│ ..                              │
│ PN [ aN0  aN1  aN2 ... aNN ]    │
│                                 │
│ 其中aij表示patch i对patch j的   │
│ attention权重                   │
└─────────────────────────────────┘

Step 4: Softmax归一化
┌─────────────────────────────────┐
│ attn_weights = softmax(scores,  │
│                       dim=-1)   │
│                                 │
│ 确保: Σ(aij) = 1, ∀i           │
│                                 │
│ 物理意义: patch i的注意力在     │
│ 所有patches上的概率分布         │
└─────────────────────────────────┘
```

### Patch坐标映射算法

```python
# 伪代码展示坐标变换过程
def pixel_to_patch_mapping(pixel_x, pixel_y, 
                          image_width, image_height, 
                          patch_size=14):
    """
    像素坐标到Patch坐标的映射算法
    
    输入坐标系统层次:
    GUI Canvas (鼠标坐标)
         ↓
    Image Grid (网格坐标) 
         ↓  
    Single Image (图像坐标)
         ↓
    Patch Grid (patch坐标)
    """
    
    # Level 1: GUI → Image Grid
    grid_cols = min(3, num_images)
    grid_rows = (num_images + grid_cols - 1) // grid_cols
    
    # Level 2: Grid → Image Index
    col = pixel_x // image_width
    row = pixel_y // image_height  
    image_idx = row * grid_cols + col
    
    # Level 3: Grid → Local Image
    local_x = pixel_x % image_width
    local_y = pixel_y % image_height
    
    # Level 4: Image → Patch
    patch_x = local_x // patch_size
    patch_y = local_y // patch_size
    
    # Validation
    patch_grid_w = image_width // patch_size
    patch_grid_h = image_height // patch_size
    
    valid = (0 <= patch_x < patch_grid_w and 
             0 <= patch_y < patch_grid_h and
             0 <= image_idx < num_images)
    
    return {
        'image_idx': image_idx,
        'patch_coords': (patch_x, patch_y),
        'local_coords': (local_x, local_y),
        'valid': valid
    }
```

## 性能优化策略

### 内存管理

```ascii
内存优化策略:
┌─────────────────────────────────────────┐
│ 1. 延迟加载 (Lazy Loading)              │
│    ├── 模型仅在需要时加载               │
│    ├── Attention maps按需计算           │
│    └── GUI组件分步初始化                │
│                                         │
│ 2. 内存回收 (Memory Cleanup)            │
│    ├── torch.cuda.empty_cache()         │
│    ├── 及时删除临时变量                 │
│    └── matplotlib figure关闭           │
│                                         │
│ 3. 数据类型优化 (Precision Control)     │
│    ├── CUDA: bfloat16/float16           │
│    ├── CPU: float32                     │
│    └── 根据GPU capability自动选择       │
│                                         │
│ 4. 批处理限制 (Batch Size Control)      │
│    ├── 最多同时处理6张图像              │
│    ├── 避免OOM错误                      │
│    └── 动态调整处理策略                 │
└─────────────────────────────────────────┘
```

### 计算优化

```ascii
计算优化技术:
┌─────────────────────────────────────────┐
│ 1. 自动混合精度 (Automatic Mixed        │
│    Precision - AMP)                     │
│    ┌─────────────────────────────────┐   │
│    │ with torch.amp.autocast('cuda', │   │
│    │   dtype=self.dtype):            │   │
│    │   output = model(images)        │   │
│    └─────────────────────────────────┘   │
│                                         │
│ 2. 推理优化 (Inference Optimization)    │
│    ├── torch.no_grad() 禁用梯度计算    │
│    ├── model.eval() 设置评估模式        │
│    └── 避免不必要的tensor复制           │
│                                         │
│ 3. 并行处理 (Parallel Processing)       │
│    ├── 多线程GUI更新                    │
│    ├── 异步attention分析                │
│    └── 后台数据预处理                   │
│                                         │
│ 4. 缓存策略 (Caching Strategy)          │
│    ├── Attention maps复用               │
│    ├── 预处理图像缓存                   │
│    └── 可视化结果缓存                   │
└─────────────────────────────────────────┘
```

## 可配置参数详解

### 模型相关参数

```python
class ModelConfig:
    """模型配置参数"""
    
    # 设备配置
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.bfloat16  # 可选: float16, float32
    
    # Attention捕获参数  
    target_layers: List[int] = [0, 5, 11, 17, 23]  # 目标层索引
    num_heads_to_keep: int = 8  # 保留的attention头数量
    attention_layer: str = 'global_block_23'  # 默认分析层
    
    # 图像处理参数
    max_images: int = 6  # 最大图像数量
    patch_size: int = 14  # Patch尺寸(像素)
    image_size: Tuple[int, int] = (224, 224)  # 标准图像尺寸
    
    # 内存管理
    enable_autocast: bool = True  # 自动混合精度
    clear_cache_frequency: int = 1  # 缓存清理频率
```

### 可视化参数

```python
class VisualizationConfig:
    """可视化配置参数"""
    
    # 布局参数
    max_cols: int = 3  # 网格最大列数
    figure_dpi: int = 150  # 图像分辨率
    figure_width_per_col: int = 4  # 每列宽度(英寸)
    figure_height: int = 8  # 图像高度(英寸)
    
    # 颜色和透明度
    background_alpha: float = 0.7  # 背景图像透明度
    heatmap_alpha: float = 0.8  # 热图透明度  
    colormap: str = 'hot'  # 热图颜色映射
    source_patch_color: str = 'red'  # 源patch标记颜色
    source_patch_linewidth: int = 3  # 标记线宽
    
    # 文本和标注
    title_fontsize: int = 16  # 标题字体大小
    subtitle_fontsize: int = 12  # 子标题字体大小
    colorbar_fraction: float = 0.046  # colorbar比例
    colorbar_pad: float = 0.04  # colorbar间距
    
    # 插值和上采样
    upsampling_mode: str = 'bilinear'  # 上采样方法
    upsampling_align_corners: bool = False  # 对齐角点
```

### GUI交互参数

```python
class GUIConfig:
    """GUI界面配置参数"""
    
    # 窗口设置
    window_title: str = "VGGT Interactive Attention Visualizer"
    window_size: Tuple[int, int] = (1000, 700)
    canvas_bg_color: str = "white"
    cursor_style: str = "crosshair"
    
    # 坐标显示
    coordinate_update_delay: int = 50  # 坐标更新延迟(ms)
    show_realtime_coords: bool = True  # 实时坐标显示
    coordinate_precision: int = 0  # 坐标显示精度
    
    # 结果窗口
    result_window_size: Tuple[int, int] = (800, 600)
    result_image_max_size: Tuple[int, int] = (700, 500)
    enable_scrollbar: bool = True  # 滚动条
    
    # 交互响应
    click_response_delay: int = 100  # 点击响应延迟(ms)
    thread_timeout: int = 30  # 线程超时时间(秒)
    enable_threading: bool = True  # 多线程处理
```

## 错误处理与调试

### 常见错误类型

```ascii
错误处理体系:
┌─────────────────────────────────────────┐
│ 1. 模型加载错误                         │
│    ├── CUDA不可用 → 自动切换CPU         │
│    ├── 内存不足 → 降低精度/减少图像     │
│    └── 模型文件损坏 → 重新下载提示       │
│                                         │
│ 2. 图像处理错误                         │
│    ├── 格式不支持 → 格式转换             │
│    ├── 尺寸异常 → 自动resize             │
│    └── 数量限制 → 截取前N张              │
│                                         │
│ 3. 坐标映射错误                         │
│    ├── 越界检查 → 边界限制               │
│    ├── 无效区域 → 用户提示               │
│    └── 精度问题 → 坐标舍入               │
│                                         │
│ 4. Attention计算错误                    │
│    ├── 维度不匹配 → 自动reshape          │
│    ├── 数值异常 → NaN/Inf检查           │
│    └── 内存溢出 → 分批处理               │
│                                         │
│ 5. GUI响应错误                          │
│    ├── 线程死锁 → 超时重置               │
│    ├── 窗口崩溃 → 自动重启               │
│    └── 事件堆积 → 队列管理               │
└─────────────────────────────────────────┘
```

### 调试工具

```python
class DebugConfig:
    """调试配置参数"""
    
    # 日志设置
    enable_debug_log: bool = True
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_file: str = "vggt_attention_debug.log"
    
    # 性能监控
    enable_profiling: bool = False
    profile_memory: bool = True
    profile_compute: bool = True
    
    # 中间结果保存
    save_intermediate_results: bool = False
    intermediate_save_dir: str = "debug_outputs/"
    save_attention_maps: bool = False
    save_coordinate_info: bool = False
    
    # 可视化调试
    show_patch_grid: bool = False  # 显示patch网格
    show_attention_stats: bool = True  # 显示attention统计
    highlight_source_patch: bool = True  # 高亮源patch
```

## 扩展接口

### 自定义Attention层

```python
def register_custom_attention_layer(layer_name: str, 
                                  extraction_func: Callable):
    """
    注册自定义attention层提取函数
    
    Args:
        layer_name: 层名称标识
        extraction_func: 提取函数,签名为:
            func(attention_maps, images, img_idx, 
                 patch_y, patch_x) -> Dict
    """
    pass

def add_attention_aggregation_method(method_name: str,
                                   aggregation_func: Callable):
    """
    添加自定义attention聚合方法
    
    支持的内置方法:
    - 'mean': 多头平均
    - 'max': 最大值池化  
    - 'weighted': 加权平均
    - 'pca': 主成分分析
    """
    pass
```

### 可视化扩展

```python
def register_custom_colormap(name: str, colormap: Any):
    """注册自定义颜色映射"""
    pass

def add_visualization_overlay(overlay_func: Callable):
    """添加可视化叠加层"""
    pass

def export_attention_data(format: str = 'json'):
    """
    导出attention数据
    支持格式: 'json', 'npy', 'csv', 'hdf5'
    """
    pass
```

## 总结

VGGT Interactive Attention Visualizer通过深度集成的模型hook系统、精确的坐标映射算法和高效的可视化管道，实现了对VGGT模型attention机制的实时交互式分析。系统具有高度的可配置性和扩展性，支持从模型参数到GUI界面的全方位定制，为深度学习模型的可解释性研究提供了强大的工具支持。

主要技术亮点:
- 🔧 **Hook-based Attention捕获**: 无需修改模型结构即可提取attention
- 🎯 **像素级精确交互**: 支持任意像素点击的patch-level attention分析  
- 🖼️ **多图像联合分析**: 同时支持最多6张图像的multi-view attention可视化
- ⚡ **性能优化**: 混合精度、内存管理、多线程等优化策略
- 🎨 **灵活可视化**: 支持多种布局、颜色映射和导出格式
- 🔍 **实时交互**: 鼠标悬停即时坐标反馈，点击即时attention分析

该系统为理解视觉Transformer在多视图几何任务中的attention模式提供了直观且强大的分析工具。
