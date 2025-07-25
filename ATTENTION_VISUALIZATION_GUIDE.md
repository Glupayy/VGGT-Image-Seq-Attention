# VGGT Attention Maps 可视化指南

## 概述

这个notebook展示了如何优雅地可视化VGGT（Visual Geometry Grounded Transformer）模型内部的attention机制。VGGT使用了独特的交替注意力机制，包括Frame Attention和Global Attention。

## 主要功能

### 1. Attention Maps捕获
通过PyTorch hooks在指定层捕获attention权重：
```python
attention_maps, hooks = create_attention_visualizer(model, target_layers=[0, 5, 11, 17, 23])
```

### 2. 多层次可视化
- **热图可视化**: 显示完整的attention矩阵
- **空间映射**: 将attention权重映射到图像空间
- **统计分析**: 分析attention的集中程度和分布

### 3. 两种Attention类型
- **Frame Attention**: 在单帧内进行空间交互，关注帧内的空间关系
- **Global Attention**: 在所有帧之间进行时序交互，关注跨帧的时序关系

## 关键发现

通过可视化分析，我们观察到了以下重要现象：

### 层次化学习过程
1. **浅层 (Layer 0-5)**: 
   - Attention相对分散，熵值较高 (6.9+)
   - 学习基础的视觉特征

2. **中层 (Layer 11)**: 
   - 开始出现特定的attention模式
   - 熵值适中，开始形成结构化表示

3. **深层 (Layer 23)**: 
   - Attention高度集中，熵值较低 (3.0-3.7)
   - 专注于任务相关的关键区域

### Attention模式分析
- **Frame Attention**: 主要关注同一帧内patch tokens之间的空间关系
- **Global Attention**: 负责整合跨帧信息，建立时序一致性

## 使用方法

### 基础使用
```python
# 1. 创建可视化器
attention_maps, hooks = create_attention_visualizer(model)

# 2. 运行模型以捕获attention
with torch.no_grad():
    predictions = model(images)

# 3. 可视化attention热图
plot_attention_heatmap(attention_maps, layer_idx=11, attn_type='global')

# 4. 清理资源
cleanup_hooks(hooks)
```

### 高级功能
```python
# 统计分析
analyze_attention_statistics(attention_maps)

# 简化可视化
simple_attention_visualization(attention_maps, images, layer_idx=23, attn_type='frame')
```

## 技术细节

### 内存优化
- 只监控指定的关键层以节省内存
- 立即将attention权重移到CPU
- 只保留部分attention heads
- 处理较小的图像批次

### Token结构
VGGT的token序列结构：
- Camera tokens (1个)
- Register tokens (4个) 
- Patch tokens (根据图像大小确定)

### 可视化技巧
- 使用不同颜色映射区分attention类型
- 添加网格线标识不同token类型
- 支持多头并行可视化
- 提供统计信息辅助分析

## 扩展应用

这个可视化工具可以用于：
1. **模型调试**: 理解模型的注意力机制
2. **特征分析**: 观察不同层学到的特征
3. **性能优化**: 识别注意力瓶颈
4. **科研分析**: 深入理解transformer架构

## 文件输出
- `attention_checkpoint.pth`: 保存的attention maps数据
- 各类可视化图片（如启用保存功能）

---

**注意**: 运行这个可视化需要足够的GPU内存。如果遇到内存不足，可以：
- 减少监控的层数
- 降低图像分辨率 
- 减少batch size
- 只保留关键的attention heads
