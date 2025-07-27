# VGGT Attention 可视化系统 - 技术总结

## 项目概述
VGGT (Visual Geometry Grounded Transformer) 模型的多视图attention可视化系统，支持交互式点击分析不同patch的attention patterns。

## 核心技术架构

### 1. Attention缓存系统 (Cache-based)
- **文件**: `attention_cache.py`
- **核心类**: `AttentionCacheManager`
- **作用**: 替代hook-based方式，直接缓存模型前向传播时的attention权重
- **优势**: 更准确，避免重复计算，代表模型实际使用的attention

### 2. VGGT模型结构理解
- **Token结构**: 每帧1041个tokens = 1(camera) + 4(register) + 1036(patches)
- **图像尺寸**: 实际392×518 (not 518×518), patch_size=14
- **Patch grid**: 28×37 = 1036 patches per frame
- **Global attention**: N=6246 (6帧×1041tokens)
- **Alternating attention**: Frame attention vs Global attention

### 3. 坐标映射修复
- **问题**: Canvas显示坐标与原始图像坐标不匹配
- **解决**: 记录display_scale_x/y，正确映射点击坐标到原始图像坐标
- **文件**: `interactive_attention_tkinter.py:182-184`

### 4. 关键文件修改记录

#### attention_cache.py
```python
# 修复patch grid推断 (line 182-186)
if actual_patch_tokens == 1036:
    patch_h, patch_w = 28, 37  # 392÷14=28, 518÷14=37
    
# 默认值修复 (line 271-272)  
patch_h = self.model_info.get('patch_h', 28)  # 不是37
patch_w = self.model_info.get('patch_w', 37)
```

#### vggt/layers/attention_vis.py
```python
# 支持实际观察到的N=6246结构 (line 131-133)
if N == 6246:  # 实际观察到的Global Attention (392×518图像)
    print(f"✅ 这是实际的Global Attention! (N=6246, 6帧×1041tokens)")
```

#### interactive_attention_tkinter.py
```python
# 坐标映射修复 (line 182-184)
original_x = x / self.display_scale_x
original_y = y / self.display_scale_y
```

## 技术要点

### Fused Attention处理
- `fused_attn=True`: 使用PyTorch优化的F.scaled_dot_product_attention，需重新计算attention matrix用于缓存
- `fused_attn=False`: 使用标准attention计算，可直接缓存attention matrix

### 模型信息推断
系统能自动识别两种配置：
1. **实际结构**: 392×518图像，28×37 patch grid，N=6246
2. **理论结构**: 518×518图像，37×37 patch grid，N=8244

### 重要修复历史
1. **Softmax归一化误解**: attention weights本身已经softmax归一化，无需额外处理
2. **Patch grid计算错误**: 从错误的32×32修复为正确的28×37
3. **Token结构验证**: 支持N=6246作为有效的global attention结构
4. **坐标映射问题**: Canvas显示缩放导致的点击坐标偏移

## 系统运行状态
- ✅ Cache系统正常工作，替代了hook-based方式  
- ✅ 坐标映射已修复，点击位置准确
- ✅ 支持实际图像尺寸392×518和28×37 patch grid
- ✅ 不同patch显示不同attention patterns
- ✅ 多视图attention可视化正常
- ✅ 高分辨率可视化已实现（2025-07-27）

## 分辨率优化改进 (2025-07-27)

### 主要改进
1. **动态分辨率控制**: 添加"高分辨率模式"复选框，用户可自主选择分辨率
2. **DPI提升**: 高质量模式从150 DPI提升到300 DPI
3. **图形尺寸优化**: 
   - 高质量模式: figsize增大50% (4→6, 8→12, 12→18, 10→15)
   - 标准模式: 保持原始尺寸以确保兼容性
4. **窗口尺寸适配**:
   - 主窗口: 1000x700 → 1300x900
   - 结果窗口: 高质量1500x1100, 标准900x700
   - 图像显示: 高质量1400x1000, 标准800x600
5. **插值质量提升**: 高质量模式使用bicubic插值，标准模式使用bilinear
6. **热图显示优化**: alpha调整为0.85，添加bilinear插值，提升数值显示精度(:.6f)

### 技术特点
- **向后兼容**: 保持标准分辨率模式确保在低配置设备上正常运行
- **动态切换**: 运行时可以切换分辨率模式，无需重启
- **内存友好**: 只有在需要时才使用高分辨率设置
- **用户体验**: 默认启用高分辨率模式，提供最佳视觉效果

## 待优化项目
- 支持更多层的attention分析
- 优化内存使用和缓存管理
- 添加attention权重导出功能

## 关键命令
```bash
# 运行交互式可视化器
python interactive_attention_tkinter.py

# 清理GPU缓存
torch.cuda.empty_cache()
```

## 注意事项
- 始终使用cache-based系统，避免hook-based方式
- 图像预处理会影响最终的patch数量和token结构
- Canvas坐标需要正确映射到原始图像坐标
- attention weights已经过softmax归一化，不要重复处理
- 高分辨率模式会消耗更多内存和计算资源，但提供更好的视觉效果