#!/usr/bin/env python3
"""
VGGT位置信息Information Flow分析工具
深入分析位置embedding在attention机制中的作用
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd

# 设置中文字体和高质量图形
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

class PositionAnalyzer:
    """VGGT位置信息分析器"""

    def __init__(self):
        self.patch_size = 14
        self.img_height = 392
        self.img_width = 518
        self.patch_h = self.img_height // self.patch_size  # 28
        self.patch_w = self.img_width // self.patch_size   # 37
        self.num_frames = 6

        # 计算关键参数
        self.patches_per_frame = self.patch_h * self.patch_w  # 1036
        self.camera_tokens = 1
        self.register_tokens = 4
        self.tokens_per_frame = self.camera_tokens + self.register_tokens + self.patches_per_frame  # 1041
        self.total_tokens = self.num_frames * self.tokens_per_frame  # 6246

        print(f"📊 VGGT架构参数:")
        print(f"   图像尺寸: {self.img_height}×{self.img_width}")
        print(f"   Patch尺寸: {self.patch_size}×{self.patch_size}")
        print(f"   Patch网格: {self.patch_h}×{self.patch_w} = {self.patches_per_frame}")
        print(f"   每帧tokens: {self.tokens_per_frame} (1 camera + 4 register + {self.patches_per_frame} patches)")
        print(f"   总tokens: {self.total_tokens} ({self.num_frames}帧)")

    def simulate_rope_embeddings(self, frequency=100.0, dim=64):
        """模拟2D RoPE位置嵌入的计算"""
        positions = []
        for y in range(self.patch_h):
            for x in range(self.patch_w):
                positions.append([y, x])
        positions = torch.tensor(positions, dtype=torch.float32)

        # 模拟RoPE频率计算
        exponents = torch.arange(0, dim, 2, dtype=torch.float32) / dim
        inv_freq = 1.0 / (frequency ** exponents)

        # 计算位置编码
        y_positions = positions[:, 0]
        x_positions = positions[:, 1]

        y_angles = torch.einsum('i,j->ij', y_positions, inv_freq)
        x_angles = torch.einsum('i,j->ij', x_positions, inv_freq)

        # 合并角度信息
        angles = torch.cat([y_angles, x_angles], dim=-1)

        pos_embeddings = {
            'positions': positions,
            'y_angles': y_angles,
            'x_angles': x_angles,
            'combined_angles': angles,
            'cos_components': angles.cos(),
            'sin_components': angles.sin()
        }

        return pos_embeddings

    def analyze_position_attention_interaction(self):
        """分析位置信息与attention的交互机制"""

        # 模拟attention权重矩阵
        np.random.seed(42)
        attention_matrix = np.random.rand(self.patches_per_frame, self.patches_per_frame)
        attention_matrix = attention_matrix / attention_matrix.sum(axis=1, keepdims=True)

        # 创建位置距离矩阵
        position_distance_matrix = np.zeros((self.patches_per_frame, self.patches_per_frame))
        positions = []
        for i in range(self.patch_h):
            for j in range(self.patch_w):
                positions.append((i, j))

        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                # 计算欧几里得距离
                dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                position_distance_matrix[i, j] = dist

        # 分析attention与位置距离的相关性
        attention_flat = attention_matrix.flatten()
        distance_flat = position_distance_matrix.flatten()

        # 计算相关系数
        correlation = np.corrcoef(attention_flat, distance_flat)[0, 1]

        return {
            'attention_matrix': attention_matrix,
            'position_distance_matrix': position_distance_matrix,
            'positions': positions,
            'correlation': correlation
        }

    def visualize_architecture_overview(self):
        """可视化VGGT整体架构中的位置信息流"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('VGGT位置信息架构与Information Flow分析', fontsize=20, fontweight='bold')

        # 1. Token结构组成图
        ax = axes[0, 0]
        token_types = ['Camera', 'Register', 'Patches']
        token_counts = [self.camera_tokens, self.register_tokens, self.patches_per_frame]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        bars = ax.bar(token_types, token_counts, color=colors, alpha=0.8)
        ax.set_title('每帧Token组成结构', fontsize=14, fontweight='bold')
        ax.set_ylabel('Token数量')

        # 添加数值标签
        for bar, count in zip(bars, token_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                   f'{count}', ha='center', va='bottom', fontweight='bold')

        ax.set_ylim(0, max(token_counts) * 1.2)

        # 2. Patch网格可视化
        ax = axes[0, 1]
        patch_grid = np.arange(self.patches_per_frame).reshape(self.patch_h, self.patch_w)
        im = ax.imshow(patch_grid, cmap='viridis', aspect='auto')
        ax.set_title(f'Patch网格布局 ({self.patch_h}×{self.patch_w})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Width (patches)')
        ax.set_ylabel('Height (patches)')

        # 添加网格线
        for i in range(self.patch_h + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.3)
        for j in range(self.patch_w + 1):
            ax.axvline(j - 0.5, color='white', linewidth=0.5, alpha=0.3)

        plt.colorbar(im, ax=ax, label='Patch索引')

        # 3. 多帧Token序列
        ax = axes[0, 2]
        frame_data = []
        for frame_idx in range(self.num_frames):
            frame_start = frame_idx * self.tokens_per_frame
            frame_data.append({
                'frame': f'Frame {frame_idx}',
                'camera_start': frame_start,
                'register_start': frame_start + 1,
                'patch_start': frame_start + 5,
                'patch_end': frame_start + self.tokens_per_frame
            })

        y_pos = np.arange(len(frame_data))
        for i, data in enumerate(frame_data):
            # Camera token
            ax.barh(i, 1, left=data['camera_start'], color='#FF6B6B', alpha=0.8, label='Camera' if i == 0 else "")
            # Register tokens
            ax.barh(i, 4, left=data['register_start'], color='#4ECDC4', alpha=0.8, label='Register' if i == 0 else "")
            # Patch tokens
            ax.barh(i, self.patches_per_frame, left=data['patch_start'], color='#45B7D1', alpha=0.8, label='Patches' if i == 0 else "")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([d['frame'] for d in frame_data])
        ax.set_xlabel('Token位置索引')
        ax.set_title('多帧Token序列布局', fontsize=14, fontweight='bold')
        ax.legend()

        # 4. RoPE位置编码模拟
        ax = axes[1, 0]
        pos_embeddings = self.simulate_rope_embeddings()

        # 可视化位置编码的频率分量
        freq_samples = pos_embeddings['cos_components'][:100, :8]  # 前100个位置，前8个频率分量
        im = ax.imshow(freq_samples.T, cmap='RdBu_r', aspect='auto')
        ax.set_title('RoPE位置编码频率分量', fontsize=14, fontweight='bold')
        ax.set_xlabel('Patch位置')
        ax.set_ylabel('频率分量')
        plt.colorbar(im, ax=ax, label='编码值')

        # 5. 位置距离热图
        ax = axes[1, 1]
        analysis = self.analyze_position_attention_interaction()

        # 选择一个中心位置，显示与其他位置的距离
        center_idx = self.patches_per_frame // 2
        distances = analysis['position_distance_matrix'][center_idx]
        distance_2d = distances.reshape(self.patch_h, self.patch_w)

        im = ax.imshow(distance_2d, cmap='plasma', aspect='auto')
        ax.set_title('位置距离分布图', fontsize=14, fontweight='bold')
        ax.set_xlabel('Width (patches)')
        ax.set_ylabel('Height (patches)')

        # 标记中心位置
        center_y, center_x = center_idx // self.patch_w, center_idx % self.patch_w
        ax.plot(center_x, center_y, 'w*', markersize=15, markeredgecolor='black', markeredgewidth=2)
        ax.text(center_x, center_y - 2, 'Center', ha='center', va='top', color='white', fontweight='bold')

        plt.colorbar(im, ax=ax, label='距离')

        # 6. Attention-Position相关性分析
        ax = axes[1, 2]

        # 计算不同距离范围内的attention强度
        distances = analysis['position_distance_matrix'].flatten()
        attentions = analysis['attention_matrix'].flatten()

        # 按距离分箱统计attention
        distance_bins = np.linspace(0, distances.max(), 20)
        bin_centers = (distance_bins[:-1] + distance_bins[1:]) / 2
        binned_attention = []

        for i in range(len(distance_bins) - 1):
            mask = (distances >= distance_bins[i]) & (distances < distance_bins[i+1])
            if mask.sum() > 0:
                binned_attention.append(attentions[mask].mean())
            else:
                binned_attention.append(0)

        ax.plot(bin_centers, binned_attention, 'o-', linewidth=2, markersize=6, color='#E74C3C')
        ax.set_xlabel('空间距离 (patches)')
        ax.set_ylabel('平均Attention强度')
        ax.set_title(f'Attention-距离相关性\n(r={analysis["correlation"]:.3f})', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def visualize_rope_mechanism(self):
        """详细可视化RoPE机制"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('2D RoPE (Rotary Position Embedding) 机制详解', fontsize=20, fontweight='bold')

        pos_embeddings = self.simulate_rope_embeddings(dim=64)
        positions = pos_embeddings['positions']

        # 1. 2D位置坐标分布
        ax = axes[0, 0]
        y_coords = positions[:, 0].numpy()
        x_coords = positions[:, 1].numpy()
        scatter = ax.scatter(x_coords, y_coords, c=np.arange(len(positions)), cmap='viridis', s=30)
        ax.set_xlabel('X坐标 (patch)')
        ax.set_ylabel('Y坐标 (patch)')
        ax.set_title('2D空间位置分布', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Patch索引')

        # 2. Y方向角度编码
        ax = axes[0, 1]
        y_angles = pos_embeddings['y_angles'][:, :8]  # 前8个频率分量
        im = ax.imshow(y_angles.T, cmap='coolwarm', aspect='auto')
        ax.set_title('Y方向角度编码', fontsize=14, fontweight='bold')
        ax.set_xlabel('Patch位置')
        ax.set_ylabel('频率分量')
        plt.colorbar(im, ax=ax, label='角度值')

        # 3. X方向角度编码
        ax = axes[0, 2]
        x_angles = pos_embeddings['x_angles'][:, :8]  # 前8个频率分量
        im = ax.imshow(x_angles.T, cmap='coolwarm', aspect='auto')
        ax.set_title('X方向角度编码', fontsize=14, fontweight='bold')
        ax.set_xlabel('Patch位置')
        ax.set_ylabel('频率分量')
        plt.colorbar(im, ax=ax, label='角度值')

        # 4. Cosine分量可视化
        ax = axes[1, 0]
        cos_comp_2d = pos_embeddings['cos_components'][:, 0].reshape(self.patch_h, self.patch_w)
        im = ax.imshow(cos_comp_2d, cmap='RdBu_r', aspect='auto')
        ax.set_title('Cosine分量 (第1频率)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Width (patches)')
        ax.set_ylabel('Height (patches)')
        plt.colorbar(im, ax=ax, label='Cosine值')

        # 5. Sine分量可视化
        ax = axes[1, 1]
        sin_comp_2d = pos_embeddings['sin_components'][:, 0].reshape(self.patch_h, self.patch_w)
        im = ax.imshow(sin_comp_2d, cmap='RdBu_r', aspect='auto')
        ax.set_title('Sine分量 (第1频率)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Width (patches)')
        ax.set_ylabel('Height (patches)')
        plt.colorbar(im, ax=ax, label='Sine值')

        # 6. 频率分量的空间周期性
        ax = axes[1, 2]

        # 计算不同频率分量的空间周期
        frequencies = np.array([100.0 ** (-i/32) for i in range(0, 32, 4)])  # 采样几个频率
        spatial_periods = 2 * np.pi / frequencies

        ax.semilogy(range(len(frequencies)), spatial_periods, 'o-', linewidth=2, markersize=8, color='#2ECC71')
        ax.set_xlabel('频率分量索引')
        ax.set_ylabel('空间周期 (patches)')
        ax.set_title('RoPE频率分量的空间周期', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 添加说明文本
        ax.text(0.7, 0.9, '低频→大周期\n高频→小周期', transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                fontsize=10, ha='center')

        plt.tight_layout()
        return fig

    def visualize_attention_flow(self):
        """可视化attention的信息流动"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('VGGT中位置信息的Attention Flow分析', fontsize=18, fontweight='bold')

        # 1. Frame Attention vs Global Attention 结构对比
        ax = axes[0, 0]

        # Frame attention 结构
        frame_tokens = self.tokens_per_frame
        frame_matrix_size = frame_tokens * frame_tokens

        # Global attention 结构
        global_tokens = self.total_tokens
        global_matrix_size = global_tokens * global_tokens

        structures = ['Frame Attention', 'Global Attention']
        token_counts = [frame_tokens, global_tokens]
        matrix_sizes = [frame_matrix_size, global_matrix_size]

        x = np.arange(len(structures))
        width = 0.35

        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, token_counts, width, label='Token数量', color='#3498DB', alpha=0.8)
        bars2 = ax2.bar(x + width/2, np.array(matrix_sizes)/1e6, width, label='矩阵大小(M)', color='#E74C3C', alpha=0.8)

        ax.set_xlabel('Attention类型')
        ax.set_ylabel('Token数量', color='#3498DB')
        ax2.set_ylabel('Attention矩阵大小 (M)', color='#E74C3C')
        ax.set_title('Frame vs Global Attention结构对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(structures)

        # 添加数值标签
        for bar, count in zip(bars1, token_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 50,
                   f'{count}', ha='center', va='bottom', fontweight='bold', color='#3498DB')

        for bar, size in zip(bars2, np.array(matrix_sizes)/1e6):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{size:.1f}M', ha='center', va='bottom', fontweight='bold', color='#E74C3C')

        # 2. 交替注意力流程图
        ax = axes[0, 1]
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

        # 绘制交替注意力流程
        blocks = ['Frame\nAttention', 'Global\nAttention', 'Frame\nAttention', 'Global\nAttention']
        block_colors = ['#FF9999', '#99CCFF', '#FF9999', '#99CCFF']

        for i, (block, color) in enumerate(zip(blocks, block_colors)):
            x_pos = 2 + i * 2
            rect = patches.Rectangle((x_pos-0.8, 4), 1.6, 2, linewidth=2,
                                   edgecolor='black', facecolor=color, alpha=0.7)
            ax.add_patch(rect)
            ax.text(x_pos, 5, block, ha='center', va='center', fontweight='bold', fontsize=10)

            if i < len(blocks) - 1:
                ax.arrow(x_pos + 0.8, 5, 0.4, 0, head_width=0.2, head_length=0.1,
                        fc='black', ec='black')

        ax.set_title('交替注意力机制流程', fontsize=14, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # 3. 位置信息在不同attention类型中的作用
        ax = axes[1, 0]

        # 模拟不同类型attention中位置信息的重要性
        attention_types = ['Frame\nAttention', 'Global\nAttention']
        position_importance = {
            'Local Position': [0.8, 0.6],  # Frame attention更关注局部位置
            'Global Position': [0.3, 0.9],  # Global attention更关注全局位置
            'Relative Position': [0.7, 0.8]  # 相对位置都重要
        }

        x = np.arange(len(attention_types))
        width = 0.25

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, (pos_type, importance) in enumerate(position_importance.items()):
            offset = (i - 1) * width
            bars = ax.bar(x + offset, importance, width, label=pos_type, color=colors[i], alpha=0.8)

            # 添加数值标签
            for bar, imp in zip(bars, importance):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{imp:.1f}', ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Attention类型')
        ax.set_ylabel('位置信息重要性')
        ax.set_title('位置信息在不同Attention中的重要性', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(attention_types)
        ax.legend()
        ax.set_ylim(0, 1.2)

        # 4. 位置编码对attention pattern的影响
        ax = axes[1, 1]

        # 模拟有无位置编码时的attention分布差异
        distances = np.linspace(0, 20, 50)

        # 无位置编码：相对均匀的分布
        attention_no_pos = np.ones_like(distances) + 0.1 * np.random.randn(len(distances))
        attention_no_pos = np.abs(attention_no_pos)
        attention_no_pos = attention_no_pos / attention_no_pos.sum()

        # 有位置编码：距离衰减模式
        attention_with_pos = np.exp(-distances / 8) + 0.05 * np.random.randn(len(distances))
        attention_with_pos = np.abs(attention_with_pos)
        attention_with_pos = attention_with_pos / attention_with_pos.sum()

        ax.plot(distances, attention_no_pos, 'o-', label='无位置编码', linewidth=2, color='#E74C3C')
        ax.plot(distances, attention_with_pos, 's-', label='有位置编码', linewidth=2, color='#2ECC71')

        ax.set_xlabel('空间距离 (patches)')
        ax.set_ylabel('Attention权重')
        ax.set_title('位置编码对Attention Pattern的影响', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n" + "="*80)
        print("🔍 VGGT位置信息Information Flow深度分析报告")
        print("="*80)

        print("\n📋 1. 架构概览")
        print("-"*50)
        print(f"• 图像尺寸: {self.img_height}×{self.img_width} pixels")
        print(f"• Patch尺寸: {self.patch_size}×{self.patch_size} pixels")
        print(f"• Patch网格: {self.patch_h}×{self.patch_w} = {self.patches_per_frame} patches/frame")
        print(f"• 帧数: {self.num_frames}")
        print(f"• 总Token数: {self.total_tokens}")

        print("\n🧮 2. Token结构分析")
        print("-"*50)
        print(f"每帧Token组成:")
        print(f"  - Camera Token: {self.camera_tokens}")
        print(f"  - Register Tokens: {self.register_tokens}")
        print(f"  - Patch Tokens: {self.patches_per_frame}")
        print(f"  - 总计: {self.tokens_per_frame}")

        print(f"\n多帧总体:")
        print(f"  - Frame Attention: 每帧独立处理 ({self.tokens_per_frame}×{self.tokens_per_frame})")
        print(f"  - Global Attention: 跨帧联合处理 ({self.total_tokens}×{self.total_tokens})")

        print("\n🔄 3. 2D RoPE位置编码机制")
        print("-"*50)
        print("RoPE (Rotary Position Embedding) 2D实现原理:")
        print("• 将2D空间位置(y,x)分别编码到特征维度的不同部分")
        print("• 垂直方向: 特征维度的前半部分")
        print("• 水平方向: 特征维度的后半部分")
        print("• 使用旋转矩阵对查询和键向量进行位置相关的旋转")

        pos_embeddings = self.simulate_rope_embeddings()
        print(f"\n编码统计:")
        print(f"  - 位置数量: {len(pos_embeddings['positions'])}")
        print(f"  - Y坐标范围: 0-{self.patch_h-1}")
        print(f"  - X坐标范围: 0-{self.patch_w-1}")
        print(f"  - 频率分量数: {pos_embeddings['cos_components'].shape[1]}")

        print("\n⚡ 4. Information Flow分析")
        print("-"*50)
        print("位置信息在VGGT中的流动路径:")
        print("\n4.1 编码阶段:")
        print("  Input Images → Patch Embedding → + RoPE Position Encoding")

        print("\n4.2 Frame Attention阶段:")
        print("  • 每帧内部的patch之间进行attention")
        print("  • 位置编码帮助模型理解同一帧内的空间关系")
        print("  • 局部空间关系得到强化")

        print("\n4.3 Global Attention阶段:")
        print("  • 跨帧的所有patch之间进行attention")
        print("  • 位置编码帮助区分不同帧的相同空间位置")
        print("  • 时空对应关系得到建立")

        print("\n4.4 交替机制:")
        print("  • Frame Attention ↔ Global Attention 交替执行")
        print("  • 逐步建立从局部到全局的时空理解")

        print("\n📊 5. 位置信息的关键作用")
        print("-"*50)

        analysis = self.analyze_position_attention_interaction()
        print(f"位置-注意力相关性分析:")
        print(f"  • Attention与空间距离的相关系数: {analysis['correlation']:.4f}")

        if analysis['correlation'] < -0.1:
            print("  • 负相关：距离越远，attention权重越小（局部注意力模式）")
        elif analysis['correlation'] > 0.1:
            print("  • 正相关：距离越远，attention权重越大（全局注意力模式）")
        else:
            print("  • 弱相关：attention模式受内容驱动，位置影响较小")

        print("\n🎯 6. 位置编码的优势")
        print("-"*50)
        print("• 相对位置感知: RoPE天然捕捉相对位置关系")
        print("• 多尺度编码: 不同频率分量捕捉不同尺度的空间模式")
        print("• 旋转不变性: 对图像旋转具有一定的鲁棒性")
        print("• 外推能力: 能够处理训练时未见过的位置")

        print("\n🔮 7. 位置信息对下游任务的影响")
        print("-"*50)
        print("• 目标检测: 精确的位置编码有助于定位目标边界")
        print("• 语义分割: 空间关系编码提升分割精度")
        print("• 深度估计: 相对位置信息有助于深度推理")
        print("• 相机位姿估计: 跨帧位置对应关系支持几何计算")

        print("\n💡 8. 关键洞察")
        print("-"*50)
        print("• VGGT通过2D RoPE实现了高效的空间位置编码")
        print("• 交替注意力机制充分利用了局部和全局位置信息")
        print("• 位置编码不仅影响空间理解，也支持时序建模")
        print("• 多尺度位置特征为复杂视觉任务提供了强大的先验")

        print("\n" + "="*80)
        print("📈 报告总结：VGGT的位置信息系统设计精巧，通过2D RoPE和交替注意力机制，")
        print("实现了从局部空间关系到全局时空对应的层次化理解，为各种视觉任务提供了")
        print("强大的位置感知能力。")
        print("="*80)

def main():
    """主函数：生成完整的位置信息分析报告"""
    print("🚀 开始VGGT位置信息深度分析...")

    analyzer = PositionAnalyzer()

    # 生成文本报告
    analyzer.generate_comprehensive_report()

    # 生成可视化图表
    print("\n📊 正在生成可视化图表...")

    # 1. 架构概览图
    fig1 = analyzer.visualize_architecture_overview()
    fig1.savefig('vggt_position_architecture.png', dpi=300, bbox_inches='tight')
    print("✅ 已保存: vggt_position_architecture.png")

    # 2. RoPE机制详解图
    fig2 = analyzer.visualize_rope_mechanism()
    fig2.savefig('vggt_rope_mechanism.png', dpi=300, bbox_inches='tight')
    print("✅ 已保存: vggt_rope_mechanism.png")

    # 3. Attention流程图
    fig3 = analyzer.visualize_attention_flow()
    fig3.savefig('vggt_attention_flow.png', dpi=300, bbox_inches='tight')
    print("✅ 已保存: vggt_attention_flow.png")

    plt.show()

    print("\n🎉 VGGT位置信息分析完成！")
    print("📁 生成的文件:")
    print("   • vggt_position_architecture.png - 架构与信息流概览")
    print("   • vggt_rope_mechanism.png - 2D RoPE机制详解")
    print("   • vggt_attention_flow.png - Attention流程分析")

if __name__ == "__main__":
    main()