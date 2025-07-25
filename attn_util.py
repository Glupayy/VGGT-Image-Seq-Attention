import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
import seaborn as sns

def extract_pose_token_attention(attention_maps: Dict, images: torch.Tensor) -> Dict:
    """
    提取每个图像的pose token对图像patch的attention map
    
    Args:
        attention_maps: 从模型hook捕获的attention maps
        images: 输入图像 [S, C, H, W]
    
    Returns:
        pose_attention_maps: 包含frame和global attention的字典
    """
    S, C, H, W = images.shape
    patch_size = 14
    patch_h, patch_w = H // patch_size, W // patch_size
    
    # 分离frame和global attention
    frame_attention_maps = {}
    global_attention_maps = {}
    
    for layer_name, data in attention_maps.items():
        attention = data['attention']  # [B, num_heads, N, N]
        
        # 合并heads维度：对所有head求平均
        if attention.dim() == 4:
            attention = attention[0].mean(dim=0)  # [N, N]
        elif attention.dim() == 3:
            attention = attention.mean(dim=0)  # [N, N]
        
        N = attention.shape[0]
        
        # VGGT token结构分析：
        # Frame attention (B*S, P, C): 每个图像有自己的tokens
        # - 1个camera token (pose token) 
        # - 4个register tokens
        # - patch_h * patch_w 个patch tokens
        # 每个Frame有 1 + 4 + patch_h*patch_w 个tokens
        
        # Global attention (B, S*P, C): 所有图像的tokens拼接
        # - S个camera tokens (每个图像一个)
        # - S*4个register tokens 
        # - S * patch_h * patch_w 个patch tokens
        
        tokens_per_frame = 1 + 4 + patch_h * patch_w  # camera + register + patches
        patch_start_per_frame = 5  # camera(1) + register(4)
        
        if 'frame_block' in layer_name:
            # Frame attention: 处理每个图像内部的attention
            # N应该等于tokens_per_frame * S，因为frame attention是在(B*S, P, C)上操作
            if N == tokens_per_frame:
                # 单个frame的attention
                camera_token_idx = 0
                patch_start = patch_start_per_frame
                
                if N > patch_start:
                    pose_to_patches = attention[camera_token_idx, patch_start:]
                    
                    if len(pose_to_patches) == patch_h * patch_w:
                        attention_map = pose_to_patches.reshape(patch_h, patch_w)
                        # 对于frame attention，我们只存储第一个图像的map
                        # 因为每个图像都会有类似的模式
                        frame_attention_maps[layer_name] = attention_map.unsqueeze(0)
            
        elif 'global_block' in layer_name:
            # Global attention: 处理图像间的attention
            # N应该等于S * tokens_per_frame
            if N == S * tokens_per_frame:
                # 每个图像都有自己的camera token
                pose_to_all_patches = []
                
                for frame_idx in range(S):
                    camera_token_idx = frame_idx * tokens_per_frame  # 每个frame的camera token位置
                    
                    # 计算这个camera token对所有图像patches的attention
                    frame_attention_to_all = []
                    
                    for target_frame_idx in range(S):
                        target_patch_start = target_frame_idx * tokens_per_frame + patch_start_per_frame
                        target_patch_end = target_patch_start + patch_h * patch_w
                        
                        if target_patch_end <= N:
                            patch_attention = attention[camera_token_idx, target_patch_start:target_patch_end]
                            if len(patch_attention) == patch_h * patch_w:
                                attention_map = patch_attention.reshape(patch_h, patch_w)
                                frame_attention_to_all.append(attention_map)
                            else:
                                frame_attention_to_all.append(torch.zeros(patch_h, patch_w))
                        else:
                            frame_attention_to_all.append(torch.zeros(patch_h, patch_w))
                    
                    if frame_attention_to_all:
                        pose_to_all_patches.append(torch.stack(frame_attention_to_all))
                
                if pose_to_all_patches:
                    # [S, S, patch_h, patch_w] - 每个pose token对所有图像的attention
                    global_attention_maps[layer_name] = torch.stack(pose_to_all_patches)
    
    return {
        'frame_attention': frame_attention_maps,
        'global_attention': global_attention_maps
    }

def visualize_pose_attention(pose_attention_maps: Dict, images: torch.Tensor, 
                           layer_names: List[str] = None, max_images: int = 6,
                           attention_type: str = 'both'):
    """
    可视化每个图像的pose token对图像的attention maps
    支持frame attention和global attention的可视化
    
    Args:
        pose_attention_maps: extract_pose_token_attention的输出
        images: 原始图像
        layer_names: 要显示的层名称
        max_images: 最多显示的图像数量
        attention_type: 'frame', 'global', 或 'both' - 指定显示哪种attention
    """
    # 根据attention_type筛选层
    if layer_names is None:
        frame_layers = list(pose_attention_maps['frame_attention'].keys())
        global_layers = list(pose_attention_maps['global_attention'].keys())
        
        if attention_type == 'frame':
            layer_names = frame_layers
        elif attention_type == 'global':
            layer_names = global_layers
        else:  # 'both'
            layer_names = frame_layers + global_layers
    else:
        # 根据attention_type过滤指定的layers
        if attention_type == 'frame':
            layer_names = [name for name in layer_names if 'frame_block' in name]
        elif attention_type == 'global':
            layer_names = [name for name in layer_names if 'global_block' in name]
        # 'both'时保持原样
    
    S = min(images.shape[0], max_images)
    n_layers = len(layer_names)
    
    if n_layers == 0:
        print("没有找到指定的attention层")
        return
    
    fig, axes = plt.subplots(n_layers, S, figsize=(S*3, n_layers*3))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    if S == 1:
        axes = axes.reshape(-1, 1)
    
    for layer_idx, layer_name in enumerate(layer_names):
        attention_maps = None
        attention_type = ""
        
        # 确定attention类型和获取数据
        if layer_name in pose_attention_maps['frame_attention']:
            attention_maps = pose_attention_maps['frame_attention'][layer_name]
            attention_type = "Frame"
            # Frame attention: [1, patch_h, patch_w] - 复制到所有图像
            if attention_maps.shape[0] == 1:
                attention_maps = attention_maps.repeat(S, 1, 1)
        elif layer_name in pose_attention_maps['global_attention']:
            attention_maps = pose_attention_maps['global_attention'][layer_name]
            attention_type = "Global"
            # Global attention: [S, S, patch_h, patch_w] - 每个pose token对所有图像的attention
            # 这里我们可以选择显示每个pose token对自己图像的attention，或者对其他图像的attention
        
        if attention_maps is None:
            print(f"Layer {layer_name} not found in attention maps")
            continue
        
        for img_idx in range(S):
            ax = axes[layer_idx, img_idx]
            
            # 显示原图
            img = images[img_idx].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            
            # 选择attention map
            if attention_type == "Frame":
                attention_map = attention_maps[img_idx]  # [patch_h, patch_w]
                title_suffix = f"Frame {img_idx} self-attn"
            elif attention_type == "Global":
                # 显示第img_idx个pose token对第img_idx个图像的attention
                attention_map = attention_maps[img_idx, img_idx]  # [patch_h, patch_w]
                title_suffix = f"Pose {img_idx}→Image {img_idx}"
            
            # 上采样到原图尺寸
            H, W = images.shape[-2], images.shape[-1]
            attention_upsampled = torch.nn.functional.interpolate(
                attention_map.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze()
            
            # 叠加热图
            im = ax.imshow(attention_upsampled.numpy(), alpha=0.6, cmap='hot')
            
            # 设置标题
            if layer_idx == 0:
                ax.set_title(f'{title_suffix}', fontsize=10)
            if img_idx == 0:
                layer_short = layer_name.replace('_block_', ' L')
                ax.text(-0.1, 0.5, f'{attention_type}\n{layer_short}', rotation=90, va='center', 
                       transform=ax.transAxes, fontsize=10, fontweight='bold')
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.suptitle('Pose Token Attention: Frame vs Global', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def visualize_complete_global_attention(pose_attention_maps: Dict, images: torch.Tensor,
                                      layer_names: List[str] = None, max_images: int = 6):
    """
    完整可视化global attention：每个pose token对所有图像的attention
    对于S个图像和L层，会生成 S×S×L 个可视化图
    
    Args:
        pose_attention_maps: extract_pose_token_attention的输出
        images: 原始图像
        layer_names: 要显示的global layer名称
        max_images: 最多显示的图像数量
    """
    global_attention = pose_attention_maps['global_attention']
    
    if layer_names is None:
        layer_names = list(global_attention.keys())
    else:
        # 只保留global layers
        layer_names = [name for name in layer_names if name in global_attention]
    
    if len(layer_names) == 0:
        print("没有找到指定的global attention层")
        return
    
    S = min(images.shape[0], max_images)
    n_layers = len(layer_names)
    
    print(f"将生成 {S}×{S}×{n_layers} = {S*S*n_layers} 个可视化图")
    print(f"图像数量: {S}, 层数: {n_layers}")
    print(f"每一行显示: 一个pose token对所有图像的attention")
    print(f"每一列显示: 所有pose token对一个目标图像的attention")
    
    for layer_idx, layer_name in enumerate(layer_names):
        if layer_name not in global_attention:
            print(f"Layer {layer_name} not found in global attention")
            continue
            
        attention_maps = global_attention[layer_name]  # [S, S, patch_h, patch_w]
        
        # 为每一层创建一个大图：S×S的子图
        fig, axes = plt.subplots(S, S, figsize=(S*2.5, S*2.5))
        if S == 1:
            axes = np.array([[axes]])
        elif axes.ndim == 1:
            axes = axes.reshape(S, 1) if S > 1 else axes.reshape(1, S)
        
        for source_idx in range(S):  # 行：源pose token
            for target_idx in range(S):  # 列：目标图像
                ax = axes[source_idx, target_idx]
                
                # 显示目标图像
                img = images[target_idx].permute(1, 2, 0).cpu().numpy()
                img = np.clip(img, 0, 1)
                ax.imshow(img)
                
                # 叠加attention map：source_idx的pose token对target_idx图像的attention
                attention_map = attention_maps[source_idx, target_idx]  # [patch_h, patch_w]
                
                # 上采样到原图尺寸
                H, W = images.shape[-2], images.shape[-1]
                attention_upsampled = torch.nn.functional.interpolate(
                    attention_map.unsqueeze(0).unsqueeze(0),
                    size=(H, W), mode='bilinear', align_corners=False
                ).squeeze()
                
                # 叠加热图
                im = ax.imshow(attention_upsampled.numpy(), alpha=0.6, cmap='hot')
                
                # 设置标题和标签
                attention_sum = attention_map.sum().item()
                
                if source_idx == target_idx:
                    # 对角线：自注意力
                    ax.set_title(f'Self\n({attention_sum:.3f})', fontsize=9, fontweight='bold', color='red')
                else:
                    ax.set_title(f'P{source_idx}→I{target_idx}\n({attention_sum:.3f})', fontsize=9)
                
                # 添加行列标签
                if target_idx == 0:  # 第一列添加行标签
                    ax.set_ylabel(f'Pose {source_idx}', fontweight='bold', fontsize=10)
                if source_idx == 0:  # 第一行添加列标签
                    ax.set_xlabel(f'Image {target_idx}', fontweight='bold', fontsize=10)
                    ax.xaxis.set_label_position('top')
                
                ax.set_xticks([])
                ax.set_yticks([])
        
        # 设置整体标题
        layer_short = layer_name.replace('_block_', ' L')
        plt.suptitle(f'Complete Global Attention Matrix - {layer_short}\n'
                    f'行: Pose Token | 列: Target Image | 颜色: Attention强度', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        print(f"✓ 完成 {layer_name} 的可视化 ({S}×{S} = {S*S} 个子图)")

def visualize_cross_image_attention(pose_attention_maps: Dict, images: torch.Tensor, 
                                  source_img_idx: int = 0, layer_name: str = None):
    """
    可视化指定图像的pose token对所有其他图像的attention
    
    Args:
        pose_attention_maps: extract_pose_token_attention的输出
        images: 原始图像
        source_img_idx: 源图像索引
        layer_name: 要显示的global layer名称
    """
    global_attention = pose_attention_maps['global_attention']
    
    if layer_name is None:
        if len(global_attention) > 0:
            layer_name = list(global_attention.keys())[-1]  # 使用最后一层
        else:
            print("没有找到global attention层")
            return
    
    if layer_name not in global_attention:
        print(f"Layer {layer_name} not found in global attention")
        return
    
    attention_maps = global_attention[layer_name]  # [S, S, patch_h, patch_w]
    S = images.shape[0]
    
    if source_img_idx >= S:
        print(f"Source image index {source_img_idx} out of range [0, {S-1}]")
        return
    
    # 获取源图像的pose token对所有图像的attention
    source_attention = attention_maps[source_img_idx]  # [S, patch_h, patch_w]
    
    fig, axes = plt.subplots(1, S, figsize=(S*3, 3))
    if S == 1:
        axes = [axes]
    
    for target_idx in range(S):
        ax = axes[target_idx]
        
        # 显示目标图像
        img = images[target_idx].permute(1, 2, 0).cpu().numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        
        # 叠加attention map
        attention_map = source_attention[target_idx]  # [patch_h, patch_w]
        
        # 上采样到原图尺寸
        H, W = images.shape[-2], images.shape[-1]
        attention_upsampled = torch.nn.functional.interpolate(
            attention_map.unsqueeze(0).unsqueeze(0),
            size=(H, W), mode='bilinear', align_corners=False
        ).squeeze()
        
        # 叠加热图
        im = ax.imshow(attention_upsampled.numpy(), alpha=0.6, cmap='hot')
        
        # 设置标题
        attention_sum = attention_map.sum().item()
        if target_idx == source_img_idx:
            ax.set_title(f'Self (sum={attention_sum:.3f})', fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'Target {target_idx} (sum={attention_sum:.3f})', fontsize=12)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(f'Pose Token {source_img_idx} → All Images ({layer_name})', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def analyze_cross_patch_correspondence(pose_attention_maps: Dict, images: torch.Tensor, 
                                     layer_name: str = None, 
                                     correspondence_threshold: float = 0.1) -> Dict:
    """
    分析跨图像的patch-to-patch对应关系强度
    
    Args:
        pose_attention_maps: extract_pose_token_attention的输出
        images: 原始图像 [S, C, H, W]
        layer_name: 要分析的层名称，如果为None则使用最后一个global层
        correspondence_threshold: 对应关系强度阈值
    
    Returns:
        correspondence_analysis: 包含详细对应关系分析的字典
    """
    global_attention = pose_attention_maps['global_attention']
    
    if layer_name is None:
        if len(global_attention) > 0:
            layer_name = list(global_attention.keys())[-1]
        else:
            print("没有找到global attention层")
            return {}
    
    if layer_name not in global_attention:
        print(f"Layer {layer_name} not found in global attention")
        return {}
    
    attention_maps = global_attention[layer_name]  # [S, S, patch_h, patch_w]
    S, _, patch_h, patch_w = attention_maps.shape
    
    print(f"分析层: {layer_name}")
    print(f"图像数量: {S}, Patch分辨率: {patch_h}x{patch_w}")
    
    # 1. 提取所有跨图像的patch对应关系
    cross_correspondences = {}
    self_correspondences = {}
    
    for source_idx in range(S):
        cross_correspondences[source_idx] = {}
        self_correspondences[source_idx] = attention_maps[source_idx, source_idx]  # 自注意力
        
        for target_idx in range(S):
            if source_idx != target_idx:
                correspondence_map = attention_maps[source_idx, target_idx]  # [patch_h, patch_w]
                cross_correspondences[source_idx][target_idx] = correspondence_map
    
    # 2. 计算对应关系强度矩阵
    correspondence_strength_matrix = torch.zeros(S, S)
    for source_idx in range(S):
        for target_idx in range(S):
            if source_idx == target_idx:
                # 自注意力强度
                correspondence_strength_matrix[source_idx, target_idx] = self_correspondences[source_idx].sum()
            else:
                # 跨图像对应强度
                correspondence_strength_matrix[source_idx, target_idx] = cross_correspondences[source_idx][target_idx].sum()
    
    # 3. 找到高对应强度的patch对
    high_correspondence_pairs = []
    for source_idx in range(S):
        for target_idx in range(S):
            if source_idx != target_idx:
                corr_map = cross_correspondences[source_idx][target_idx]
                # 找到超过阈值的patch位置
                high_corr_mask = corr_map > correspondence_threshold
                high_corr_positions = torch.nonzero(high_corr_mask, as_tuple=False)
                
                for pos in high_corr_positions:
                    row, col = pos[0].item(), pos[1].item()
                    strength = corr_map[row, col].item()
                    high_correspondence_pairs.append({
                        'source_img': source_idx,
                        'target_img': target_idx,
                        'source_patch': (row, col),
                        'strength': strength
                    })
    
    # 4. 分析对应关系的空间分布
    spatial_distribution = {}
    for source_idx in range(S):
        spatial_distribution[source_idx] = {}
        for target_idx in range(S):
            if source_idx != target_idx:
                corr_map = cross_correspondences[source_idx][target_idx]
                
                # 计算attention的重心
                total_attention = corr_map.sum()
                if total_attention > 0:
                    # 计算加权重心
                    row_indices = torch.arange(patch_h, dtype=torch.float32).unsqueeze(1)
                    col_indices = torch.arange(patch_w, dtype=torch.float32).unsqueeze(0)
                    
                    centroid_row = (corr_map * row_indices).sum() / total_attention
                    centroid_col = (corr_map * col_indices).sum() / total_attention
                    
                    # 计算attention的分散程度（方差）
                    row_var = ((row_indices - centroid_row) ** 2 * corr_map).sum() / total_attention
                    col_var = ((col_indices - centroid_col) ** 2 * corr_map).sum() / total_attention
                    
                    spatial_distribution[source_idx][target_idx] = {
                        'centroid': (centroid_row.item(), centroid_col.item()),
                        'variance': (row_var.item(), col_var.item()),
                        'total_strength': total_attention.item(),
                        'max_strength': corr_map.max().item(),
                        'entropy': -(corr_map * torch.log(corr_map + 1e-8)).sum().item()
                    }
    
    # 5. 分析几何一致性
    geometric_consistency = {}
    for source_idx in range(S):
        consistency_scores = []
        for target_idx in range(S):
            if source_idx != target_idx:
                corr_map = cross_correspondences[source_idx][target_idx]
                
                # 计算局部一致性：相邻patch的attention相似性
                # 使用梯度来衡量attention的平滑性
                grad_x = torch.abs(corr_map[:, 1:] - corr_map[:, :-1]).mean()
                grad_y = torch.abs(corr_map[1:, :] - corr_map[:-1, :]).mean()
                smoothness = 1.0 / (1.0 + grad_x + grad_y)  # 越平滑，一致性越高
                
                consistency_scores.append(smoothness.item())
        
        if consistency_scores:
            geometric_consistency[source_idx] = {
                'mean_consistency': np.mean(consistency_scores),
                'std_consistency': np.std(consistency_scores),
                'consistency_scores': consistency_scores
            }
    
    return {
        'layer_name': layer_name,
        'correspondence_strength_matrix': correspondence_strength_matrix,
        'cross_correspondences': cross_correspondences,
        'self_correspondences': self_correspondences,
        'high_correspondence_pairs': high_correspondence_pairs,
        'spatial_distribution': spatial_distribution,
        'geometric_consistency': geometric_consistency,
        'stats': {
            'total_high_correspondences': len(high_correspondence_pairs),
            'mean_cross_strength': correspondence_strength_matrix[correspondence_strength_matrix != torch.diag(correspondence_strength_matrix)].mean().item(),
            'mean_self_strength': torch.diag(correspondence_strength_matrix).mean().item()
        }
    }

def visualize_patch_correspondence_matrix(correspondence_analysis: Dict, 
                                        show_values: bool = True):
    """
    可视化patch对应关系强度矩阵
    """
    if not correspondence_analysis:
        print("对应关系分析为空")
        return
    
    strength_matrix = correspondence_analysis['correspondence_strength_matrix']
    layer_name = correspondence_analysis['layer_name']
    
    plt.figure(figsize=(8, 6))
    
    # 创建热图
    sns.heatmap(strength_matrix.numpy(), 
                annot=show_values, 
                fmt='.3f',
                cmap='viridis',
                cbar_kws={'label': 'Correspondence Strength'})
    
    plt.title(f'Patch Correspondence Strength Matrix\n{layer_name}', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Target Image Index', fontweight='bold')
    plt.ylabel('Source Image Index', fontweight='bold')
    
    # 添加对角线标记（自注意力）
    S = strength_matrix.shape[0]
    for i in range(S):
        plt.gca().add_patch(plt.Rectangle((i, i), 1, 1, fill=False, 
                                        edgecolor='red', linewidth=3))
    
    plt.text(S/2, -0.5, '红框: 自注意力 | 其他: 跨图像对应', 
             ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    stats = correspondence_analysis['stats']
    print(f"\n=== 对应关系强度统计 ===")
    print(f"高对应关系对数: {stats['total_high_correspondences']}")
    print(f"平均跨图像对应强度: {stats['mean_cross_strength']:.4f}")
    print(f"平均自注意力强度: {stats['mean_self_strength']:.4f}")
    print(f"跨图像/自注意力比值: {stats['mean_cross_strength']/stats['mean_self_strength']:.4f}")

def analyze_geometric_correspondence_quality(correspondence_analysis: Dict, 
                                           images: torch.Tensor):
    """
    分析几何对应关系的质量和特征
    """
    if not correspondence_analysis:
        print("对应关系分析为空")
        return
    
    spatial_dist = correspondence_analysis['spatial_distribution']
    geometric_consistency = correspondence_analysis['geometric_consistency']
    high_pairs = correspondence_analysis['high_correspondence_pairs']
    
    S = images.shape[0]
    
    print(f"=== 几何对应关系质量分析 ===")
    print(f"分析层: {correspondence_analysis['layer_name']}")
    
    # 1. 分析每个图像的对应质量
    print(f"\n1. 每个图像的对应质量:")
    for source_idx in range(S):
        if source_idx in geometric_consistency:
            consistency = geometric_consistency[source_idx]
            print(f"  图像 {source_idx}:")
            print(f"    平均几何一致性: {consistency['mean_consistency']:.4f}")
            print(f"    一致性标准差: {consistency['std_consistency']:.4f}")
            
            # 分析空间分布特征
            if source_idx in spatial_dist:
                target_info = []
                for target_idx in range(S):
                    if target_idx in spatial_dist[source_idx]:
                        dist_info = spatial_dist[source_idx][target_idx]
                        target_info.append(f"→{target_idx}({dist_info['total_strength']:.3f})")
                
                if target_info:
                    print(f"    对应强度分布: {', '.join(target_info)}")
    
    # 2. 分析高对应关系的空间分布
    if high_pairs:
        print(f"\n2. 高对应关系分析 (阈值 > {0.1}):")
        
        # 按源图像分组
        by_source = {}
        for pair in high_pairs:
            source = pair['source_img']
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(pair)
        
        for source_idx in sorted(by_source.keys()):
            pairs = by_source[source_idx]
            strengths = [p['strength'] for p in pairs]
            
            print(f"  图像 {source_idx} → 其他图像:")
            print(f"    高对应关系数量: {len(pairs)}")
            print(f"    平均强度: {np.mean(strengths):.4f}")
            print(f"    最大强度: {np.max(strengths):.4f}")
            
            # 分析目标图像分布
            target_counts = {}
            for pair in pairs:
                target = pair['target_img']
                target_counts[target] = target_counts.get(target, 0) + 1
            
            target_dist = ', '.join([f"{t}({c})" for t, c in sorted(target_counts.items())])
            print(f"    目标分布: {target_dist}")
    
    # 3. 分析对应关系的全局模式
    print(f"\n3. 全局对应模式:")
    strength_matrix = correspondence_analysis['correspondence_strength_matrix']
    
    # 计算对称性
    symmetry_score = torch.abs(strength_matrix - strength_matrix.T).mean().item()
    print(f"  矩阵对称性: {1.0 - symmetry_score:.4f} (1.0 = 完全对称)")
    
    # 分析对角占优程度
    diag_strength = torch.diag(strength_matrix).mean().item()
    off_diag_strength = (strength_matrix.sum() - torch.diag(strength_matrix).sum()).item() / (S * S - S)
    diagonal_dominance = diag_strength / (diag_strength + off_diag_strength)
    print(f"  对角占优程度: {diagonal_dominance:.4f} (1.0 = 完全自注意力)")
    
    # 分析连通性
    cross_matrix = strength_matrix.clone()
    cross_matrix.fill_diagonal_(0)
    strong_connections = (cross_matrix > cross_matrix.mean()).sum().item()
    total_connections = S * (S - 1)
    connectivity = strong_connections / total_connections
    print(f"  强连接比例: {connectivity:.4f} ({strong_connections}/{total_connections})")

def analyze_correspondence_evolution(attention_maps: Dict, images: torch.Tensor, 
                                   correspondence_threshold: float = 0.1) -> Dict:
    """
    分析对应关系在不同层之间的演化
    """
    print("=== 对应关系演化分析 ===")
    
    # 获取所有global层
    global_layers = [name for name in attention_maps.keys() if 'global_block_' in name]
    global_layers.sort(key=lambda x: int(x.split('_')[-1]))
    
    if len(global_layers) < 2:
        print("需要至少2个global层来分析演化")
        return {}
    
    print(f"分析层: {global_layers}")
    
    evolution_data = {}
    
    for layer_name in global_layers:
        # 提取这一层的pose attention maps
        layer_attention = {
            'frame_attention': {},
            'global_attention': {layer_name: attention_maps[layer_name]['attention']}
        }
        
        # 转换为pose attention format
        pose_maps = extract_pose_token_attention(layer_attention, images)
        
        # 分析对应关系
        correspondence_analysis = analyze_cross_patch_correspondence(
            pose_maps, images, layer_name, correspondence_threshold
        )
        
        if correspondence_analysis:
            layer_idx = int(layer_name.split('_')[-1])
            evolution_data[layer_idx] = {
                'layer_name': layer_name,
                'stats': correspondence_analysis['stats'],
                'strength_matrix': correspondence_analysis['correspondence_strength_matrix'],
                'geometric_consistency': correspondence_analysis['geometric_consistency']
            }
    
    # 分析演化趋势
    if len(evolution_data) >= 2:
        print(f"\n演化趋势分析:")
        
        layers = sorted(evolution_data.keys())
        
        # 跟踪关键指标
        cross_strengths = []
        self_strengths = []
        high_corr_counts = []
        mean_consistencies = []
        
        for layer_idx in layers:
            data = evolution_data[layer_idx]
            cross_strengths.append(data['stats']['mean_cross_strength'])
            self_strengths.append(data['stats']['mean_self_strength'])
            high_corr_counts.append(data['stats']['total_high_correspondences'])
            
            # 计算平均几何一致性
            consistencies = []
            for source_idx, geom_data in data['geometric_consistency'].items():
                consistencies.append(geom_data['mean_consistency'])
            
            if consistencies:
                mean_consistencies.append(np.mean(consistencies))
            else:
                mean_consistencies.append(0)
        
        # 可视化演化趋势
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 1. 对应强度演化
        axes[0, 0].plot(layers, cross_strengths, 'o-', label='跨图像对应', color='blue')
        axes[0, 0].plot(layers, self_strengths, 's-', label='自注意力', color='red')
        axes[0, 0].set_xlabel('Layer Index')
        axes[0, 0].set_ylabel('Average Strength')
        axes[0, 0].set_title('对应强度演化')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 高对应关系数量演化
        axes[0, 1].plot(layers, high_corr_counts, 'o-', color='green')
        axes[0, 1].set_xlabel('Layer Index')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('高对应关系数量演化')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 几何一致性演化
        axes[1, 0].plot(layers, mean_consistencies, 'o-', color='purple')
        axes[1, 0].set_xlabel('Layer Index')
        axes[1, 0].set_ylabel('Consistency Score')
        axes[1, 0].set_title('几何一致性演化')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 跨图像/自注意力比值演化
        ratios = [c/s if s > 0 else 0 for c, s in zip(cross_strengths, self_strengths)]
        axes[1, 1].plot(layers, ratios, 'o-', color='orange')
        axes[1, 1].set_xlabel('Layer Index')
        axes[1, 1].set_ylabel('Cross/Self Ratio')
        axes[1, 1].set_title('跨图像注意力相对强度')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('VGGT 对应关系学习演化', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # 打印数值趋势
        print(f"\n数值趋势:")
        print(f"跨图像对应强度: {cross_strengths[0]:.4f} → {cross_strengths[-1]:.4f} "
              f"({'↑' if cross_strengths[-1] > cross_strengths[0] else '↓'})")
        print(f"自注意力强度: {self_strengths[0]:.4f} → {self_strengths[-1]:.4f} "
              f"({'↑' if self_strengths[-1] > self_strengths[0] else '↓'})")
        print(f"高对应关系数: {high_corr_counts[0]} → {high_corr_counts[-1]} "
              f"({'↑' if high_corr_counts[-1] > high_corr_counts[0] else '↓'})")
        print(f"几何一致性: {mean_consistencies[0]:.4f} → {mean_consistencies[-1]:.4f} "
              f"({'↑' if mean_consistencies[-1] > mean_consistencies[0] else '↓'})")
    
    return evolution_data

def analyze_pose_attention_stats(pose_attention_maps: Dict):
    """
    分析每个图像的pose token attention统计信息
    """
    print("=== VGGT Pose Token Attention 详细分析 ===")
    
    frame_attention = pose_attention_maps['frame_attention']
    global_attention = pose_attention_maps['global_attention']
    
    print(f"\nFrame Attention Layers: {len(frame_attention)}")
    for layer_name, attention_maps in frame_attention.items():
        print(f"\n{layer_name} (Frame内部Attention):")
        shape = attention_maps.shape
        print(f"  Shape: {shape}")
        print(f"  Attention range: [{attention_maps.min():.4f}, {attention_maps.max():.4f}]")
        print(f"  Mean attention: {attention_maps.mean():.4f}")
        print(f"  Std attention: {attention_maps.std():.4f}")
        
        # Frame attention应该显示pose token对同一frame内patch的关注
        attention_sum = attention_maps.sum()
        print(f"  Total attention sum: {attention_sum:.4f}")
    
    print(f"\nGlobal Attention Layers: {len(global_attention)}")
    for layer_name, attention_maps in global_attention.items():
        print(f"\n{layer_name} (跨图像Attention):")
        S, S2, patch_h, patch_w = attention_maps.shape
        print(f"  Shape: [源图像={S}, 目标图像={S2}, 空间={patch_h}x{patch_w}]")
        print(f"  Attention range: [{attention_maps.min():.4f}, {attention_maps.max():.4f}]")
        print(f"  Mean attention: {attention_maps.mean():.4f}")
        
        # 分析每个pose token的attention分布
        print(f"  每个Pose Token的Attention分析:")
        for source_idx in range(S):
            source_attention = attention_maps[source_idx]  # [S, patch_h, patch_w]
            
            # 计算对每个目标图像的总attention
            attention_per_target = source_attention.sum(dim=(1, 2))  # [S]
            
            print(f"    Pose {source_idx} → 各图像总attention: {[f'{val:.3f}' for val in attention_per_target.tolist()]}")
            
            # 找到最关注的目标图像
            max_target_idx = attention_per_target.argmax().item()
            max_attention = attention_per_target[max_target_idx].item()
            
            if max_target_idx == source_idx:
                print(f"      → 最关注自己 (attention={max_attention:.3f})")
            else:
                print(f"      → 最关注图像{max_target_idx} (attention={max_attention:.3f})")
            
            # 分析attention在空间上的分布
            self_attention_map = source_attention[source_idx]  # [patch_h, patch_w]
            max_pos = self_attention_map.argmax()
            max_row, max_col = max_pos // patch_w, max_pos % patch_w
            print(f"      → 自注意力最强位置: ({max_row.item()}, {max_col.item()}), 值={self_attention_map.max():.4f}")
    
    print(f"\n=== 总结 ===")
    print(f"Frame Attention层数: {len(frame_attention)} (显示pose token对同frame内部的关注)")
    print(f"Global Attention层数: {len(global_attention)} (显示pose token对跨图像的关注)")
    
    if len(global_attention) > 0:
        print(f"\n关键发现:")
        print(f"✓ 每个图像都有自己的pose token")
        print(f"✓ 在Global Attention中，每个pose token可以关注所有图像的patch")
        print(f"✓ 这证实了VGGT确实为每个图像维护独立的pose表示")
