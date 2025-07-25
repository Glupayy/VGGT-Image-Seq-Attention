#!/usr/bin/env python3
"""
测试修复后的VGGT attention可视化
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

def analyze_attention_token_structure(attention_maps, images):
    """
    分析attention maps中的token结构，帮助调试
    """
    print("=== VGGT Token Structure Analysis ===")
    
    if len(images.shape) == 5:
        B, S, C, H, W = images.shape
        images = images[0]  # 取第一个batch
    else:
        S, C, H, W = images.shape
    
    patch_size = 14
    patch_h, patch_w = H // patch_size, W // patch_size
    
    print(f"Input images: {S} frames, {H}x{W} pixels")
    print(f"Patch grid: {patch_h}x{patch_w} = {patch_h * patch_w} patches per frame")
    
    for key, attn_data in attention_maps.items():
        print(f"\n--- {key} ---")
        attention = attn_data['attention']  # [B, num_heads, N, N]
        num_heads = attn_data['num_heads']
        
        if attention.shape[0] > 1:
            attention = attention[0]  # 取第一个batch
        
        N = attention.shape[1]  # token数量
        print(f"Attention matrix: {num_heads} heads, {N}x{N} tokens")
        
        # 分析token类型
        if 'frame' in key:
            # Frame attention分析
            expected_tokens = 1 + 4 + patch_h * patch_w  # camera + register + patches
            print(f"Frame attention expected tokens: 1(camera) + 4(register) + {patch_h * patch_w}(patches) = {expected_tokens}")
            
            if N == expected_tokens:
                print("✓ Token count matches expected structure")
                patch_start = 5
            elif N > expected_tokens:
                print(f"⚠ More tokens than expected (got {N}, expected {expected_tokens})")
                patch_start = N - patch_h * patch_w  # 假设patch tokens在最后
            else:
                print(f"⚠ Fewer tokens than expected (got {N}, expected {expected_tokens})")
                patch_start = max(5, N - patch_h * patch_w)  # 保守估计
            
            print(f"Estimated patch_start: {patch_start}")
            print(f"Estimated patch tokens: {N - patch_start}")
            
        else:
            # Global attention分析
            expected_tokens = S * (1 + 4 + patch_h * patch_w)  # frames * (camera + register + patches)
            print(f"Global attention expected tokens: {S} frames × (1+4+{patch_h * patch_w}) = {expected_tokens}")
            
            if N == expected_tokens:
                print("✓ Token count matches expected structure")
                patch_start = S * 5
            else:
                print(f"⚠ Token count mismatch (got {N}, expected {expected_tokens})")
                # 尝试推断结构
                if N > S * 5:
                    patch_start = S * 5
                    available_patches = N - patch_start
                    patches_per_frame = available_patches // S if S > 0 else 0
                    print(f"Estimated {patches_per_frame} patches per frame")
                else:
                    patch_start = N // 2  # 保守估计
            
            print(f"Estimated patch_start: {patch_start}")
            print(f"Estimated total patch tokens: {N - patch_start}")
        
        # 分析attention权重分布
        sample_attn = attention[0]  # 取第一个head
        print(f"Attention weight stats: min={sample_attn.min():.6f}, max={sample_attn.max():.6f}, mean={sample_attn.mean():.6f}")
        
        # 检查special tokens的attention模式
        if N >= 5:
            special_attn = sample_attn[:5, :5]  # 前5个token之间的attention
            print(f"Special tokens attention range: {special_attn.min():.6f} - {special_attn.max():.6f}")


def simple_attention_visualization(attention_maps, images, layer_idx=0, attn_type='frame'):
    """
    简化的单层attention可视化，修复了reshape bug
    """
    key = f'{attn_type}_block_{layer_idx}'
    if key not in attention_maps:
        print(f"未找到 {key}")
        return
    
    attn_data = attention_maps[key]
    attention = attn_data['attention']
    if attention.shape[0] > 1:
        attention = attention[0]  # [num_heads, N, N]
    num_heads = attn_data['num_heads']
    
    print(f"Layer {layer_idx} ({attn_type}):")
    print(f"  Attention shape: {attention.shape}")
    print(f"  Number of heads: {num_heads}")
    
    # 选择前4个头进行可视化
    n_heads_to_show = min(4, num_heads)
    
    fig, axes = plt.subplots(1, n_heads_to_show, figsize=(16, 4))
    if n_heads_to_show == 1:
        axes = [axes]
    
    for head_idx in range(n_heads_to_show):
        ax = axes[head_idx]
        
        # 获取attention权重矩阵
        head_attn = attention[head_idx]  # [N, N]
        
        # 显示attention权重的热图
        im = ax.imshow(head_attn.numpy(), cmap='viridis', aspect='auto')
        ax.set_title(f'Head {head_idx}')
        ax.set_xlabel('Token Index')
        ax.set_ylabel('Token Index')
        
        # 添加colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle(f'Layer {layer_idx} - {attn_type.title()} Attention Matrix', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def create_mock_attention_maps():
    """
    创建模拟的attention maps进行测试
    """
    # 模拟4张518x518的图像
    S, H, W = 4, 518, 518
    patch_size = 14
    patch_h, patch_w = H // patch_size, W // patch_size  # 37 x 37
    
    attention_maps = {}
    
    # Frame attention: 每帧有 1 camera + 4 register + 37*37 patch tokens = 1374 tokens
    frame_tokens = 1 + 4 + patch_h * patch_w
    frame_attention = torch.randn(1, 8, frame_tokens, frame_tokens).softmax(dim=-1)
    
    attention_maps['frame_block_0'] = {
        'attention': frame_attention,
        'tokens_shape': (1, frame_tokens, 1024),
        'num_heads': 8
    }
    
    # Global attention: 4帧总共有 4 * 1374 = 5496 tokens
    global_tokens = S * frame_tokens
    global_attention = torch.randn(1, 8, global_tokens, global_tokens).softmax(dim=-1)
    
    attention_maps['global_block_0'] = {
        'attention': global_attention,
        'tokens_shape': (1, global_tokens, 1024),
        'num_heads': 8
    }
    
    # 创建模拟图像
    images = torch.randn(S, 3, H, W)
    
    return attention_maps, images


def test_attention_visualization():
    """
    测试修复后的attention可视化
    """
    print("创建模拟数据...")
    attention_maps, images = create_mock_attention_maps()
    
    print("分析token结构...")
    analyze_attention_token_structure(attention_maps, images)
    
    print("\n测试简化可视化...")
    try:
        simple_attention_visualization(attention_maps, images, layer_idx=0, attn_type='frame')
        print("✓ Frame attention可视化成功！")
    except Exception as e:
        print(f"✗ Frame attention可视化失败: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        simple_attention_visualization(attention_maps, images, layer_idx=0, attn_type='global')
        print("✓ Global attention可视化成功！")
    except Exception as e:
        print(f"✗ Global attention可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_attention_visualization()
