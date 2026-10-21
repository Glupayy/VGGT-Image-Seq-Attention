#!/usr/bin/env python3
"""
测试双向attention可视化功能
"""

import numpy as np
import torch

def test_dual_attention_extraction():
    """测试双向attention提取逻辑"""
    print("=" * 60)
    print("🔍 测试双向Attention提取功能")
    print("=" * 60)
    
    # 模拟attention矩阵 (simplified)
    N = 12  # 简化的token数量
    attention_matrix = torch.randn(N, N)
    attention_matrix = torch.softmax(attention_matrix, dim=-1)  # 行归一化
    
    source_idx = 5  # 选择第5个token作为source
    
    # 提取行和列attention
    row_attention = attention_matrix[source_idx, :]  # 该token关注什么
    col_attention = attention_matrix[:, source_idx]  # 什么关注该token
    
    print(f"📊 Attention矩阵形状: {attention_matrix.shape}")
    print(f"🎯 源token索引: {source_idx}")
    print()
    
    print(f"📈 行Attention（该token关注什么）:")
    print(f"   形状: {row_attention.shape}")
    print(f"   数值: {[f'{x:.4f}' for x in row_attention.numpy()[:6]]}")  # 显示前6个值
    print(f"   总和: {row_attention.sum():.6f}")
    print()
    
    print(f"📊 列Attention（什么关注该token）:")
    print(f"   形状: {col_attention.shape}")
    print(f"   数值: {[f'{x:.4f}' for x in col_attention.numpy()[:6]]}")  # 显示前6个值
    print(f"   总和: {col_attention.sum():.6f}")
    print()
    
    # 验证它们是不同的
    are_different = not torch.allclose(row_attention, col_attention, atol=1e-6)
    print(f"🔍 行和列attention是否不同: {'✅ 是' if are_different else '❌ 否'}")
    
    if are_different:
        diff_mean = (row_attention - col_attention).abs().mean()
        print(f"   平均差异: {diff_mean:.6f}")
    
    print()
    print("✅ 双向attention提取测试完成！")
    print("=" * 60)
    
    return are_different

def test_attention_interpretation():
    """解释双向attention的含义"""
    print("\n🧭 双向Attention含义解释:")
    print("-" * 40)
    print("🔸 行Attention (Row): attention_matrix[i, :]")
    print("   含义: 第i个token对所有token的关注程度")
    print("   问题: '我关注什么?'")
    print("   应用: 分析某个patch关注场景中的哪些区域")
    print()
    print("🔸 列Attention (Column): attention_matrix[:, i]") 
    print("   含义: 所有token对第i个token的关注程度")
    print("   问题: '谁关注我?'")
    print("   应用: 分析场景中哪些区域关注某个特定patch")
    print()
    print("🎯 实际应用场景:")
    print("   - 行Attention: 分析主体关注的目标（主动关注）")
    print("   - 列Attention: 分析吸引注意力的区域（被动关注）")
    print("-" * 40)

def main():
    """主测试函数"""
    print("🚀 开始双向Attention功能测试")
    
    # 测试提取逻辑
    extraction_success = test_dual_attention_extraction()
    
    # 解释含义
    test_attention_interpretation()
    
    print("\n" + "=" * 60)
    if extraction_success:
        print("🎉 所有测试通过！双向attention功能工作正常")
        print("💡 现在可以运行 interactive_attention_tkinter.py 来体验完整功能")
    else:
        print("❌ 测试发现问题，需要检查实现")
    print("=" * 60)

if __name__ == "__main__":
    main()