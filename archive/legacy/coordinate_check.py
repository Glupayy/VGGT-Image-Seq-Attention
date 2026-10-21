#!/usr/bin/env python3
"""
坐标系统一致性检查工具
验证从鼠标点击到attention提取到最终可视化的坐标处理是否一致
"""

def check_coordinate_consistency():
    """检查坐标系统的一致性"""
    
    print("🔍 分析坐标系统一致性...")
    print("="*60)
    
    # 模拟参数
    image_width = 518
    image_height = 518
    patch_size = 14
    
    # 模拟点击坐标
    click_x = 150  # 水平像素坐标
    click_y = 200  # 垂直像素坐标
    
    print(f"📍 模拟鼠标点击坐标: ({click_x}, {click_y})")
    print()
    
    # === 1. Tkinter GUI 中的坐标转换 ===
    print("1️⃣  Tkinter GUI 坐标转换:")
    print(f"   - 原始点击: x={click_x}, y={click_y}")
    
    # 计算图像索引 (假设单张图像)
    col = click_x // image_width
    row = click_y // image_height
    img_idx = row * 3 + col  # 假设3列布局
    
    # 计算图像内相对坐标
    local_x = click_x % image_width
    local_y = click_y % image_height
    
    # 计算patch坐标
    patch_x = local_x // patch_size  # 水平patch索引
    patch_y = local_y // patch_size  # 垂直patch索引
    
    print(f"   - 图像索引: {img_idx}")
    print(f"   - 局部坐标: local_x={local_x}, local_y={local_y}")
    print(f"   - Patch坐标: patch_x={patch_x}, patch_y={patch_y}")
    print()
    
    # === 2. attn_util.py 中的patch索引计算 ===
    print("2️⃣  attn_util.py 中的patch索引计算:")
    
    patch_h = image_height // patch_size
    patch_w = image_width // patch_size
    
    print(f"   - Patch网格: {patch_h}h × {patch_w}w")
    
    # source_patch_global_idx计算 (来自extract_single_patch_attention_simple)
    tokens_per_frame = 1 + 4 + patch_h * patch_w  # camera + register + patches
    source_patch_global_idx = (img_idx * tokens_per_frame + 5 + 
                             patch_y * patch_w + patch_x)
    
    print(f"   - tokens_per_frame: {tokens_per_frame}")
    print(f"   - patch在token序列中的索引: {patch_y * patch_w + patch_x}")
    print(f"   - 全局token索引: {source_patch_global_idx}")
    print()
    
    # === 3. 验证patch坐标到索引的转换 ===
    print("3️⃣  Patch坐标↔索引转换验证:")
    
    # 从patch坐标到线性索引
    linear_idx = patch_y * patch_w + patch_x
    print(f"   - patch_y={patch_y} × patch_w={patch_w} + patch_x={patch_x} = {linear_idx}")
    
    # 从线性索引回到patch坐标
    recovered_patch_y = linear_idx // patch_w
    recovered_patch_x = linear_idx % patch_w
    print(f"   - 反向计算: linear_idx={linear_idx} // {patch_w} = {recovered_patch_y}, {linear_idx} % {patch_w} = {recovered_patch_x}")
    
    # 检查是否一致
    consistent = (recovered_patch_x == patch_x and recovered_patch_y == patch_y)
    print(f"   - 坐标一致性: {'✅ 一致' if consistent else '❌ 不一致'}")
    print()
    
    # === 4. 最终显示中的patch矩形坐标 ===
    print("4️⃣  最终可视化中的patch矩形:")
    
    # 在最终显示的图像中，patch矩形应该覆盖的像素范围
    rect_start_x = patch_x * patch_size
    rect_end_x = (patch_x + 1) * patch_size
    rect_start_y = patch_y * patch_size 
    rect_end_y = (patch_y + 1) * patch_size
    
    print(f"   - Patch矩形范围:")
    print(f"     x: [{rect_start_x}, {rect_end_x})")
    print(f"     y: [{rect_start_y}, {rect_end_y})")
    print(f"   - 原始点击 ({local_x}, {local_y}) 是否在矩形内:")
    
    in_rect_x = rect_start_x <= local_x < rect_end_x
    in_rect_y = rect_start_y <= local_y < rect_end_y
    in_rect = in_rect_x and in_rect_y
    
    print(f"     x方向: {rect_start_x} <= {local_x} < {rect_end_x} = {in_rect_x}")
    print(f"     y方向: {rect_start_y} <= {local_y} < {rect_end_y} = {in_rect_y}")
    print(f"     整体: {'✅ 在矩形内' if in_rect else '❌ 不在矩形内'}")
    print()
    
    # === 5. 总结坐标系统约定 ===
    print("5️⃣  坐标系统约定总结:")
    print("   🎯 鼠标点击坐标: (x, y) = (水平, 垂直)")
    print("   🎯 Patch坐标: (patch_x, patch_y) = (水平索引, 垂直索引)")
    print("   🎯 线性索引: patch_y * patch_w + patch_x")
    print("   🎯 Token索引: img_idx * tokens_per_frame + 5 + linear_idx")
    print()
    
    # === 6. 潜在问题检查 ===
    print("6️⃣  潜在问题检查:")
    
    issues = []
    
    # 检查是否有xy坐标混淆
    if patch_x * patch_size > local_x or patch_y * patch_size > local_y:
        issues.append("⚠️  Patch坐标计算可能有误")
    
    # 检查patch索引是否合理
    if linear_idx >= patch_h * patch_w:
        issues.append("⚠️  线性索引超出范围")
    
    # 检查token索引是否合理  
    if source_patch_global_idx >= tokens_per_frame:
        issues.append("⚠️  全局token索引可能超出单帧范围")
    
    if not issues:
        print("   ✅ 未发现明显问题")
    else:
        for issue in issues:
            print(f"   {issue}")
    
    print()
    print("="*60)
    print("💡 建议检查点:")
    print("   1. 确认在visualize_patch_attention函数中绘制patch矩形时使用正确的坐标")
    print("   2. 验证attention提取时patch_y和patch_x的使用顺序")
    print("   3. 检查最终显示图像中矩形坐标与原始点击的对应关系")


if __name__ == "__main__":
    check_coordinate_consistency()
