#!/usr/bin/env python3
"""
从指定文件夹中均匀抽取32帧的脚本
"""

import os
import shutil
import argparse
from pathlib import Path
import re

def extract_timestamp_from_filename(filename):
    """从文件名中提取时间戳进行排序"""
    # 匹配文件名中的时间戳部分，例如: 40753679_6845.023.png -> 6845.023
    match = re.search(r'_(\d+\.\d+)\.png$', filename)
    if match:
        return float(match.group(1))
    return 0.0

def extract_frames(source_dir, output_dir, num_frames=32):
    """
    从源文件夹中均匀抽取指定数量的帧
    
    Args:
        source_dir: 源文件夹路径
        output_dir: 输出文件夹路径  
        num_frames: 要抽取的帧数，默认32
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 检查源文件夹是否存在
    if not source_path.exists():
        print(f"错误: 源文件夹不存在: {source_dir}")
        return False
    
    # 获取所有PNG文件
    png_files = list(source_path.glob("*.png"))
    
    if not png_files:
        print(f"错误: 在 {source_dir} 中没有找到PNG文件")
        return False
    
    # 按时间戳排序
    png_files.sort(key=lambda x: extract_timestamp_from_filename(x.name))
    
    total_frames = len(png_files)
    print(f"找到 {total_frames} 个PNG文件")
    
    if total_frames < num_frames:
        print(f"警告: 文件数量({total_frames})少于请求的帧数({num_frames})")
        num_frames = total_frames
    
    # 创建输出文件夹
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 计算抽取间隔
    if num_frames == 1:
        indices = [total_frames // 2]
    else:
        indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    
    print(f"将从 {total_frames} 帧中均匀抽取 {num_frames} 帧")
    print(f"抽取的索引: {indices}")
    
    # 复制选中的文件
    for i, idx in enumerate(indices):
        source_file = png_files[idx]
        # 创建新的文件名，保持顺序
        new_filename = f"frame_{i:03d}_{source_file.name}"
        target_file = output_path / new_filename
        
        try:
            shutil.copy2(source_file, target_file)
            print(f"复制: {source_file.name} -> {new_filename}")
        except Exception as e:
            print(f"错误: 复制文件 {source_file.name} 失败: {e}")
            return False
    
    print(f"\n成功! 已将 {len(indices)} 帧保存到: {output_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="从文件夹中均匀抽取指定数量的帧")
    parser.add_argument("source_dir", help="源文件夹路径")
    parser.add_argument("-o", "--output", help="输出文件夹路径", default="./extracted_frames")
    parser.add_argument("-n", "--num_frames", type=int, default=32, help="要抽取的帧数 (默认: 32)")
    
    args = parser.parse_args()
    
    success = extract_frames(args.source_dir, args.output, args.num_frames)
    
    if success:
        print("抽取完成!")
    else:
        print("抽取失败!")
        exit(1)

if __name__ == "__main__":
    main()
