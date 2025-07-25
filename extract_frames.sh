#!/bin/bash

# 从指定文件夹中均匀抽取32帧的bash脚本

# 默认参数
SOURCE_DIR="/mnt/8t/caizhuojun/ego4d/ARKitScenes/vsi_train_highres/raw/Training/40753679/vga_wide"
OUTPUT_DIR="./extracted_frames"
NUM_FRAMES=32

# 显示使用说明
show_usage() {
    echo "用法: $0 <源文件夹路径> [选项]"
    echo "选项:"
    echo "  -o <输出路径>    指定输出文件夹 (默认: ./extracted_frames)"
    echo "  -n <帧数>       要抽取的帧数 (默认: 32)"
    echo "  -h              显示此帮助信息"
    echo ""
    echo "例子:"
    echo "  $0 /mnt/8t/caizhuojun/ego4d/ARKitScenes/vsi_train_highres/raw/Training/40753679/vga_wide"
    echo "  $0 /path/to/frames -o ./output -n 16"
}

# 解析命令行参数
while [ $# -gt 0 ]; do
    case $1 in
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -n|--num-frames)
            NUM_FRAMES="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            if [ -z "$SOURCE_DIR" ]; then
                SOURCE_DIR="$1"
            else
                echo "错误: 未知参数 $1"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# 检查源文件夹参数
if [ -z "$SOURCE_DIR" ]; then
    echo "错误: 请指定源文件夹路径"
    show_usage
    exit 1
fi

# 检查源文件夹是否存在
if [ ! -d "$SOURCE_DIR" ]; then
    echo "错误: 源文件夹不存在: $SOURCE_DIR"
    exit 1
fi

echo "源文件夹: $SOURCE_DIR"
echo "输出文件夹: $OUTPUT_DIR"
echo "抽取帧数: $NUM_FRAMES"
echo ""

# 创建输出文件夹
mkdir -p "$OUTPUT_DIR"

# 获取所有PNG文件并按名称排序，使用临时文件
temp_file=$(mktemp)
find "$SOURCE_DIR" -name "*.png" -type f | sort > "$temp_file"

# 计算文件数量
total_files=$(wc -l < "$temp_file")

if [ "$total_files" -eq 0 ]; then
    echo "错误: 在源文件夹中没有找到PNG文件"
    rm "$temp_file"
    exit 1
fi

echo "找到 $total_files 个PNG文件"

# 如果文件数少于请求的帧数，调整帧数
if [ "$total_files" -lt "$NUM_FRAMES" ]; then
    echo "警告: 文件数量($total_files)少于请求的帧数($NUM_FRAMES)，将抽取所有文件"
    NUM_FRAMES=$total_files
fi

echo "将从 $total_files 帧中均匀抽取 $NUM_FRAMES 帧"

# 计算间隔并抽取文件
i=0
while [ $i -lt $NUM_FRAMES ]; do
    if [ $NUM_FRAMES -eq 1 ]; then
        index=$((total_files / 2 + 1))
    else
        index=$((i * (total_files - 1) / (NUM_FRAMES - 1) + 1))
    fi
    
    source_file=$(sed -n "${index}p" "$temp_file")
    filename=$(basename "$source_file")
    target_file="$OUTPUT_DIR/frame_$(printf "%03d" $i)_$filename"
    
    cp "$source_file" "$target_file"
    echo "复制: $filename -> frame_$(printf "%03d" $i)_$filename"
    
    i=$((i + 1))
done

# 清理临时文件
rm "$temp_file"

echo ""
echo "成功! 已将 $NUM_FRAMES 帧保存到: $OUTPUT_DIR"
