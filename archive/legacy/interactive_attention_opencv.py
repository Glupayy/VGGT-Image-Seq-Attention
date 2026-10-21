#!/usr/bin/env python3
"""
基于OpenCV的VGGT交互式Attention可视化工具
支持真正的鼠标点击获取坐标并显示attention结果
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import io
from typing import Dict, List, Tuple, Optional

# 导入VGGT相关模块
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# 导入现有的attention工具
from attn_util import extract_single_patch_attention_simple, pixel_to_patch_coords

class OpenCVAttentionVisualizer:
    def __init__(self):
        # 设备配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        
        # 初始化模型
        print("🚀 Loading VGGT model...")
        self.model = VGGT.from_pretrained("VGGT-1B").to(self.device)
        self.model.eval()
        
        # 存储当前状态
        self.current_images = None
        self.current_attention_maps = None
        self.patch_size = 14
        self.attention_hooks = []
        self.grid_config = None
        
        # OpenCV窗口状态
        self.window_name = "VGGT Interactive Attention Visualizer"
        self.grid_image = None
        self.display_image = None
        self.mouse_pos = (0, 0)
        
        print("✅ Model loaded successfully!")
    
    def setup_attention_hooks(self, target_layers=[0, 5, 11, 17, 23]):
        """设置attention hooks来捕获中间层的attention maps"""
        self.attention_maps = {}
        self.attention_hooks = []
        
        def create_hook(name):
            def hook(module, input, output):
                if hasattr(module, 'attn') and hasattr(module.attn, 'qkv'):
                    x = input[0]
                    B, N, C = x.shape
                    
                    if B > 1:
                        x = x[:1]
                        B = 1
                    
                    qkv = module.attn.qkv(x).reshape(B, N, 3, module.attn.num_heads, module.attn.head_dim).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    
                    num_heads_to_keep = min(8, module.attn.num_heads)
                    q = q[:, :num_heads_to_keep]
                    k = k[:, :num_heads_to_keep]
                    
                    attn = (q @ k.transpose(-2, -1)) * module.attn.scale
                    attn = attn.softmax(dim=-1)
                    
                    self.attention_maps[name] = {
                        'attention': attn.detach().cpu(),
                        'tokens_shape': (B, N, C),
                        'num_heads': num_heads_to_keep
                    }
                    
                    del attn, q, k, v, qkv
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            return hook
        
        # 清除旧的hooks
        for hook in self.attention_hooks:
            hook.remove()
        self.attention_hooks = []
        
        # 注册新的hooks
        for i in target_layers:
            if i < len(self.model.aggregator.frame_blocks):
                hook_name = f'frame_block_{i}'
                hook = self.model.aggregator.frame_blocks[i].register_forward_hook(create_hook(hook_name))
                self.attention_hooks.append(hook)
            
            if i < len(self.model.aggregator.global_blocks):
                hook_name = f'global_block_{i}'
                hook = self.model.aggregator.global_blocks[i].register_forward_hook(create_hook(hook_name))
                self.attention_hooks.append(hook)
    
    def load_images(self, image_paths: List[str]):
        """加载图像并运行模型"""
        print(f"📸 Loading {len(image_paths)} images...")
        
        # 加载和预处理图像
        self.current_images = load_and_preprocess_images(image_paths).to(self.device)
        
        # 设置attention hooks
        self.setup_attention_hooks()
        
        # 运行模型
        print("🔄 Running model to capture attention...")
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    self.attention_maps.clear()
                    images_subset = self.current_images[:6] if self.current_images.shape[0] > 6 else self.current_images
                    _ = self.model(images_subset)
                    torch.cuda.empty_cache()
            else:
                print('processing on CPU...')
                self.attention_maps.clear()
                images_subset = self.current_images[:6] if self.current_images.shape[0] > 6 else self.current_images
                _ = self.model(images_subset)
        
        self.current_attention_maps = self.attention_maps.copy()
        
        # 创建图像网格
        self.create_image_grid(images_subset)
        
        print(f"✅ 成功处理 {len(image_paths)} 张图像，捕获了 {len(self.attention_maps)} 个attention层")
    
    def create_image_grid(self, images: torch.Tensor):
        """创建图像网格用于显示"""
        S, C, H, W = images.shape
        
        # 计算网格尺寸
        cols = min(3, S)
        rows = (S + cols - 1) // cols
        
        # 存储网格配置信息
        self.grid_config = {
            'num_images': S,
            'cols': cols,
            'rows': rows,
            'image_height': H,
            'image_width': W,
            'patch_size': self.patch_size
        }
        
        # 创建网格图像
        grid_h = rows * H
        grid_w = cols * W
        grid_array = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        
        for i in range(S):
            row, col = i // cols, i % cols
            img = images[i].permute(1, 2, 0).cpu().numpy()
            
            # 转换到0-255范围
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
            
            start_h, end_h = row * H, (row + 1) * H
            start_w, end_w = col * W, (col + 1) * W
            
            # OpenCV使用BGR格式
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            grid_array[start_h:end_h, start_w:end_w] = img_bgr
        
        self.grid_image = grid_array
        self.display_image = self.grid_image.copy()
        
        print(f"🖼️ 创建网格图像: {grid_array.shape}")
    
    def get_coordinate_info(self, x: int, y: int) -> Dict:
        """根据坐标获取详细信息"""
        if self.grid_config is None:
            return None
        
        config = self.grid_config
        
        # 确定点击的是哪个图像
        col = x // config['image_width']
        row = y // config['image_height']
        img_idx = row * config['cols'] + col
        
        if img_idx >= config['num_images'] or col >= config['cols'] or row >= config['rows']:
            return None
        
        # 计算在单个图像内的相对坐标
        local_x = x % config['image_width']
        local_y = y % config['image_height']
        
        # 计算patch坐标
        patch_x = local_x // config['patch_size']
        patch_y = local_y // config['patch_size']
        
        patch_h = config['image_height'] // config['patch_size']
        patch_w = config['image_width'] // config['patch_size']
        
        if patch_x >= patch_w or patch_y >= patch_h:
            return None
        
        return {
            'global_coords': (x, y),
            'image_idx': img_idx,
            'local_coords': (local_x, local_y),
            'patch_coords': (patch_x, patch_y),
            'patch_grid_size': (patch_h, patch_w),
            'valid': True
        }
    
    def extract_patch_attention(self, img_idx: int, patch_y: int, patch_x: int, layer_name: str = 'global_block_23') -> Dict:
        """提取指定patch的attention maps"""
        return extract_single_patch_attention_simple(
            self.current_attention_maps, self.current_images, 
            img_idx, patch_y, patch_x, layer_name
        )
    
    def visualize_patch_attention(self, patch_attention_data: Dict) -> np.ndarray:
        """可视化patch attention结果"""
        if patch_attention_data is None:
            return None
        
        source_img_idx = patch_attention_data['source_img_idx']
        source_patch = patch_attention_data['source_patch']
        attention_maps = patch_attention_data['attention_maps']
        layer_name = patch_attention_data['layer_name']
        
        S, C, H, W = self.current_images.shape
        
        # 创建可视化
        fig, axes = plt.subplots(2, min(3, S), figsize=(min(3, S) * 4, 8))
        if S <= 3:
            axes = axes.reshape(2, -1)
        
        for i in range(min(S, axes.shape[1])):
            # 上方显示原图 + source patch标记
            ax_img = axes[0, i]
            img = self.current_images[i].permute(1, 2, 0).cpu().numpy()
            img = np.clip(img, 0, 1)
            ax_img.imshow(img)
            
            # 如果是source图像，标记source patch
            if i == source_img_idx:
                patch_y, patch_x = source_patch
                rect = patches.Rectangle(
                    (patch_x * self.patch_size, patch_y * self.patch_size),
                    self.patch_size, self.patch_size,
                    linewidth=3, edgecolor='red', facecolor='none'
                )
                ax_img.add_patch(rect)
                ax_img.set_title(f'Source Image {i}\nPatch ({patch_y}, {patch_x})', color='red', fontweight='bold')
            else:
                ax_img.set_title(f'Target Image {i}')
            
            ax_img.axis('off')
            
            # 下方显示attention热图
            ax_attn = axes[1, i]
            attention_map = attention_maps[i]
            
            # 上采样到原图尺寸
            attention_upsampled = torch.nn.functional.interpolate(
                attention_map.unsqueeze(0).unsqueeze(0),
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze()
            
            # 显示原图作为背景
            ax_attn.imshow(img, alpha=0.7)
            
            # 叠加attention热图
            im = ax_attn.imshow(attention_upsampled.numpy(), alpha=0.8, cmap='hot')
            ax_attn.set_title(f'Attention to Image {i}\nMax: {attention_map.max():.4f}')
            ax_attn.axis('off')
            
            # 添加colorbar
            plt.colorbar(im, ax=ax_attn, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Patch Attention Visualization - {layer_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 转换为OpenCV图像
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # 读取为OpenCV格式
        img_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        return img
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        self.mouse_pos = (x, y)
        
        # 更新显示图像
        self.display_image = self.grid_image.copy()
        
        # 获取坐标信息
        coord_info = self.get_coordinate_info(x, y)
        
        if coord_info and coord_info['valid']:
            # 绘制十字线
            cv2.line(self.display_image, (x-20, y), (x+20, y), (0, 255, 0), 1)
            cv2.line(self.display_image, (x, y-20), (x, y+20), (0, 255, 0), 1)
            
            # 绘制patch边界
            config = self.grid_config
            patch_x, patch_y = coord_info['patch_coords']
            img_idx = coord_info['image_idx']
            
            # 计算patch在网格中的绝对位置
            row = img_idx // config['cols']
            col = img_idx % config['cols']
            
            patch_start_x = col * config['image_width'] + patch_x * config['patch_size']
            patch_start_y = row * config['image_height'] + patch_y * config['patch_size']
            patch_end_x = patch_start_x + config['patch_size']
            patch_end_y = patch_start_y + config['patch_size']
            
            # 绘制patch矩形
            cv2.rectangle(self.display_image, 
                         (patch_start_x, patch_start_y), 
                         (patch_end_x, patch_end_y), 
                         (255, 0, 0), 2)
            
            # 显示坐标信息
            info_text = [
                f"Global: ({x}, {y})",
                f"Image: {img_idx} Local: {coord_info['local_coords']}",
                f"Patch: ({patch_x}, {patch_y})",
                "Click to analyze attention"
            ]
            
            for i, text in enumerate(info_text):
                cv2.putText(self.display_image, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(self.display_image, text, (10, 30 + i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # 鼠标点击事件
        if event == cv2.EVENT_LBUTTONDOWN:
            if coord_info and coord_info['valid']:
                print(f"🎯 点击坐标: ({x}, {y})")
                print(f"🎯 图像索引: {coord_info['image_idx']}")
                print(f"🎯 Patch坐标: {coord_info['patch_coords']}")
                
                # 提取并显示attention
                self.show_attention_for_click(coord_info)
        
        # 更新显示
        cv2.imshow(self.window_name, self.display_image)
    
    def show_attention_for_click(self, coord_info: Dict):
        """显示点击位置的attention结果"""
        img_idx = coord_info['image_idx']
        patch_x, patch_y = coord_info['patch_coords']
        
        print(f"🔄 正在提取图像 {img_idx} 中patch ({patch_y}, {patch_x}) 的attention...")
        
        # 提取patch attention
        patch_attention_data = self.extract_patch_attention(img_idx, patch_y, patch_x)
        
        if patch_attention_data is None:
            print("❌ 无法提取attention数据")
            return
        
        # 生成可视化结果
        result_image = self.visualize_patch_attention(patch_attention_data)
        
        if result_image is not None:
            # 显示结果
            result_window = "Attention Visualization Result"
            cv2.namedWindow(result_window, cv2.WINDOW_NORMAL)
            cv2.imshow(result_window, result_image)
            
            print(f"✅ 成功显示图像 {img_idx} 中patch ({patch_y}, {patch_x}) 的attention")
            print(f"层: {patch_attention_data['layer_name']}")
            print("💡 按任意键关闭attention结果窗口")
    
    def run(self, image_paths: List[str]):
        """运行交互式可视化"""
        try:
            # 加载图像
            self.load_images(image_paths)
            
            # 创建OpenCV窗口
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            
            print("\n🎯 交互式VGGT Attention可视化器已启动！")
            print("📋 操作说明：")
            print("  🖱️  移动鼠标查看实时坐标和patch信息")
            print("  🎯  点击任意位置分析该patch的attention")
            print("  ⌨️  按ESC键退出程序")
            print("  📊  attention结果将在新窗口中显示")
            print(f"📍 网格配置: {self.grid_config['num_images']}张图像, {self.grid_config['rows']}行{self.grid_config['cols']}列")
            
            # 显示初始图像
            cv2.imshow(self.window_name, self.grid_image)
            
            # 主循环
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC键退出
                    break
                elif key == ord('s'):  # S键保存当前显示
                    cv2.imwrite("current_display.png", self.display_image)
                    print("💾 已保存当前显示为 current_display.png")
            
            print("👋 程序已退出")
            
        except Exception as e:
            print(f"❌ 运行出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()

def main():
    """主函数"""
    print("🚀 启动基于OpenCV的VGGT交互式Attention可视化器")
    
    # 示例图像路径 - 请根据需要修改
    image_paths = [
        "temp_images/image_000.jpg",
        "temp_images/image_001.jpg"
    ]
    
    # 检查图像文件是否存在
    existing_paths = [path for path in image_paths if os.path.exists(path)]
    
    if len(existing_paths) < 2:
        print("❌ 需要至少2张图像文件")
        print("💡 请将图像文件放在temp_images目录中，或修改image_paths变量")
        
        # 尝试使用extracted_frames目录中的图像
        extracted_dir = "extracted_frames"
        if os.path.exists(extracted_dir):
            frame_files = sorted([f for f in os.listdir(extracted_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            if len(frame_files) >= 2:
                existing_paths = [os.path.join(extracted_dir, f) for f in frame_files[:6]]
                print(f"🔄 使用extracted_frames中的图像: {len(existing_paths)}张")
            else:
                return
        else:
            return
    
    print(f"📸 使用图像文件: {existing_paths}")
    
    # 创建可视化器并运行
    visualizer = OpenCVAttentionVisualizer()
    visualizer.run(existing_paths)

if __name__ == "__main__":
    main()
