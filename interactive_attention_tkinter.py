#!/usr/bin/env python3
"""
基于Tkinter的VGGT交互式Attention可视化工具
支持真正的鼠标点击获取坐标并显示attention结果
避免OpenCV段错误问题
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import io
from typing import Dict, List, Tuple, Optional
import threading

# 导入VGGT相关模块（使用可视化版本）
from vggt.models.vggt_vis import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

# 导入现有的attention工具（用于兼容性）
from attn_util import pixel_to_patch_coords

class TkinterAttentionVisualizer:
    def __init__(self):
        # 设备配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        else:
            self.dtype = torch.float32
        
        # 存储当前状态
        self.current_images = None
        self.patch_size = 14
        self.grid_config = None
        
        # 🔥 移除hook相关变量
        # self.current_attention_maps = None  # 现在通过模型缓存获取
        # self.attention_hooks = []           # 不再需要hooks
        
        # GUI相关
        self.root = None
        self.canvas = None
        self.grid_image_pil = None
        self.grid_image_tk = None
        self.info_label = None
        self.result_window = None
        
        # 显示缩放信息
        self.display_scale_x = 1.0
        self.display_scale_y = 1.0
        
        # 初始化模型（延迟加载）
        self.model = None
        self.model_loaded = False
        
        print("🚀 初始化Tkinter可视化器...")
    
    def __del__(self):
        """析构函数，清理资源"""
        if hasattr(self, 'model') and self.model:
            self.model.disable_attention_caching()
            self.model.clear_attention_cache()
    
    def load_model(self):
        """延迟加载VGGT模型"""
        if not self.model_loaded:
            print("🚀 Loading VGGT model...")
            self.model = VGGT.from_pretrained("VGGT-1B").to(self.device)
            self.model.eval()
            self.model_loaded = True
            print("✅ Model loaded successfully!")
    
    # 🔥 移除整个setup_attention_hooks方法，不再需要！
    # 现在直接使用模型内置的attention缓存系统
    
    def load_images(self, image_paths: List[str]):
        """加载图像并运行模型"""
        try:
            # 确保模型已加载
            self.load_model()
            
            print(f"📸 Loading {len(image_paths)} images...")
            
            # 加载和预处理图像
            self.current_images = load_and_preprocess_images(image_paths).to(self.device)
            print('loaded images ')
            # 🔥 新的缓存系统：启用attention缓存
            target_layers = ['global_block_20', 'global_block_21', 'global_block_22', 'global_block_23']
            self.model.enable_attention_caching(target_layers)
            self.model.start_new_attention_run()
            
            # 运行模型（只需一次！）
            print("🔄 Running model with attention caching...")
            with torch.no_grad():
                if torch.cuda.is_available():
                    with torch.amp.autocast('cuda', dtype=self.dtype):
                        images_subset = self.current_images[:6] if self.current_images.shape[0] > 6 else self.current_images
                        _ = self.model(images_subset)
                        torch.cuda.empty_cache()
                else:
                    print('processing on CPU...')
                    images_subset = self.current_images[:6] if self.current_images.shape[0] > 6 else self.current_images
                    _ = self.model(images_subset)
            
            # 创建图像网格
            self.create_image_grid(images_subset)
            
            # 获取缓存信息用于验证
            cache_info = self.model.get_attention_cache_info()
            cached_layers = cache_info.get('cached_layers', [])
            
            print(f"✅ 成功处理 {len(image_paths)} 张图像，缓存了 {len(cached_layers)} 个attention层: {cached_layers}")
            return True
            
        except Exception as e:
            print(f"❌ 加载图像时出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 清理可能的缓存
            if hasattr(self, 'model') and self.model:
                self.model.clear_attention_cache()
            
            return False
    
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
            
            grid_array[start_h:end_h, start_w:end_w] = img
        
        # 转换为PIL图像
        self.grid_image_pil = Image.fromarray(grid_array)
        
        # 更新Canvas显示
        if self.canvas:
            self.update_canvas_display()
        
        print(f"🖼️ 创建网格图像: {grid_array.shape}")
    
    def get_coordinate_info(self, x: int, y: int) -> Dict:
        """根据坐标获取详细信息（修复了Canvas缩放问题）"""
        if self.grid_config is None:
            return None
        
        config = self.grid_config
        
        # 🔥 关键修复：将显示坐标映射回原始图像坐标
        original_x = x / self.display_scale_x
        original_y = y / self.display_scale_y
        
        # 确定点击的是哪个图像
        col = int(original_x // config['image_width'])
        row = int(original_y // config['image_height'])
        img_idx = row * config['cols'] + col
        
        if img_idx >= config['num_images'] or col >= config['cols'] or row >= config['rows']:
            return None
        
        # 计算在单个图像内的相对坐标
        local_x = original_x % config['image_width']
        local_y = original_y % config['image_height']
        
        # 计算patch坐标
        patch_x = int(local_x // config['patch_size'])
        patch_y = int(local_y // config['patch_size'])
        
        patch_h = config['image_height'] // config['patch_size']
        patch_w = config['image_width'] // config['patch_size']
        
        if patch_x >= patch_w or patch_y >= patch_h:
            return None
        
        return {
            'global_coords': (x, y),
            'original_coords': (original_x, original_y),
            'image_idx': img_idx,
            'local_coords': (local_x, local_y),
            'patch_coords': (patch_x, patch_y),
            'patch_grid_size': (patch_h, patch_w),
            'valid': True
        }
    
    def extract_patch_attention(self, img_idx: int, patch_y: int, patch_x: int, layer_name: str = 'global_block_23') -> Dict:
        """提取指定patch的attention maps（使用新的缓存系统）"""
        # 🔥 使用模型的缓存系统替代hook方式
        return self.model.extract_patch_attention(img_idx, patch_y, patch_x, layer_name)
    
    def visualize_patch_attention(self, patch_attention_data: Dict) -> Image.Image:
        """可视化patch attention结果"""
        if patch_attention_data is None:
            return None
        
        source_img_idx = patch_attention_data['source_img_idx']
        source_patch = patch_attention_data['source_patch']
        attention_maps = patch_attention_data['attention_maps']
        layer_name = patch_attention_data['layer_name']
        
        S, C, H, W = self.current_images.shape
        num_images = min(S, len(attention_maps))
        
        # 根据图像数量调整布局
        if num_images <= 3:
            cols = num_images
            fig_width = cols * 4
            fig_height = 8
        else:
            # 对于4-6张图像，使用3列2行布局
            cols = 3
            fig_width = 12
            fig_height = 10
        
        # 创建足够的子图：2行显示原图和attention
        rows = 2
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        
        # 确保axes是2D数组
        if num_images == 1:
            axes = axes.reshape(2, 1)
        elif cols == 1:
            axes = axes.reshape(2, 1)
        
        # 显示前3张图像（第一组）
        for i in range(min(3, num_images)):
            # 上方显示原图 + source patch标记
            ax_img = axes[0, i] if cols > 1 else axes[0]
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
            ax_attn = axes[1, i] if cols > 1 else axes[1]
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
        
        # 隐藏未使用的子图（如果图像数量少于3）
        for col_idx in range(min(3, num_images), cols):
            if cols > 1:
                axes[0, col_idx].set_visible(False)
                axes[1, col_idx].set_visible(False)
        
        plt.suptitle(f'Patch Attention Visualization - {layer_name} (Images 0-{min(2, num_images-1)})', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 如果有超过3张图像，需要创建第二个图来显示剩余的图像
        if num_images > 3:
            # 保存第一个图
            buf1 = io.BytesIO()
            plt.savefig(buf1, format='png', dpi=150, bbox_inches='tight')
            buf1.seek(0)
            first_image = Image.open(buf1)
            plt.close()
            
            # 创建第二个图显示图像3-5
            remaining_images = num_images - 3
            cols2 = min(3, remaining_images)
            fig2, axes2 = plt.subplots(2, cols2, figsize=(cols2 * 4, 8))
            
            if remaining_images == 1:
                axes2 = axes2.reshape(2, 1)
            elif cols2 == 1:
                axes2 = axes2.reshape(2, 1)
            
            for i in range(3, num_images):
                col_idx = i - 3
                
                # 上方显示原图 + source patch标记
                ax_img = axes2[0, col_idx] if cols2 > 1 else axes2[0]
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
                ax_attn = axes2[1, col_idx] if cols2 > 1 else axes2[1]
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
            
            # 隐藏未使用的子图
            for col_idx in range(remaining_images, cols2):
                if cols2 > 1:
                    axes2[0, col_idx].set_visible(False)
                    axes2[1, col_idx].set_visible(False)
            
            plt.suptitle(f'Patch Attention Visualization - {layer_name} (Images 3-{num_images-1})', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # 保存第二个图
            buf2 = io.BytesIO()
            plt.savefig(buf2, format='png', dpi=150, bbox_inches='tight')
            buf2.seek(0)
            second_image = Image.open(buf2)
            plt.close()
            
            # 将两个图像垂直拼接
            total_width = max(first_image.width, second_image.width)
            total_height = first_image.height + second_image.height
            
            combined_image = Image.new('RGB', (total_width, total_height), 'white')
            combined_image.paste(first_image, (0, 0))
            combined_image.paste(second_image, (0, first_image.height))
            
            return combined_image
        else:
            # 只有3张或更少图像，直接返回
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            return Image.open(buf)
    
    def on_canvas_click(self, event):
        """Canvas点击事件处理"""
        if self.grid_config is None:
            return
        
        x, y = event.x, event.y
        
        # 获取坐标信息
        coord_info = self.get_coordinate_info(x, y)
        
        if coord_info and coord_info['valid']:
            print(f"🎯 点击坐标: ({x}, {y})")
            print(f"🎯 图像索引: {coord_info['image_idx']}")
            print(f"🎯 Patch坐标: {coord_info['patch_coords']}")
            
            # 更新信息显示
            info_text = f"点击坐标: ({x}, {y}) | 图像: {coord_info['image_idx']} | Patch: {coord_info['patch_coords']} | 正在分析attention..."
            self.info_label.config(text=info_text)
            
            # 在新线程中处理attention分析，避免界面卡死
            threading.Thread(target=self.show_attention_for_click, args=(coord_info,), daemon=True).start()
    
    def on_canvas_motion(self, event):
        """Canvas鼠标移动事件处理"""
        if self.grid_config is None:
            return
        
        x, y = event.x, event.y
        
        # 获取坐标信息
        coord_info = self.get_coordinate_info(x, y)
        
        if coord_info and coord_info['valid']:
            orig_x, orig_y = coord_info['original_coords']
            info_text = f"显示坐标: ({x}, {y}) | 原始坐标: ({orig_x:.1f}, {orig_y:.1f}) | 图像: {coord_info['image_idx']} | Patch: {coord_info['patch_coords']} | 点击分析attention"
        else:
            info_text = f"坐标: ({x}, {y}) | 无效区域"
        
        self.info_label.config(text=info_text)
    
    def show_attention_for_click(self, coord_info: Dict):
        """显示点击位置的attention结果"""
        img_idx = coord_info['image_idx']
        patch_x, patch_y = coord_info['patch_coords']
        
        try:
            print(f"🔄 正在提取图像 {img_idx} 中patch ({patch_y}, {patch_x}) 的attention...")
            
            # 提取patch attention
            patch_attention_data = self.extract_patch_attention(img_idx, patch_y, patch_x)
            
            if patch_attention_data is None:
                print("❌ 无法提取attention数据")
                self.root.after(0, lambda: self.info_label.config(text="❌ 无法提取attention数据"))
                return
            
            # 在主线程中生成可视化结果和显示
            self.root.after(0, self._create_and_show_result, patch_attention_data, coord_info)
            
        except Exception as e:
            print(f"❌ 分析attention时出错: {e}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.info_label.config(text=f"❌ 分析attention时出错: {e}"))
    
    def _create_and_show_result(self, patch_attention_data: Dict, coord_info: Dict):
        """在主线程中创建可视化结果并显示"""
        try:
            # 生成可视化结果（现在在主线程中）
            result_image = self.visualize_patch_attention(patch_attention_data)
            
            if result_image is not None:
                self.show_result_window(result_image, coord_info, patch_attention_data)
                
                img_idx = coord_info['image_idx']
                patch_x, patch_y = coord_info['patch_coords']
                print(f"✅ 成功显示图像 {img_idx} 中patch ({patch_y}, {patch_x}) 的attention")
                print(f"层: {patch_attention_data['layer_name']}")
            else:
                self.info_label.config(text="❌ 生成可视化结果失败")
                
        except Exception as e:
            print(f"❌ 创建可视化结果时出错: {e}")
            import traceback
            traceback.print_exc()
            self.info_label.config(text=f"❌ 创建可视化结果时出错: {e}")
    
    def show_result_window(self, result_image: Image.Image, coord_info: Dict, patch_attention_data: Dict):
        """显示结果窗口"""
        # 关闭之前的结果窗口
        if self.result_window:
            self.result_window.destroy()
        
        # 创建新的结果窗口
        self.result_window = tk.Toplevel(self.root)
        self.result_window.title("Attention可视化结果")
        self.result_window.geometry("800x600")
        
        # 创建滚动框架
        canvas = tk.Canvas(self.result_window)
        scrollbar = ttk.Scrollbar(self.result_window, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # 显示信息
        img_idx = coord_info['image_idx']
        patch_x, patch_y = coord_info['patch_coords']
        
        info_text = f"""
✅ 成功分析图像 {img_idx} 中patch ({patch_y}, {patch_x}) 的attention
全局坐标: {coord_info['global_coords']}
局部坐标: {coord_info['local_coords']}
层: {patch_attention_data['layer_name']}
        """
        
        info_label = ttk.Label(scrollable_frame, text=info_text, font=("Arial", 12))
        info_label.pack(pady=10)
        
        # 显示结果图像
        # 调整图像大小以适应窗口
        max_width, max_height = 700, 500
        result_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        
        result_photo = ImageTk.PhotoImage(result_image)
        result_label = ttk.Label(scrollable_frame, image=result_photo)
        result_label.image = result_photo  # 保持引用
        result_label.pack(pady=10)
        
        # 添加保存按钮
        def save_result():
            filename = f"attention_result_img{img_idx}_patch{patch_y}_{patch_x}.png"
            result_image.save(filename)
            messagebox.showinfo("保存成功", f"结果已保存为 {filename}")
        
        save_btn = ttk.Button(scrollable_frame, text="保存结果", command=save_result)
        save_btn.pack(pady=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # 更新主窗口信息
        self.info_label.config(text=f"✅ 已显示图像 {img_idx} 中patch ({patch_y}, {patch_x}) 的attention结果")
    
    def update_canvas_display(self):
        """更新Canvas显示（记录缩放信息）"""
        if self.grid_image_pil and self.canvas:
            # 调整图像大小以适应canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            
            if canvas_width > 1 and canvas_height > 1:  # 确保canvas已经初始化
                # 保持宽高比缩放
                img_ratio = self.grid_image_pil.width / self.grid_image_pil.height
                canvas_ratio = canvas_width / canvas_height
                
                if img_ratio > canvas_ratio:
                    # 图像更宽，以宽度为准
                    new_width = canvas_width
                    new_height = int(canvas_width / img_ratio)
                else:
                    # 图像更高，以高度为准
                    new_height = canvas_height
                    new_width = int(canvas_height * img_ratio)
                
                # 🔥 记录缩放比例
                self.display_scale_x = new_width / self.grid_image_pil.width
                self.display_scale_y = new_height / self.grid_image_pil.height
                
                print(f"🔍 Canvas缩放信息: 原始({self.grid_image_pil.width}x{self.grid_image_pil.height}) → 显示({new_width}x{new_height})")
                print(f"🔍 缩放比例: x={self.display_scale_x:.4f}, y={self.display_scale_y:.4f}")
                
                # 调整图像大小
                resized_image = self.grid_image_pil.resize((new_width, new_height), Image.Resampling.LANCZOS)
                self.grid_image_tk = ImageTk.PhotoImage(resized_image)
                
                # 清除canvas并显示图像
                self.canvas.delete("all")
                self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.grid_image_tk)
    
    def load_images_dialog(self):
        """打开文件对话框选择图像"""
        file_paths = filedialog.askopenfilenames(
            title="选择图像文件",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("所有文件", "*.*")
            ]
        )
        
        if len(file_paths) < 2:
            messagebox.showwarning("警告", "请至少选择2张图像")
            return
        
        if len(file_paths) > 6:
            messagebox.showinfo("提示", "最多支持6张图像，将使用前6张")
            file_paths = file_paths[:6]
        
        # 在新线程中加载图像，避免界面卡死
        self.info_label.config(text="正在加载图像...")
        threading.Thread(target=self.load_images_thread, args=(file_paths,), daemon=True).start()
    
    def load_images_thread(self, file_paths):
        """在线程中加载图像"""
        success = self.load_images(file_paths)
        
        if success:
            self.root.after(0, lambda: self.info_label.config(text=f"✅ 成功加载 {len(file_paths)} 张图像，点击图像查看attention"))
        else:
            self.root.after(0, lambda: self.info_label.config(text="❌ 加载图像失败"))
    
    def create_gui(self):
        """创建GUI界面"""
        self.root = tk.Tk()
        self.root.title("VGGT Interactive Attention Visualizer")
        self.root.geometry("1000x700")
        
        # 创建主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # 顶部控制面板
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(control_frame, text="🔍 VGGT Interactive Attention Visualizer", 
                 font=("Arial", 16, "bold")).pack(side="left")
        
        load_btn = ttk.Button(control_frame, text="📁 加载图像", command=self.load_images_dialog)
        load_btn.pack(side="right", padx=(10, 0))
        
        # 信息显示区域
        self.info_label = ttk.Label(main_frame, text="请点击'加载图像'按钮选择2-6张图像文件", 
                                   font=("Arial", 12))
        self.info_label.pack(fill="x", pady=(0, 10))
        
        # 图像显示区域
        canvas_frame = ttk.Frame(main_frame)
        canvas_frame.pack(fill="both", expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg="white", cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)
        
        # 绑定事件
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Configure>", lambda e: self.update_canvas_display())
        
        # 底部说明
        help_text = """
🎯 操作说明：
• 移动鼠标查看实时坐标和patch信息
• 点击任意位置分析该patch的attention（结果将在新窗口显示）
• 支持2-6张图像的多视图attention分析
        """
        
        help_label = ttk.Label(main_frame, text=help_text, font=("Arial", 10), 
                              foreground="gray")
        help_label.pack(fill="x", pady=(10, 0))
        
        # 检查是否有现成的图像可以加载
        self.check_existing_images()

    def check_existing_images(self):
        """检查是否有现成的图像可以自动加载"""
        # 检查temp_images目录
        temp_dir = "temp_images"
        if os.path.exists(temp_dir):
            temp_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if len(temp_files) >= 2:
                self.info_label.config(text=f"发现temp_images中有{len(temp_files)}张图像，正在自动加载...")
                threading.Thread(target=self.load_images_thread, args=(temp_files[:6],), daemon=True).start()
                return
        
        # 检查extracted_frames目录
        extracted_dir = "extracted_frames"
        if os.path.exists(extracted_dir):
            frame_files = sorted([os.path.join(extracted_dir, f) for f in os.listdir(extracted_dir) 
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            if len(frame_files) >= 2:
                self.info_label.config(text=f"发现extracted_frames中有{len(frame_files)}张图像，正在自动加载...")
                threading.Thread(target=self.load_images_thread, args=(frame_files[:6],), daemon=True).start()
                return
    
    def run(self):
        """运行GUI"""
        try:
            print("🚀 启动基于Tkinter的VGGT交互式Attention可视化器")
            self.create_gui()
            self.root.mainloop()
        except Exception as e:
            print(f"❌ GUI运行出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 确保清理资源
            if hasattr(self, 'model') and self.model:
                self.model.disable_attention_caching()
                self.model.clear_attention_cache()
                print("🗑️ 已清理attention缓存资源")

def main():
    """主函数"""
    visualizer = TkinterAttentionVisualizer()
    visualizer.run()

if __name__ == "__main__":
    main()
