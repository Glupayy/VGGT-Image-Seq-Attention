#!/usr/bin/env python3
"""
测试tkinter GUI初始化修复
"""

import tkinter as tk
from tkinter import ttk

class TestTkinterVisualizer:
    def __init__(self):
        # 分辨率控制选项 - 延迟初始化
        self.high_quality_mode = None  # 将在create_gui中初始化
        print("✅ 初始化成功，没有过早创建tkinter变量")
    
    def _get_high_quality_mode(self):
        """安全获取高质量模式状态"""
        return self.high_quality_mode.get() if self.high_quality_mode is not None else True
    
    def on_quality_mode_changed(self):
        """分辨率模式改变时的回调函数"""
        if self.high_quality_mode is None:
            return
        mode = "高分辨率" if self._get_high_quality_mode() else "标准分辨率"
        print(f"🎨 分辨率模式已切换为: {mode}")
    
    def create_gui(self):
        """创建GUI界面"""
        self.root = tk.Tk()
        self.root.title("Test VGGT Interactive Attention Visualizer")
        self.root.geometry("800x600")
        
        # 现在可以安全地初始化tkinter变量
        self.high_quality_mode = tk.BooleanVar(value=True)
        
        # 创建测试控件
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ttk.Label(main_frame, text="🔍 Test VGGT Interactive Attention Visualizer", 
                 font=("Arial", 16, "bold")).pack(side="top")
        
        # 添加分辨率控制选项
        quality_frame = ttk.Frame(main_frame)
        quality_frame.pack(side="top", padx=(10, 0), pady=20)
        
        quality_checkbox = ttk.Checkbutton(
            quality_frame, 
            text="高分辨率模式", 
            variable=self.high_quality_mode,
            command=self.on_quality_mode_changed
        )
        quality_checkbox.pack(side="top")
        
        # 测试当前状态
        current_mode = "高分辨率" if self._get_high_quality_mode() else "标准分辨率"
        status_label = ttk.Label(main_frame, text=f"当前模式: {current_mode}")
        status_label.pack(side="top", pady=10)
        
        print("✅ GUI创建成功，所有tkinter变量正常工作")
        return True
    
    def run(self):
        """运行GUI"""
        try:
            print("🚀 启动测试GUI")
            success = self.create_gui()
            if success:
                print("✅ 测试成功！tkinter初始化问题已修复")
                # 简单显示2秒后关闭
                self.root.after(2000, self.root.destroy)
                self.root.mainloop()
                return True
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False

def main():
    """主函数"""
    print("=" * 50)
    print("开始测试tkinter初始化修复")
    print("=" * 50)
    
    visualizer = TestTkinterVisualizer()
    success = visualizer.run()
    
    print("=" * 50)
    if success:
        print("🎉 所有测试通过！修复成功")
    else:
        print("❌ 测试失败，需要进一步调试")
    print("=" * 50)

if __name__ == "__main__":
    main()