import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
from PIL import Image, ImageTk

# 将项目根目录加入模块路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import load_trained_model, predict_single_image
from config import CLASS_NAMES_CN

class LogoPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🧋 奶茶品牌 LOGO 智能双模型对比系统")
        self.root.geometry("850x800")
        self.root.configure(bg="#f4f4f9")
        
        # 加载两个模型
        try:
            self.model_cnn = load_trained_model("simple_cnn")
            self.model_resnet = load_trained_model("resnet50")
        except Exception as e:
            messagebox.showerror("模型加载失败", f"无法加载模型，请确认是否已完成训练。\n错误信息: {e}")
            self.root.destroy()
            return

        self.setup_ui()
        
        # 绑定全局拖拽事件
        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind('<<Drop>>', self.handle_drop)
        
    def setup_ui(self):
        # 标题区域
        title_frame = tk.Frame(self.root, bg="#f4f4f9")
        title_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(
            title_frame, text="奶茶品牌智能识别 - 双模型实时对比", 
            font=("Microsoft YaHei", 24, "bold"), bg="#f4f4f9", fg="#2c3e50"
        ).pack()
        
        tk.Label(
            title_frame, text="支持直接【拖拽图片】到窗口任意位置，或者点击下方按钮选择", 
            font=("Microsoft YaHei", 12), bg="#f4f4f9", fg="#7f8c8d"
        ).pack(pady=5)
        
        self.btn_select = tk.Button(
            title_frame, text="📁 浏览本地图片", 
            font=("Microsoft YaHei", 12), bg="#3498db", fg="white", 
            relief="flat", cursor="hand2", padx=20, pady=5, 
            command=self.load_image
        )
        self.btn_select.pack(pady=5)
        
        # 图片显示区域
        self.canvas = tk.Canvas(
            self.root, width=400, height=400, 
            bg="white", relief="solid", borderwidth=1, highlightthickness=0
        )
        self.canvas.pack(pady=10)
        self.image_on_canvas = None
        self.canvas.create_text(200, 200, text="将图片拖拽至此处\n\n(Drag & Drop)", 
                              font=("Microsoft YaHei", 14), fill="#bdc3c7", justify=tk.CENTER)
        
        # 结果对比区域 (左右分栏)
        result_frame = tk.Frame(self.root, bg="#f4f4f9")
        result_frame.pack(pady=10, fill=tk.BOTH, expand=True, padx=20)
        
        # 左侧：SimpleCNN
        self.frame_cnn = tk.LabelFrame(result_frame, text=" Baseline: SimpleCNN (从零训练) ", 
                                     font=("Microsoft YaHei", 14, "bold"), bg="white", fg="#7f8c8d", padx=10, pady=10)
        self.frame_cnn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        self.cnn_res_label = tk.Label(self.frame_cnn, text="等待预测...", font=("Microsoft YaHei", 16, "bold"), bg="white", fg="#95a5a6")
        self.cnn_res_label.pack(pady=10)
        
        self.cnn_detail_label = tk.Label(self.frame_cnn, text="", font=("Microsoft YaHei", 12), bg="white", fg="#34495e", justify=tk.LEFT)
        self.cnn_detail_label.pack(pady=5)

        # 右侧：ResNet50
        self.frame_resnet = tk.LabelFrame(result_frame, text=" 🌟 主力模型: ResNet50 (迁移学习) ", 
                                        font=("Microsoft YaHei", 14, "bold"), bg="white", fg="#2980b9", padx=10, pady=10)
        self.frame_resnet.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)
        
        self.resnet_res_label = tk.Label(self.frame_resnet, text="等待预测...", font=("Microsoft YaHei", 16, "bold"), bg="white", fg="#95a5a6")
        self.resnet_res_label.pack(pady=10)
        
        self.resnet_detail_label = tk.Label(self.frame_resnet, text="", font=("Microsoft YaHei", 12), bg="white", fg="#34495e", justify=tk.LEFT)
        self.resnet_detail_label.pack(pady=5)

    def handle_drop(self, event):
        file_path = event.data
        # tkinterdnd2 会在路径包含空格时返回带大括号的花括号字符串 {C:/my path/image.jpg}
        if file_path.startswith('{') and file_path.endswith('}'):
            file_path = file_path[1:-1]
            
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp')):
            self.display_image(file_path)
            self.predict(file_path)
        else:
            messagebox.showwarning("格式不支持", "请拖入支持的图片格式 (jpg/png/webp/bmp)！")

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="选择要识别的图片",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp")]
        )
        if file_path:
            self.display_image(file_path)
            self.predict(file_path)
            
    def display_image(self, file_path):
        img = Image.open(file_path)
        img.thumbnail((400, 400), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        
        self.canvas.delete("all")
        x = (400 - img.width) // 2
        y = (400 - img.height) // 2
        self.image_on_canvas = self.canvas.create_image(x, y, anchor="nw", image=self.tk_image)
        
    def _update_result(self, label_title, label_detail, pred_class, probs):
        confidence = max(probs)
        if confidence > 0.8:
            color = "#27ae60" # 绿色
        elif confidence > 0.5:
            color = "#f39c12" # 橙色
        else:
            color = "#c0392b" # 红色
            
        label_title.config(text=f"预测: 【{pred_class}】 ({confidence:.1%})", fg=color)
        
        details = ""
        for name, prob in zip(CLASS_NAMES_CN, probs):
            # 找出最大概率，加粗或加箭头标记
            mark = " 👈" if prob == confidence else ""
            details += f"{name:　<5}: {prob:>6.1%}{mark}\n"
        label_detail.config(text=details)

    def predict(self, file_path):
        self.cnn_res_label.config(text="SimpleCNN 运算中...", fg="#f39c12")
        self.resnet_res_label.config(text="ResNet50 运算中...", fg="#f39c12")
        self.root.update()
        
        try:
            # SimpleCNN 预测
            cnn_class, _, cnn_probs = predict_single_image(self.model_cnn, file_path)
            self._update_result(self.cnn_res_label, self.cnn_detail_label, cnn_class, cnn_probs)
            
            # ResNet50 预测
            res_class, _, res_probs = predict_single_image(self.model_resnet, file_path)
            self._update_result(self.resnet_res_label, self.resnet_detail_label, res_class, res_probs)
            
        except Exception as e:
            messagebox.showerror("识别失败", f"识别过程中发生错误:\n{e}")
            self.cnn_res_label.config(text="失败", fg="#c0392b")
            self.resnet_res_label.config(text="失败", fg="#c0392b")

if __name__ == "__main__":
    # 使用 TkinterDnD.Tk 替代普通的 tk.Tk 才能支持拖拽
    root = TkinterDnD.Tk()
    app = LogoPredictorGUI(root)
    root.mainloop()
