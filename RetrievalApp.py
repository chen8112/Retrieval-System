import csv
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from sklearn.metrics.pairwise import euclidean_distances
import os
import cv2
import numpy as np
import color_M
import color_H
import multi
import GLCM
import SIFT
import multi2
model="F:\\final\\feature\\kmeans.pkl"
model2="F:\\final\\feature\\multi2.pkl"

class ImageRetrievalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("木质包装标识检索系统")
        self.geometry("980x500")

        # 顶部框架
        self.top_frame = tk.Frame(self)
        self.top_frame.pack(side=tk.TOP, padx=20, pady=20, fill=tk.X)

        self.title_label = tk.Label(self.top_frame, text="木质包装标识检索系统", font=("Helvetica", 16, "bold"))
        self.title_label.pack(pady=5)

        # 中间框架
        self.center_frame = tk.Frame(self)
        self.center_frame.pack(side=tk.TOP, pady=10, fill=tk.X)

        # 按钮区域，平行排列
        self.button_frame = tk.Frame(self.center_frame)
        self.button_frame.pack(side=tk.TOP, pady=10)

        button_style = {'font': ("Helvetica", 12), 'bg': "#4CAF50", 'fg': "white", 'relief': tk.RAISED, 'bd': 2}

        self.upload_button = tk.Button(self.button_frame, text="上传图像", command=self.upload_image, **button_style)
        self.upload_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.method_label = tk.Label(self.button_frame, text="特征提取方法选择：", font=("Helvetica", 12))
        self.method_label.pack(side=tk.LEFT, padx=10, pady=5)

        self.method_var = tk.StringVar()
        self.method_combobox = ttk.Combobox(self.button_frame, textvariable=self.method_var, font=("Helvetica", 12))
        self.method_combobox['values'] = (
        "SIFT", "颜色直方图", "颜色矩", "颜色矩与GLCM特征融合", "颜色矩与SIFT特征融合")
        self.method_combobox.pack(side=tk.LEFT, padx=10, pady=5)

        self.search_button = tk.Button(self.button_frame, text="开始检索", command=self.search_images, **button_style)
        self.search_button.pack(side=tk.LEFT, padx=10, pady=5)

        # 下方框架
        self.bottom_frame = tk.Frame(self)
        self.bottom_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 下方左侧待检索图像显示区域
        self.image_frame = tk.Frame(self.bottom_frame, relief=tk.SUNKEN, borderwidth=2)
        self.image_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.image_display = tk.Label(self.image_frame, text="输入图像")
        self.image_display.pack(pady=5)

        self.path_label = tk.Label(self.image_frame, text="图像路径")
        self.path_label.pack(pady=5)

        # 下方右侧结果展示区域
        self.result_frame = tk.Frame(self.bottom_frame, relief=tk.SUNKEN, borderwidth=2)
        self.result_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.canvas = tk.Canvas(self.result_frame)
        self.scroll_y = tk.Scrollbar(self.result_frame, orient="vertical", command=self.canvas.yview)
        self.scroll_x = tk.Scrollbar(self.result_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.scroll_y.set, xscrollcommand=self.scroll_x.set)

        self.scrollable_frame = tk.Frame(self.canvas)
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        # 设置grid布局权重
        self.bottom_frame.grid_rowconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(0, weight=1)
        self.bottom_frame.grid_columnconfigure(1, weight=2)
    def upload_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])
        if self.image_path:
            img = Image.open(self.image_path)
            img = img.resize((200, 200), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            self.image_display.configure(image=img,text="待检索图像")
            self.image_display.image = img

            # 根据图像尺寸调整框的大小
            self.image_frame.config(width=img.width(), height=img.height())
            self.image_frame.pack_propagate(False)
            # 更新文件路径标签
            text = os.path.basename(self.image_path)
            self.path_label.config(text = os.path.basename(self.image_path))
    def search_images(self):
        if not self.image_path:
            messagebox.showerror("错误", "请先上传一张图像")
            return
        if not self.method_var.get():
            messagebox.showerror("错误", "请选择特征提取方法")
            return

        if self.method_var.get() == "颜色直方图":
            results=color_H.search_colorh(self.image_path,'colorh.csv')
        elif self.method_var.get() == "颜色矩":
            results=color_M.search_colorm(self.image_path,'colorm.csv')
        elif self.method_var.get() == "GLCM":
            results=GLCM.search_glcm(self.image_path,'glcm.csv')
        elif self.method_var.get() == "颜色矩与GLCM特征融合":
            results=multi.search_multi(self.image_path,'multi.csv')
        elif self.method_var.get() == "SIFT":
            kmeans = SIFT.load_kmeans_model(model)
            results = SIFT.search_image(self.image_path, kmeans, "sift.csv")
        elif self.method_var.get() == "颜色矩与SIFT特征融合":
            kmeans = multi2.load_kmeans_model(model2)
            results = multi2.search_image(self.image_path, kmeans, "multi2.csv")

        self.show_results(results)

    def show_results(self, results):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        for i, (image_path, dist) in enumerate(results):
            img = Image.open(image_path)
            img = img.resize((100, 100), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            img_label = tk.Label(self.scrollable_frame, image=img)
            img_label.image = img
            img_label.grid(row=(i // 5) * 2, column=(i % 5), padx=5, pady=5)  # 紧凑布局

            path_label = tk.Label(self.scrollable_frame, text=os.path.basename(image_path), wraplength=100)
            path_label.grid(row=(i // 5) * 2 + 1, column=(i % 5), padx=5, pady=5)  # 紧凑布局
