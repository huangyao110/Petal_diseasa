import sys
import os
import cv2
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QTabWidget, QScrollArea, 
                             QGridLayout, QGroupBox, QLineEdit, QSlider, QColorDialog, 
                             QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                             QProgressBar, QComboBox, QCheckBox, QSpinBox, QDoubleSpinBox)
from PyQt5.QtGui import QPixmap, QImage, QColor, QPalette, QFontDatabase
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import csv
import tempfile
import shutil
import uuid
import json
from inference import InferenceRoseDisease, ImageVisualizer

# 默认配置
DEFAULT_CONFIG = {
    "cfg_file": r"configs\Base-TridentNet-Fast-C4.yaml",
    "crop_weight": r'weights_from_online\STEP1_PETAL_CROP_TRIDENTENT.pth',
    "petal_encoder_name": "timm-regnetx_032",
    "petal_decoder_name": "unet",
    "disease_encoder_name": "mobilenet_v2",
    "disease_decoder_name": "unet++",
    "petal_weights_file": r'weights_from_online\STEP2_PETAL_SEG_REGNETX_032_UNET.ckpt',
    "diease_weights_file": r'logs\EX-2025-03-16-step3_mobilenetv2_uvnet++\version_0\checkpoints\epoch=48-step=17738.ckpt',
    "box_threshold": 0.5,
    "mask_threshold": 0.5
}

# 默认颜色配置
DEFAULT_COLORS = {
    "petal_color": "#00FF00",  # 绿色
    "disease_color": "#FF0000",  # 红色
    "alpha": 0.4  # 透明度
}

# 工作线程类，用于处理耗时操作
class WorkerThread(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    def __init__(self, infer, file_path, colors, batch_mode=False, directory=None):
        super().__init__()
        self.infer = infer
        self.file_path = file_path
        self.colors = colors
        self.batch_mode = batch_mode
        self.directory = directory
        self.result_folder = 'results'
        
    def run(self):
        try:
            if self.batch_mode and self.directory:
                self.process_batch()
            else:
                self.process_single_file()
        except Exception as e:
            self.error.emit(str(e))
    
    def process_single_file(self):
        # 创建结果目录
        filename = os.path.basename(self.file_path)
        result_dir = os.path.join(self.result_folder, os.path.splitext(filename)[0])
        os.makedirs(result_dir, exist_ok=True)
        
        # 执行步骤1：裁剪图像
        self.update_progress.emit(10)
        self.infer.step1_crop_img(img_file=self.file_path, save_dir=result_dir)
        
        # 获取裁剪后的图像列表
        cropped_images = [f for f in os.listdir(result_dir) if f.endswith('.jpg')]
        
        # 执行步骤2和3：花瓣分割和病害分割
        petal_dir = os.path.join(result_dir, 'petal_seg')
        disease_dir = os.path.join(result_dir, 'petal_disease')
        os.makedirs(petal_dir, exist_ok=True)
        os.makedirs(disease_dir, exist_ok=True)
        
        total_steps = len(cropped_images) * 2
        current_step = 0
        
        for img_file in cropped_images:
            img_path = os.path.join(result_dir, img_file)
            self.infer.step2_seg_petal(img_file=img_path, save_dir=petal_dir)
            current_step += 1
            self.update_progress.emit(10 + int(40 * current_step / total_steps))
            
            self.infer.step3_seg_disease(img_file=img_path, save_dir=disease_dir)
            current_step += 1
            self.update_progress.emit(10 + int(40 * current_step / total_steps))
        
        # 创建合并图像的目录
        merged_dir = os.path.join(result_dir, 'merged')
        os.makedirs(merged_dir, exist_ok=True)
        
        # 准备结果数据
        results = []
        for i, img_file in enumerate(cropped_images):
            self.update_progress.emit(50 + int(50 * i / len(cropped_images)))
            
            # 原始裁剪图像
            img_path = os.path.join(result_dir, img_file)
            
            # 花瓣分割结果
            petal_mask_file = f"{os.path.splitext(img_file)[0]}_petal_mask.png"
            petal_mask_path = os.path.join(petal_dir, petal_mask_file)
            
            # 病害分割结果
            disease_mask_file = f"{os.path.splitext(img_file)[0]}_disease_mask.png"
            disease_mask_path = os.path.join(disease_dir, disease_mask_file)
            
            # 计算病害面积比例
            petal_mask = cv2.imread(petal_mask_path, cv2.IMREAD_GRAYSCALE)
            disease_mask = cv2.imread(disease_mask_path, cv2.IMREAD_GRAYSCALE)
            
            petal_area = np.sum(petal_mask > 0)
            disease_area = np.sum(disease_mask > 0)
            
            if petal_area > 0:
                disease_ratio = disease_area / petal_area * 100
            else:
                disease_ratio = 0
            
            # 合并花瓣掩码和原图
            petal_merged_file = f"{os.path.splitext(img_file)[0]}_petal_merged.png"
            petal_merged_path = os.path.join(merged_dir, petal_merged_file)
            petal_merged_img = self.merge_image_with_mask(img_path, petal_mask_path, self.colors["petal_color"], self.colors["alpha"])
            cv2.imwrite(petal_merged_path, petal_merged_img)
            
            # 合并病害掩码和原图
            disease_merged_file = f"{os.path.splitext(img_file)[0]}_disease_merged.png"
            disease_merged_path = os.path.join(merged_dir, disease_merged_file)
            disease_merged_img = self.merge_image_with_mask(img_path, disease_mask_path, self.colors["disease_color"], self.colors["alpha"])
            cv2.imwrite(disease_merged_path, disease_merged_img)
            
            # 添加结果数据
            results.append({
                'id': i + 1,
                'image_path': img_path,
                'petal_mask_path': petal_mask_path,
                'disease_mask_path': disease_mask_path,
                'petal_merged_path': petal_merged_path,
                'disease_merged_path': disease_merged_path,
                'disease_ratio': disease_ratio,
                'petal_area': petal_area,
                'disease_area': disease_area
            })
        
        # 保存病害比例数据到CSV文件
        csv_file = os.path.join(result_dir, f"{os.path.splitext(filename)[0]}_disease_ratio.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['花朵ID', '病害面积比例(%)', '花瓣面积(像素)', '病害面积(像素)'])
            for result in results:
                writer.writerow([result['id'], f"{result['disease_ratio']:.2f}", 
                                result['petal_area'], result['disease_area']])
        
        self.update_progress.emit(100)
        self.finished.emit(results)
    
    def process_batch(self):
        # 获取目录中的所有图像文件
        image_files = [f for f in os.listdir(self.directory) 
                      if os.path.isfile(os.path.join(self.directory, f)) and 
                      f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 创建批处理结果目录
        batch_result_dir = os.path.join(self.result_folder, 'batch_' + os.path.basename(self.directory))
        os.makedirs(batch_result_dir, exist_ok=True)
        
        # 处理每个图像
        results_summary = []
        total_files = len(image_files)
        
        for i, img_file in enumerate(image_files):
            progress = int(i / total_files * 100)
            self.update_progress.emit(progress)
            
            img_path = os.path.join(self.directory, img_file)
            img_result_dir = os.path.join(batch_result_dir, os.path.splitext(img_file)[0])
            os.makedirs(img_result_dir, exist_ok=True)
            
            # 执行步骤1：裁剪图像
            self.infer.step1_crop_img(img_file=img_path, save_dir=img_result_dir)
            
            # 获取裁剪后的图像列表
            cropped_images = [f for f in os.listdir(img_result_dir) if f.endswith('.jpg')]
            
            # 执行步骤2和3：花瓣分割和病害分割
            petal_dir = os.path.join(img_result_dir, 'petal_seg')
            disease_dir = os.path.join(img_result_dir, 'petal_disease')
            os.makedirs(petal_dir, exist_ok=True)
            os.makedirs(disease_dir, exist_ok=True)
            
            for crop_img in cropped_images:
                crop_path = os.path.join(img_result_dir, crop_img)
                self.infer.step2_seg_petal(img_file=crop_path, save_dir=petal_dir)
                self.infer.step3_seg_disease(img_file=crop_path, save_dir=disease_dir)
            
            # 计算该图像的总体病害比例
            total_petal_area = 0
            total_disease_area = 0
            
            for crop_img in cropped_images:
                base_name = os.path.splitext(crop_img)[0]
                petal_mask_path = os.path.join(petal_dir, f"{base_name}_petal_mask.png")
                disease_mask_path = os.path.join(disease_dir, f"{base_name}_disease_mask.png")
                
                if os.path.exists(petal_mask_path) and os.path.exists(disease_mask_path):
                    petal_mask = cv2.imread(petal_mask_path, cv2.IMREAD_GRAYSCALE)
                    disease_mask = cv2.imread(disease_mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    total_petal_area += np.sum(petal_mask > 0)
                    total_disease_area += np.sum(disease_mask > 0)
            
            if total_petal_area > 0:
                disease_ratio = total_disease_area / total_petal_area * 100
            else:
                disease_ratio = 0
            
            results_summary.append({
                'image': img_file,
                'crops_count': len(cropped_images),
                'disease_ratio': disease_ratio,
                'petal_area': total_petal_area,
                'disease_area': total_disease_area
            })
        
        # 保存批处理结果到CSV
        csv_file = os.path.join(batch_result_dir, f"batch_results.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['图像名称', '花朵数量', '总花瓣面积', '总病害面积', '病害比例(%)'])
            for result in results_summary:
                writer.writerow([result['image'], result['crops_count'], 
                                result['petal_area'], result['disease_area'],
                                f"{result['disease_ratio']:.2f}"])
        
        self.update_progress.emit(100)
        self.finished.emit(results_summary)
    
    def merge_image_with_mask(self, image_path, mask_path, color_hex, alpha=0.4):
        """将图像与掩码合并，创建半透明叠加效果"""
        # 读取原始图像和掩码
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 调整掩码大小以匹配图像
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 将十六进制颜色转换为BGR格式
        color_hex = color_hex.lstrip('#')
        r = int(color_hex[0:2], 16)
        g = int(color_hex[2:4], 16)
        b = int(color_hex[4:6], 16)
        color = (b, g, r)  # OpenCV使用BGR格式
        
        # 创建彩色掩码
        colored_mask = np.zeros_like(img)
        colored_mask[mask > 0] = color
        
        # 合并图像和掩码
        result = cv2.addWeighted(colored_mask, alpha, img, 1 - alpha, 0)
        
        return result

# 主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("玫瑰花瓣病害检测系统")
        self.setMinimumSize(1200, 800)
        
        # 设置窗口样式
        self.setObjectName("mainWindow")
        
        # 加载配置
        self.config_file = 'model_config.json'
        self.colors_file = 'colors_config.json'
        self.config = self.load_config()
        self.colors = self.load_colors()
        
        # 初始化模型
        self.init_model()
        
        # 创建UI
        self.init_ui()
        
        # 确保结果目录存在
        os.makedirs('results', exist_ok=True)
    
    def load_config(self):
        """加载模型配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except:
                return DEFAULT_CONFIG.copy()
        return DEFAULT_CONFIG.copy()
    
    def load_colors(self):
        """加载颜色配置"""
        if os.path.exists(self.colors_file):
            try:
                with open(self.colors_file, 'r') as f:
                    return json.load(f)
            except:
                return DEFAULT_COLORS.copy()
        return DEFAULT_COLORS.copy()
    
    def save_config(self):
        """保存模型配置"""
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=4)
    
    def save_colors(self):
        """保存颜色配置"""
        with open(self.colors_file, 'w') as f:
            json.dump(self.colors, f, indent=4)
    
    def init_model(self):
        """初始化模型"""
        try:
            self.infer = InferenceRoseDisease(
                cfg_file=self.config["cfg_file"],
                petal_encoder_name=self.config["petal_encoder_name"],
                petal_decoder_name=self.config["petal_decoder_name"],
                disease_encoder_name=self.config["disease_encoder_name"],
                disease_decoder_name=self.config["disease_decoder_name"],
                crop_weight=self.config["crop_weight"],
                petal_weights_file=self.config["petal_weights_file"],
                diease_weights_file=self.config["diease_weights_file"],
                box_threshold=self.config["box_threshold"],
                mask_threshold=self.config["mask_threshold"]
            )
            self.visualizer = ImageVisualizer()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"模型初始化失败: {str(e)}")
    
    def init_ui(self):
        """初始化UI界面"""
        # 创建主标签页
        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        self.setCentralWidget(self.tabs)
        
        # 创建各个标签页
        self.create_home_tab()
        self.create_batch_tab()
        self.create_config_tab()
        
        # 设置状态栏
        self.statusBar().showMessage("就绪")
        self.statusBar().setObjectName("statusBar")
    
    def create_home_tab(self):
        """创建首页标签页"""
        home_tab = QWidget()
        home_tab.setObjectName("homeTab")
        self.tabs.addTab(home_tab, "单图处理")
        
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("玫瑰花瓣病害检测系统")
        title_label.setProperty("title", "true")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 上传区域
        upload_group = QGroupBox("上传图片")
        upload_group.setObjectName("uploadGroup")
        upload_layout = QVBoxLayout()
        
        upload_btn = QPushButton("选择图片文件")
        upload_btn.setObjectName("uploadBtn")
        upload_btn.setIcon(QApplication.style().standardIcon(QApplication.style().SP_DialogOpenButton))
        upload_btn.clicked.connect(self.select_image)
        upload_layout.addWidget(upload_btn)
        
        self.file_path_label = QLabel("未选择文件")
        self.file_path_label.setObjectName("filePathLabel")
        upload_layout.addWidget(self.file_path_label)
        
        process_btn = QPushButton("开始处理")
        process_btn.setObjectName("processBtn")
        process_btn.setIcon(QApplication.style().standardIcon(QApplication.style().SP_MediaPlay))
        process_btn.clicked.connect(self.process_image)
        upload_layout.addWidget(process_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setVisible(False)
        upload_layout.addWidget(self.progress_bar)
        
        upload_group.setLayout(upload_layout)
        layout.addWidget(upload_group)
        
        # 结果显示区域
        self.results_scroll = QScrollArea()
        self.results_scroll.setObjectName("resultsScroll")
        self.results_scroll.setWidgetResizable(True)
        self.results_widget = QWidget()
        self.results_widget.setObjectName("resultsWidget")
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_scroll.setWidget(self.results_widget)
        self.results_scroll.setVisible(False)
        
        layout.addWidget(self.results_scroll)
        
        home_tab.setLayout(layout)
    
    def create_batch_tab(self):
        """创建批量处理标签页"""
        batch_tab = QWidget()
        batch_tab.setObjectName("batchTab")
        self.tabs.addTab(batch_tab, "批量处理")
        
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("批量处理")
        title_label.setProperty("title", "true")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 目录选择区域
        dir_group = QGroupBox("选择图片目录")
        dir_group.setObjectName("dirGroup")
        dir_layout = QVBoxLayout()
        
        dir_btn = QPushButton("选择目录")
        dir_btn.setObjectName("dirBtn")
        dir_btn.setIcon(QApplication.style().standardIcon(QApplication.style().SP_DirOpenIcon))
        dir_btn.clicked.connect(self.select_directory)
        dir_layout.addWidget(dir_btn)
        
        self.dir_path_label = QLabel("未选择目录")
        self.dir_path_label.setObjectName("dirPathLabel")
        dir_layout.addWidget(self.dir_path_label)
        
        batch_process_btn = QPushButton("开始批量处理")
        batch_process_btn.setObjectName("batchProcessBtn")
        batch_process_btn.setIcon(QApplication.style().standardIcon(QApplication.style().SP_MediaPlay))
        batch_process_btn.clicked.connect(self.process_batch)
        dir_layout.addWidget(batch_process_btn)
        
        self.batch_progress_bar = QProgressBar()
        self.batch_progress_bar.setObjectName("batchProgressBar")
        self.batch_progress_bar.setVisible(False)
        dir_layout.addWidget(self.batch_progress_bar)
        
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)
        
        # 批量结果显示
        self.batch_results_group = QGroupBox("处理结果")
        self.batch_results_group.setObjectName("batchResultsGroup")
        self.batch_results_group.setVisible(False)
        batch_results_layout = QVBoxLayout()
        
        self.batch_table = QTableWidget()
        self.batch_table.setObjectName("batchTable")
        self.batch_table.setColumnCount(5)
        self.batch_table.setHorizontalHeaderLabels(["序号", "图片名称", "花朵数量", "病害比例(%)", "操作"])
        self.batch_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        batch_results_layout.addWidget(self.batch_table)
        
        export_csv_btn = QPushButton("导出CSV结果")
        export_csv_btn.setObjectName("exportCsvBtn")
        export_csv_btn.setIcon(QApplication.style().standardIcon(QApplication.style().SP_FileIcon))
        export_csv_btn.clicked.connect(self.export_batch_csv)
        batch_results_layout.addWidget(export_csv_btn)
        
        self.batch_results_group.setLayout(batch_results_layout)
        layout.addWidget(self.batch_results_group)
        
        batch_tab.setLayout(layout)
    
    def create_config_tab(self):
        """创建配置标签页"""
        config_tab = QWidget()
        config_tab.setObjectName("configTab")
        self.tabs.addTab(config_tab, "系统配置")
        
        layout = QVBoxLayout()
        
        # 标题
        title_label = QLabel("系统配置")
        title_label.setProperty("title", "true")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # 颜色设置
        color_group = QGroupBox("颜色设置")
        color_group.setObjectName("colorGroup")
        color_layout = QGridLayout()
        
        # 花瓣颜色
        petal_color_label = QLabel("花瓣掩码颜色:")
        petal_color_label.setObjectName("petalColorLabel")
        color_layout.addWidget(petal_color_label, 0, 0)
        self.petal_color_btn = QPushButton()
        self.petal_color_btn.setObjectName("petalColorBtn")
        self.petal_color_btn.setStyleSheet(f"background-color: {self.colors['petal_color']};")
        self.petal_color_btn.clicked.connect(lambda: self.select_color('petal_color'))
        color_layout.addWidget(self.petal_color_btn, 0, 1)
        
        # 病害颜色
        disease_color_label = QLabel("病害掩码颜色:")
        disease_color_label.setObjectName("diseaseColorLabel")
        color_layout.addWidget(disease_color_label, 1, 0)
        self.disease_color_btn = QPushButton()
        self.disease_color_btn.setObjectName("diseaseColorBtn")
        self.disease_color_btn.setStyleSheet(f"background-color: {self.colors['disease_color']};")
        self.disease_color_btn.clicked.connect(lambda: self.select_color('disease_color'))
        color_layout.addWidget(self.disease_color_btn, 1, 1)
        
        # 透明度
        alpha_label = QLabel("透明度:")
        alpha_label.setObjectName("alphaLabel")
        color_layout.addWidget(alpha_label, 2, 0)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setObjectName("alphaSlider")
        self.alpha_slider.setMinimum(1)
        self.alpha_slider.setMaximum(9)
        self.alpha_slider.setValue(int(self.colors['alpha'] * 10))
        self.alpha_slider.setTickPosition(QSlider.TicksBelow)
        self.alpha_slider.setTickInterval(1)
        color_layout.addWidget(self.alpha_slider, 2, 1)
        
        self.alpha_label = QLabel(f"当前值: {self.colors['alpha']}")
        self.alpha_label.setObjectName("alphaValueLabel")
        self.alpha_slider.valueChanged.connect(self.update_alpha_label)
        color_layout.addWidget(self.alpha_label, 2, 2)
        
        save_colors_btn = QPushButton("保存颜色设置")
        save_colors_btn.setObjectName("saveColorsBtn")
        save_colors_btn.setIcon(QApplication.style().standardIcon(QApplication.style().SP_DialogSaveButton))
        save_colors_btn.clicked.connect(self.save_color_settings)
        color_layout.addWidget(save_colors_btn, 3, 0, 1, 3)
        
        color_group.setLayout(color_layout)
        layout.addWidget(color_group)
        
        # 模型配置
        model_group = QGroupBox("模型配置")
        model_layout = QGridLayout()
        
        # 配置文件路径
        model_layout.addWidget(QLabel("配置文件路径:"), 0, 0)
        self.cfg_file_edit = QLineEdit(self.config['cfg_file'])
        model_layout.addWidget(self.cfg_file_edit, 0, 1)
        
        # 裁剪模型权重路径
        model_layout.addWidget(QLabel("裁剪模型权重路径:"), 1, 0)
        self.crop_weight_edit = QLineEdit(self.config['crop_weight'])
        model_layout.addWidget(self.crop_weight_edit, 1, 1)
        
        # 花瓣编码器名称
        model_layout.addWidget(QLabel("花瓣编码器名称:"), 2, 0)
        self.petal_encoder_edit = QLineEdit(self.config['petal_encoder_name'])
        model_layout.addWidget(self.petal_encoder_edit, 2, 1)
        
        # 花瓣解码器名称
        model_layout.addWidget(QLabel("花瓣解码器名称:"), 3, 0)
        self.petal_decoder_edit = QLineEdit(self.config['petal_decoder_name'])
        model_layout.addWidget(self.petal_decoder_edit, 3, 1)
        
        # 病害编码器名称
        model_layout.addWidget(QLabel("病害编码器名称:"), 4, 0)
        self.disease_encoder_edit = QLineEdit(self.config['disease_encoder_name'])
        model_layout.addWidget(self.disease_encoder_edit, 4, 1)
        
        # 病害解码器名称
        model_layout.addWidget(QLabel("病害解码器名称:"), 5, 0)
        self.disease_decoder_edit = QLineEdit(self.config['disease_decoder_name'])
        model_layout.addWidget(self.disease_decoder_edit, 5, 1)
        
        # 花瓣分割模型权重路径
        model_layout.addWidget(QLabel("花瓣分割模型权重路径:"), 6, 0)
        self.petal_weights_edit = QLineEdit(self.config['petal_weights_file'])
        model_layout.addWidget(self.petal_weights_edit, 6, 1)
        
        # 病害分割模型权重路径
        model_layout.addWidget(QLabel("病害分割模型权重路径:"), 7, 0)
        self.disease_weights_edit = QLineEdit(self.config['diease_weights_file'])
        model_layout.addWidget(self.disease_weights_edit, 7, 1)
        
        # 边界框阈值
        model_layout.addWidget(QLabel("边界框阈值:"), 8, 0)
        self.box_threshold_spin = QDoubleSpinBox()
        self.box_threshold_spin.setMinimum(0.1)
        self.box_threshold_spin.setMaximum(1.0)
        self.box_threshold_spin.setSingleStep(0.1)
        self.box_threshold_spin.setValue(self.config['box_threshold'])
        model_layout.addWidget(self.box_threshold_spin, 8, 1)
        
        # 掩码阈值
        model_layout.addWidget(QLabel("掩码阈值:"), 9, 0)
        self.mask_threshold_spin = QDoubleSpinBox()
        self.mask_threshold_spin.setMinimum(0.1)
        self.mask_threshold_spin.setMaximum(1.0)
        self.mask_threshold_spin.setSingleStep(0.1)
        self.mask_threshold_spin.setValue(self.config['mask_threshold'])
        model_layout.addWidget(self.mask_threshold_spin, 9, 1)
        
        save_model_btn = QPushButton("保存模型配置")
        save_model_btn.clicked.connect(self.save_model_settings)
        model_layout.addWidget(save_model_btn, 10, 0, 1, 2)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # 警告信息
        warning_label = QLabel("警告：修改模型配置后需要重启应用才能生效")
        warning_label.setObjectName("warningLabel")
        warning_label.setProperty("warning", "true")
        layout.addWidget(warning_label)
        
        config_tab.setLayout(layout)
    
    def select_image(self):
        """选择图片文件"""
        file_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png)")
        if file_path:
            self.file_path_label.setText(file_path)
            self.results_scroll.setVisible(False)
    
    def select_directory(self):
        """选择目录"""
        directory = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if directory:
            self.dir_path_label.setText(directory)
            self.batch_results_group.setVisible(False)
    
    def select_color(self, color_type):
        """选择颜色"""
        current_color = QColor(self.colors[color_type])
        color = QColorDialog.getColor(current_color, self, "选择颜色")
        if color.isValid():
            self.colors[color_type] = color.name()
            if color_type == 'petal_color':
                self.petal_color_btn.setStyleSheet(f"background-color: {color.name()};")
            else:
                self.disease_color_btn.setStyleSheet(f"background-color: {color.name()};")
    
    def update_alpha_label(self):
        """更新透明度标签"""
        alpha = self.alpha_slider.value() / 10.0
        self.alpha_label.setText(f"当前值: {alpha}")
    
    def save_color_settings(self):
        """保存颜色设置"""
        self.colors['alpha'] = self.alpha_slider.value() / 10.0
        self.save_colors()
        QMessageBox.information(self, "成功", "颜色设置已保存")
    
    def save_model_settings(self):
        """保存模型设置"""
        self.config['cfg_file'] = self.cfg_file_edit.text()
        self.config['crop_weight'] = self.crop_weight_edit.text()
        self.config['petal_encoder_name'] = self.petal_encoder_edit.text()
        self.config['petal_decoder_name'] = self.petal_decoder_edit.text()
        self.config['disease_encoder_name'] = self.disease_encoder_edit.text()
        self.config['disease_decoder_name'] = self.disease_decoder_edit.text()
        self.config['petal_weights_file'] = self.petal_weights_edit.text()
        self.config['diease_weights_file'] = self.disease_weights_edit.text()
        self.config['box_threshold'] = self.box_threshold_spin.value()
        self.config['mask_threshold'] = self.mask_threshold_spin.value()
        
        self.save_config()
        QMessageBox.information(self, "成功", "模型配置已保存，重启应用后生效")
    
    def process_image(self):
        """处理单张图片"""
        file_path = self.file_path_label.text()
        if file_path == "未选择文件":
            QMessageBox.warning(self, "警告", "请先选择图片文件")
            return
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # 创建工作线程
        self.worker = WorkerThread(self.infer, file_path, self.colors)
        self.worker.update_progress.connect(self.update_progress)
        self.worker.finished.connect(self.display_results)
        self.worker.error.connect(self.show_error)
        self.worker.start()
    
    def process_batch(self):
        """批量处理图片"""
        directory = self.dir_path_label.text()
        if directory == "未选择目录":
            QMessageBox.warning(self, "警告", "请先选择图片目录")
            return
        
        # 显示进度条
        self.batch_progress_bar.setVisible(True)
        self.batch_progress_bar.setValue(0)
        
        # 创建工作线程
        self.batch_worker = WorkerThread(self.infer, None, self.colors, True, directory)
        self.batch_worker.update_progress.connect(self.update_batch_progress)
        self.batch_worker.finished.connect(self.display_batch_results)
        self.batch_worker.error.connect(self.show_error)
        self.batch_worker.start()
    
    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)
    
    def update_batch_progress(self, value):
        """更新批处理进度条"""
        self.batch_progress_bar.setValue(value)
    
    def show_error(self, error_msg):
        """显示错误信息"""
        QMessageBox.critical(self, "错误", error_msg)
        self.progress_bar.setVisible(False)
        self.batch_progress_bar.setVisible(False)
    
    def display_results(self, results):
        """显示处理结果"""
        # 清空之前的结果
        for i in reversed(range(self.results_layout.count())): 
            self.results_layout.itemAt(i).widget().setParent(None)
        
        # 添加结果标题
        title_label = QLabel("处理结果")
        title_label.setObjectName("resultsTitle")
        title_label.setProperty("title", "true")
        title_label.setAlignment(Qt.AlignCenter)
        self.results_layout.addWidget(title_label)
        
        # 添加CSV导出按钮
        export_btn = QPushButton("导出CSV结果")
        export_btn.setObjectName("exportBtn")
        export_btn.setIcon(QApplication.style().standardIcon(QApplication.style().SP_FileIcon))
        export_btn.clicked.connect(lambda: self.export_csv(results))
        self.results_layout.addWidget(export_btn)
        
        # 显示每个花朵的结果
        for result in results:
            result_group = QGroupBox(f"花朵 #{result['id']}")
            result_layout = QGridLayout()
            
            # 原始图像
            result_layout.addWidget(QLabel("原始图像"), 0, 0)
            img_label = QLabel()
            img_pixmap = QPixmap(result['image_path'])
            img_label.setPixmap(img_pixmap.scaled(300, 300, Qt.KeepAspectRatio))
            result_layout.addWidget(img_label, 1, 0)
            
            # 花瓣分割
            result_layout.addWidget(QLabel("花瓣分割"), 0, 1)
            petal_label = QLabel()
            petal_pixmap = QPixmap(result['petal_mask_path'])
            petal_label.setPixmap(petal_pixmap.scaled(300, 300, Qt.KeepAspectRatio))
            result_layout.addWidget(petal_label, 1, 1)
            
            # 病害分割
            result_layout.addWidget(QLabel("病害分割"), 0, 2)
            disease_label = QLabel()
            disease_pixmap = QPixmap(result['disease_mask_path'])
            disease_label.setPixmap(disease_pixmap.scaled(300, 300, Qt.KeepAspectRatio))
            result_layout.addWidget(disease_label, 1, 2)
            
            # 花瓣叠加
            result_layout.addWidget(QLabel("花瓣叠加"), 2, 0)
            petal_merged_label = QLabel()
            petal_merged_pixmap = QPixmap(result['petal_merged_path'])
            petal_merged_label.setPixmap(petal_merged_pixmap.scaled(300, 300, Qt.KeepAspectRatio))
            result_layout.addWidget(petal_merged_label, 3, 0)
            
            # 病害叠加
            result_layout.addWidget(QLabel("病害叠加"), 2, 1)
            disease_merged_label = QLabel()
            disease_merged_pixmap = QPixmap(result['disease_merged_path'])
            disease_merged_label.setPixmap(disease_merged_pixmap.scaled(300, 300, Qt.KeepAspectRatio))
            result_layout.addWidget(disease_merged_label, 3, 1)
            
            # 病害比例
            ratio_label = QLabel(f"病害面积比例: {result['disease_ratio']:.2f}%")
            ratio_label.setObjectName("ratioLabel")
            ratio_label.setProperty("ratio", "true")
            result_layout.addWidget(ratio_label, 2, 2, 2, 1, Qt.AlignCenter)
            
            result_group.setLayout(result_layout)
            self.results_layout.addWidget(result_group)
        
        # 显示结果区域
        self.results_scroll.setVisible(True)
        self.progress_bar.setVisible(False)
    
    def display_batch_results(self, results):
        """显示批处理结果"""
        # 清空表格
        self.batch_table.setRowCount(0)
        
        # 添加结果到表格
        for i, result in enumerate(results):
            row = self.batch_table.rowCount()
            self.batch_table.insertRow(row)
            
            self.batch_table.setItem(row, 0, QTableWidgetItem(str(i+1)))
            self.batch_table.setItem(row, 1, QTableWidgetItem(result['image']))
            self.batch_table.setItem(row, 2, QTableWidgetItem(str(result['crops_count'])))
            self.batch_table.setItem(row, 3, QTableWidgetItem(f"{result['disease_ratio']:.2f}%"))
            
            # 添加查看详情按钮
            view_btn = QPushButton("查看详情")
            view_btn.clicked.connect(lambda checked, r=result: self.view_batch_detail(r))
            self.batch_table.setCellWidget(row, 4, view_btn)
        
        # 显示结果区域
        self.batch_results_group.setVisible(True)
        self.batch_progress_bar.setVisible(False)
    
    def view_batch_detail(self, result):
        """查看批处理详情"""
        # 这里可以实现查看详情的功能，例如打开结果目录或显示详细信息
        QMessageBox.information(self, "详情", f"图像: {result['image']}\n花朵数量: {result['crops_count']}\n病害比例: {result['disease_ratio']:.2f}%")
    
    def export_csv(self, results):
        """导出CSV结果"""
        file_path, _ = QFileDialog.getSaveFileName(self, "保存CSV文件", "", "CSV文件 (*.csv)")
        if file_path:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['花朵ID', '病害面积比例(%)', '花瓣面积(像素)', '病害面积(像素)'])
                for result in results:
                    writer.writerow([result['id'], f"{result['disease_ratio']:.2f}", 
                                    result['petal_area'], result['disease_area']])
            QMessageBox.information(self, "成功", f"CSV文件已保存到: {file_path}")
    
    def export_batch_csv(self):
        """导出批处理CSV结果"""
        file_path, _ = QFileDialog.getSaveFileName(self, "保存CSV文件", "", "CSV文件 (*.csv)")
        if file_path:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['图像名称', '花朵数量', '总花瓣面积', '总病害面积', '病害比例(%)'])
                for row in range(self.batch_table.rowCount()):
                    image = self.batch_table.item(row, 1).text()
                    crops_count = self.batch_table.item(row, 2).text()
                    disease_ratio = self.batch_table.item(row, 3).text().replace('%', '')
                    writer.writerow([image, crops_count, '', '', disease_ratio])
            QMessageBox.information(self, "成功", f"CSV文件已保存到: {file_path}")

# 主函数
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 加载字体
    QFontDatabase.addApplicationFont("msyh.ttc")
    
    # 加载样式表
    style_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.qss")
    if os.path.exists(style_file):
        with open(style_file, "r", encoding="utf-8") as f:
            app.setStyleSheet(f.read())
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())