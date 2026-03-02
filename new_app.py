import sys
import os
import cv2
import numpy as np
import json
import csv
import logging
import traceback
from typing import Dict, List, Optional, Tuple

# PyQt5 core modules
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QScrollArea,
    QGridLayout, QGroupBox, QLineEdit, QSlider, QColorDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QProgressBar, QSpinBox, QDoubleSpinBox, QSplitter, QStackedWidget,
    QDialog, QGraphicsView, QGraphicsScene, QSizePolicy, QFrame
)
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QPixmap, QColor, QFontDatabase, QIcon, QImage, QPainter
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QRectF

# Deep learning libraries
import torch
# Assuming utils.inference file exists in the same directory
from utils.inference_onnx import InferenceRoseDisease

# ==========================================
# 1. Basic Configuration & Path Management
# ==========================================

def get_resource_path(relative_path):
    """Get absolute resource path, compatible with packaged environment"""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
    else:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, relative_path)

# Path constants
CONFIG_DIR = "configs/app_config"
MODEL_CONFIG_FILE = get_resource_path(os.path.join(CONFIG_DIR, "model_config.json"))
COLORS_CONFIG_FILE = get_resource_path(os.path.join(CONFIG_DIR, "colors_config.json"))
STYLE_FILE = get_resource_path(os.path.join(CONFIG_DIR, "style.qss"))
LOGO_FILE = get_resource_path(os.path.join(CONFIG_DIR, 'logo.jpg'))
RESULTS_DIR = get_resource_path('demo')

# Default model configuration
DEFAULT_MODEL_CONFIG = {
    "cfg_file": get_resource_path(os.path.join('configs', 'model', 'Base-TridentNet-Fast-C4.yaml')),
    "crop_weight": get_resource_path(os.path.join('configs', 'model', 'STEP1_PETAL_CROP_TRIDENTENT.pth')),
    "petal_encoder_name": "timm-regnetx_032",
    "petal_decoder_name": "pan", 
    "disease_encoder_name": "timm-regnety_064",
    "disease_decoder_name": "pan",
    "petal_weights_file": get_resource_path(os.path.join('configs', 'model', 'STEP2_PETAL_SEG_REGNETX_032_UNET.ckpt')),
    "diease_weights_file": get_resource_path(os.path.join('configs', 'model', 'epoch=99_regent64_manet_ft_decoder.ckpt')),
    "box_threshold": 0.5,
    "mask_threshold": 0.5, 
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
}

# Default color configuration
DEFAULT_COLORS_CONFIG = {
    "petal_color": "#00FF00",
    "disease_color": "#FF0000",
    "alpha": 0.4
}

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ==========================================
# 2. Functional Classes (Fullscreen Viewer, Config Manager)
# ==========================================

class ImageViewer(QDialog):
    """Fullscreen image viewer with zoom and pan support"""
    def __init__(self, image_path, title="Full Screen Viewer", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximized) # Maximized by default
        self.resize(1000, 800)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Use GraphicsView for high-performance rendering
        self.view = QGraphicsView()
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        
        # Load image
        self.pixmap = QPixmap(image_path)
        if not self.pixmap.isNull():
            self.scene.addPixmap(self.pixmap)
        
        # View settings
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setDragMode(QGraphicsView.ScrollHandDrag) # Enable dragging
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse) # Zoom around mouse
        self.view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setBackgroundBrush(QColor(30, 30, 30)) # Dark background
        
        layout.addWidget(self.view)
        
        # Bottom operation hints
        hint = QLabel("Instructions: Mouse wheel zoom | Left-click drag | Double-click reset")
        hint.setStyleSheet("color: white; background-color: rgba(0,0,0,150); padding: 8px; font-weight: bold;")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)
        
        # Delay to fit window after loading
        QtCore.QTimer.singleShot(100, self.fit_in_view)

    def fit_in_view(self):
        """Fit image to window size"""
        if not self.pixmap.isNull():
            rect = QRectF(self.pixmap.rect())
            if not rect.isNull():
                self.view.fitInView(rect, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        """Mouse wheel zoom event"""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.view.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.view.scale(zoom_out_factor, zoom_out_factor)

    def mouseDoubleClickEvent(self, event):
        """Double-click to reset view"""
        self.fit_in_view()


class ConfigManager:
    """Configuration file manager for reading/writing JSON configs"""
    def __init__(self, model_config_path: str, colors_config_path: str):
        self.model_config_path = model_config_path
        self.colors_config_path = colors_config_path
        os.makedirs(os.path.dirname(model_config_path), exist_ok=True)
        os.makedirs(os.path.dirname(colors_config_path), exist_ok=True)
        self.model_config = self._load_json(model_config_path, DEFAULT_MODEL_CONFIG)
        self.colors_config = self._load_json(colors_config_path, DEFAULT_COLORS_CONFIG)
    
    def _load_json(self, file_path: str, default_data: Dict) -> Dict:
        if os.path.exists(file_path):
            try:
                # Critical fix: explicitly specify utf-8 encoding
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Unable to load config file {file_path}: {e}")
                return default_data.copy()
        return default_data.copy()
    
    def _save_json(self, file_path: str, data: Dict) -> bool:
        try:
            # Critical fix: explicitly specify utf-8 encoding
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"Unable to save config file {file_path}: {e}")
            return False
    
    def get_model_config(self) -> Dict: return self.model_config
    def get_colors_config(self) -> Dict: return self.colors_config
    def save_model_config(self) -> bool: return self._save_json(self.model_config_path, self.model_config)
    def save_colors_config(self) -> bool: return self._save_json(self.colors_config_path, self.colors_config)
    def update_model_config(self, new_config: Dict): self.model_config.update(new_config)
    def update_colors_config(self, new_config: Dict): self.colors_config.update(new_config)

# ==========================================
# 3. Core Business Logic Thread (with area calculation fix)
# ==========================================


class ProcessingWorker(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    
    # 修改 1: 增加 mode 参数
    def __init__(self, infer_instance, colors, file_path=None, directory=None, mode='dl'):
        super().__init__()
        self.infer = infer_instance
        self.colors = colors
        self.file_path = file_path
        self.directory = directory
        self.mode = mode  # 保存模式 ('dl' 或 'cv')
        self.batch_mode = bool(directory)
        self.result_folder = RESULTS_DIR
        os.makedirs(self.result_folder, exist_ok=True)
    
    def run(self):
        try:
            if self.batch_mode:
                results = self._process_batch()
            else:
                results = self._process_single_file()
            self.finished.emit(results)
        except Exception as e:
            logging.error(f"Error during processing: {traceback.format_exc()}")
            self.error.emit(str(e))
    
    def _extract_id_from_filename(self, filename):
        try:
            return int(os.path.splitext(filename)[0].split('_')[-1])
        except:
            return -1

    def _process_single_file(self):
        filename = os.path.basename(self.file_path)
        base_name = os.path.splitext(filename)[0]
        result_dir = os.path.join(self.result_folder, base_name)
        os.makedirs(result_dir, exist_ok=True)
        
        self.update_progress.emit(10)
        self.infer.step1_crop_img(img_file=self.file_path, save_dir=result_dir)
        
        boxes_path = os.path.join(result_dir, f"{base_name}_boxes_info.npy")
        try:
            all_boxes = np.load(boxes_path)
        except:
            logging.warning("boxes_info.npy not found")
            return []
        
        crops = [f for f in os.listdir(result_dir) if f.lower().endswith('.jpg')]
        crops.sort(key=self._extract_id_from_filename)
        
        petal_dir = os.path.join(result_dir, 'petal_seg')
        disease_dir = os.path.join(result_dir, 'petal_disease')
        os.makedirs(petal_dir, exist_ok=True)
        os.makedirs(disease_dir, exist_ok=True)
        
        results_data = []
        limit = min(len(crops), len(all_boxes))
        
        for i in range(limit):
            img_file = crops[i]
            box = all_boxes[i].astype(int)
            img_path = os.path.join(result_dir, img_file)
            crop_base = os.path.splitext(img_file)[0]
            
            # 修改 2: 传入 self.mode
            self.infer.step2_seg_petal(img_file=img_path, save_dir=petal_dir, mode=self.mode)
            
            # 病害分割通常保持 DL，除非你有专门的 CV 病害算法，否则这里不变
            self.infer.step3_seg_disease(img_file=img_path, save_dir=disease_dir)
            
            p_mask_path = os.path.join(petal_dir, f"{crop_base}_petal_mask.png")
            d_mask_path = os.path.join(disease_dir, f"{crop_base}_disease_mask.png")
            
            if os.path.exists(p_mask_path) and os.path.exists(d_mask_path):
                pm = cv2.imread(p_mask_path, cv2.IMREAD_GRAYSCALE)
                dm = cv2.imread(d_mask_path, cv2.IMREAD_GRAYSCALE)
                
                real_width = abs(box[2] - box[0])
                real_height = abs(box[3] - box[1])
                real_box_area = real_width * real_height
                
                mask_total_pixels = pm.shape[0] * pm.shape[1]
                
                petal_ratio = np.count_nonzero(pm) / mask_total_pixels
                disease_ratio_in_mask = np.count_nonzero(dm) / mask_total_pixels
                
                p_area = int(petal_ratio * real_box_area)
                d_area = int(disease_ratio_in_mask * real_box_area)
                
                ratio = (d_area / p_area * 100) if p_area > 0 else 0
                
                results_data.append({
                    'box': box,
                    'id': self._extract_id_from_filename(img_file),
                    'image_path': img_path,
                    'petal_mask_path': p_mask_path,
                    'disease_mask_path': d_mask_path,
                    'disease_ratio': ratio,
                    'petal_area': p_area,
                    'disease_area': d_area,
                    'source_image': filename,
                    'result_dir': result_dir
                })
            
            current_prog = 10 + int(80 * (i + 1) / limit)
            self.update_progress.emit(current_prog)

        self._save_csv(results_data, os.path.join(result_dir, f"{base_name}_disease_report.csv"))
        self._create_annotated_image(self.file_path, results_data, result_dir, base_name)
        
        self.update_progress.emit(100)
        return results_data

    def _process_batch(self):
        image_files = [f for f in os.listdir(self.directory) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            return []
            
        batch_name = 'batch_' + os.path.basename(self.directory)
        batch_dir = os.path.join(self.result_folder, batch_name)
        os.makedirs(batch_dir, exist_ok=True)
        
        summary_results = []
        total_files = len(image_files)
        
        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(self.directory, img_file)
            sub_res_dir = os.path.join(batch_dir, os.path.splitext(img_file)[0])
            os.makedirs(sub_res_dir, exist_ok=True)
            
            self.infer.step1_crop_img(img_file=img_path, save_dir=sub_res_dir)
            
            crops = [f for f in os.listdir(sub_res_dir) if f.lower().endswith('.jpg')]
            try:
                boxes = np.load(os.path.join(sub_res_dir, f"{os.path.splitext(img_file)[0]}_boxes_info.npy"))
            except:
                boxes = []
                
            total_p_area = 0
            total_d_area = 0
            
            petal_dir = os.path.join(sub_res_dir, 'petal_seg')
            disease_dir = os.path.join(sub_res_dir, 'petal_disease')
            os.makedirs(petal_dir, exist_ok=True)
            os.makedirs(disease_dir, exist_ok=True)
            
            for k, crop_img in enumerate(crops):
                if k >= len(boxes): break
                box = boxes[k]
                c_path = os.path.join(sub_res_dir, crop_img)
                
                # 修改 3: 批量处理时也传入 self.mode
                self.infer.step2_seg_petal(c_path, petal_dir, mode=self.mode)
                self.infer.step3_seg_disease(c_path, disease_dir)
                
                c_base = os.path.splitext(crop_img)[0]
                pp = os.path.join(petal_dir, f"{c_base}_petal_mask.png")
                dp = os.path.join(disease_dir, f"{c_base}_disease_mask.png")
                
                if os.path.exists(pp) and os.path.exists(dp):
                    pm = cv2.imread(pp, cv2.IMREAD_GRAYSCALE)
                    dm = cv2.imread(dp, cv2.IMREAD_GRAYSCALE)
                    
                    real_area = abs(box[2]-box[0]) * abs(box[3]-box[1])
                    mask_size = pm.shape[0] * pm.shape[1]
                    
                    pa = int((np.count_nonzero(pm)/mask_size) * real_area)
                    da = int((np.count_nonzero(dm)/mask_size) * real_area)
                    
                    total_p_area += pa
                    total_d_area += da
            
            ratio = (total_d_area / total_p_area * 100) if total_p_area > 0 else 0
            summary_results.append({
                'image': img_file,
                'crops_count': len(crops),
                'disease_ratio': ratio,
                'petal_area': total_p_area,
                'disease_area': total_d_area,
                'batch_result_dir': batch_dir
            })
            
            self.update_progress.emit(int((idx+1)/total_files * 100))
            
        return summary_results

    # ... _save_csv 和 _create_annotated_image 保持不变 ...
    def _save_csv(self, data, path):
        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Flower ID', 'Disease Area Ratio(%)', 'Petal Area(pixels)', 'Disease Area(pixels)'])
                for res in sorted(data, key=lambda x: x['id']):
                    writer.writerow([
                        res['id'], f"{res['disease_ratio']:.2f}", 
                        res['petal_area'], res['disease_area']
                    ])
        except Exception as e:
            logging.error(f"CSV save failed: {e}")

    def _create_annotated_image(self, original_path, results, result_dir, base_name):
        try:
            img = cv2.imread(original_path)
            if img is None: return
            h, w = img.shape[:2]
            
            for r in results:
                x1, y1, x2, y2 = r['box']
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                cw, ch = x2-x1, y2-y1
                
                for key, col_key in [('petal_mask_path', 'petal_color'), ('disease_mask_path', 'disease_color')]:
                    if os.path.exists(r[key]):
                        mask = cv2.imread(r[key], cv2.IMREAD_GRAYSCALE)
                        mask = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
                        
                        color_hex = self.colors[col_key].lstrip('#')
                        bgr = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
                        
                        overlay = img[y1:y2, x1:x2].copy()
                        overlay[mask > 0] = bgr
                        cv2.addWeighted(overlay, self.colors['alpha'], img[y1:y2, x1:x2], 1-self.colors['alpha'], 0, img[y1:y2, x1:x2])

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{r['id']} {r['disease_ratio']:.1f}%"
                cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out_path = os.path.join(result_dir, f"{base_name}_annotated.png")
            cv2.imwrite(out_path, img)
            
            if results:
                results[0]['annotated_image_path'] = out_path
                results[0]['original_image_path'] = original_path
        except Exception as e:
            logging.error(f"Annotated image creation failed: {e}")

# 必须先添加这两个导入
from PyQt5.QtWidgets import QRadioButton, QButtonGroup

class HomeTab(QWidget):
    """Main operation interface: file selection, result display, view switching"""
    # 修改 1: 信号现在携带两个字符串 (path, mode)
    start_processing_requested = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_input_path = None
        self.is_batch_mode = False
        self._init_ui()
        self._init_logo() 

    # ... _init_logo 和 resizeEvent 保持不变 ...
    def _init_logo(self):
        self.logo_label = QLabel(self)
        self.logo_label.setAttribute(Qt.WA_TransparentForMouseEvents)
        if os.path.exists(LOGO_FILE):
            pix = QPixmap(LOGO_FILE)
            if not pix.isNull():
                image = pix.toImage()
                image = image.convertToFormat(QImage.Format_ARGB32)
                for y in range(image.height()):
                    for x in range(image.width()):
                        pixel = image.pixelColor(x, y)
                        pixel.setAlpha(250)
                        image.setPixelColor(x, y, pixel)
                final_pix = QPixmap.fromImage(image)
                scaled = final_pix.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.logo_label.setPixmap(scaled)
        self.logo_label.resize(320, 320)
        self.logo_label.move(20, self.height() - 190)
        self.logo_label.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'logo_label'):
            self.logo_label.move(20, self.height() - 220)
            self.logo_label.raise_()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        
        # --- Left: Control Area ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        gb_input = QGroupBox("Input Selection")
        gb_layout = QVBoxLayout(gb_input)
        
        self.lbl_path = QLabel("No file selected")
        self.lbl_path.setWordWrap(True)
        self.lbl_path.setStyleSheet("color: #666; font-style: italic;")
        
        btn_img = QPushButton(" Select Single Image")
        btn_img.setIcon(QApplication.style().standardIcon(QApplication.style().SP_FileIcon))
        btn_img.clicked.connect(self._select_img)
        
        btn_dir = QPushButton(" Select Batch Folder")
        btn_dir.setIcon(QApplication.style().standardIcon(QApplication.style().SP_DirIcon))
        btn_dir.clicked.connect(self._select_dir)
        
        # 修改 2: 增加算法选择
        gb_mode = QGroupBox("Segmentation Algorithm")
        v_mode = QVBoxLayout(gb_mode)
        self.rb_dl = QRadioButton("Deep Learning (DL)")
        self.rb_dl.setChecked(True) # 默认选中 DL
        self.rb_dl.setToolTip("Use trained SegModel for high accuracy in complex backgrounds")
        
        self.rb_cv = QRadioButton("Computer Vision (CV)")
        self.rb_cv.setToolTip("Use HSV+Otsu algorithm. Faster, best for simple backgrounds (e.g. white paper)")
        
        self.mode_group = QButtonGroup(self)
        self.mode_group.addButton(self.rb_dl)
        self.mode_group.addButton(self.rb_cv)
        
        v_mode.addWidget(self.rb_dl)
        v_mode.addWidget(self.rb_cv)
        
        btn_run = QPushButton(" Start Analysis")
        btn_run.setObjectName("processBtn")
        btn_run.setIcon(QApplication.style().standardIcon(QApplication.style().SP_MediaPlay))
        btn_run.clicked.connect(self._run_process)
        
        gb_layout.addWidget(self.lbl_path)
        gb_layout.addWidget(btn_img)
        gb_layout.addWidget(btn_dir)
        gb_layout.addWidget(gb_mode) # 添加模式选择框
        gb_layout.addWidget(btn_run)
        
        left_layout.addWidget(gb_input)
        
        self.pbar = QProgressBar()
        self.pbar.setVisible(False)
        left_layout.addWidget(self.pbar)
        left_layout.addStretch()
        
        # --- Right Area 保持不变 ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        toolbar = QHBoxLayout()
        self.btn_view_switch = QPushButton("Switch to Data Table View")
        self.btn_view_switch.setCheckable(True)
        self.btn_view_switch.setIcon(QApplication.style().standardIcon(QApplication.style().SP_FileDialogListView))
        self.btn_view_switch.clicked.connect(self._toggle_view)
        
        self.btn_export = QPushButton("Export Results CSV")
        self.btn_export.setEnabled(False)
        self.btn_export.setIcon(QApplication.style().standardIcon(QApplication.style().SP_DialogSaveButton))
        
        toolbar.addWidget(self.btn_view_switch)
        toolbar.addWidget(self.btn_export)
        toolbar.addStretch()
        right_layout.addLayout(toolbar)
        
        self.stack = QStackedWidget()
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.res_layout = QVBoxLayout(self.scroll_content)
        self.res_layout.setAlignment(Qt.AlignTop)
        self.scroll_area.setWidget(self.scroll_content)
        
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        
        self.stack.addWidget(self.scroll_area)
        self.stack.addWidget(self.table)
        
        right_layout.addWidget(self.stack)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([300, 900])
        main_layout.addWidget(splitter)

    # ... _select_img 和 _select_dir 保持不变 ...
    def _select_img(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.jpg *.png *.jpeg)")
        if f:
            self.current_input_path = f
            self.is_batch_mode = False
            self.lbl_path.setText(os.path.basename(f))
            self._reset_results()
            pix = QPixmap(f)
            if not pix.isNull():
                lbl = QLabel()
                lbl.setPixmap(pix.scaled(400, 300, Qt.KeepAspectRatio))
                lbl.setAlignment(Qt.AlignCenter)
                self.res_layout.addWidget(lbl)

    def _select_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Folder")
        if d:
            self.current_input_path = d
            self.is_batch_mode = True
            self.lbl_path.setText(os.path.basename(d))
            self._reset_results()
            self.res_layout.addWidget(QLabel("Folder selected, click Start to begin batch processing..."))

    def _run_process(self):
        if self.current_input_path:
            self.pbar.setValue(0)
            self.pbar.setVisible(True)
            self._reset_results()
            
            # 修改 3: 获取选中的模式
            mode = 'dl' if self.rb_dl.isChecked() else 'cv'
            
            # 发送路径和模式
            self.start_processing_requested.emit(self.current_input_path, mode)
        else:
            QMessageBox.warning(self, "Warning", "Please select an image or folder first")

    # ... 后面的方法保持不变 ...
    def _reset_results(self):
        while self.res_layout.count():
            item = self.res_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.table.clear()
        self.table.setRowCount(0)
        self.btn_export.setEnabled(False)

    def _toggle_view(self, checked):
        if checked:
            self.stack.setCurrentIndex(1)
            self.btn_view_switch.setText("Switch to Image Preview View")
        else:
            self.stack.setCurrentIndex(0)
            self.btn_view_switch.setText("Switch to Data Table View")

    @QtCore.pyqtSlot(int)
    def update_progress(self, val):
        self.pbar.setValue(val)

    @QtCore.pyqtSlot(list)
    def display_results(self, results):
        self.pbar.setVisible(False)
        if not results:
            self.res_layout.addWidget(QLabel("No results detected"))
            return
        
        self.btn_export.setEnabled(True)
        try: self.btn_export.clicked.disconnect() 
        except: pass
        
        if self.is_batch_mode:
            self.table.setColumnCount(5)
            self.table.setHorizontalHeaderLabels(["Image Name", "Flower Count", "Total Petal Area", "Total Disease Area", "Disease Ratio(%)"])
            self.table.setRowCount(len(results))
            for i, r in enumerate(results):
                self.table.setItem(i, 0, QTableWidgetItem(r['image']))
                self.table.setItem(i, 1, QTableWidgetItem(str(r['crops_count'])))
                self.table.setItem(i, 2, QTableWidgetItem(str(r['petal_area'])))
                self.table.setItem(i, 3, QTableWidgetItem(str(r['disease_area'])))
                self.table.setItem(i, 4, QTableWidgetItem(f"{r['disease_ratio']:.2f}"))
            self.res_layout.addWidget(QLabel("Batch processing complete, switch to table view for summary."))
        else:
            self.table.setColumnCount(4)
            self.table.setHorizontalHeaderLabels(["ID", "Disease Ratio(%)", "Petal Area(px)", "Disease Area(px)"])
            self.table.setRowCount(len(results))
            for i, r in enumerate(results):
                self.table.setItem(i, 0, QTableWidgetItem(str(r['id'])))
                self.table.setItem(i, 1, QTableWidgetItem(f"{r['disease_ratio']:.2f}"))
                self.table.setItem(i, 2, QTableWidgetItem(str(r['petal_area'])))
                self.table.setItem(i, 3, QTableWidgetItem(str(r['disease_area'])))
            
            self._reset_results_layout_only()
            res = results[0]
            items_to_show = [("Original Image", res.get('original_image_path')), 
                             ("Analysis Result (Annotated)", res.get('annotated_image_path'))]
            for title, path in items_to_show:
                if path and os.path.exists(path):
                    group = QGroupBox(title)
                    g_layout = QVBoxLayout(group)
                    lbl = QLabel()
                    pix = QPixmap(path)
                    if not pix.isNull():
                        lbl.setPixmap(pix.scaledToWidth(800, Qt.SmoothTransformation))
                        lbl.setAlignment(Qt.AlignCenter)
                        g_layout.addWidget(lbl)
                    btn_full = QPushButton(f" Fullscreen View {title}")
                    btn_full.setIcon(self.style().standardIcon(self.style().SP_TitleBarMaxButton))
                    btn_full.clicked.connect(lambda checked, p=path, t=title: self._open_fullscreen(p, t))
                    g_layout.addWidget(btn_full)
                    self.res_layout.addWidget(group)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def _reset_results_layout_only(self):
        while self.res_layout.count():
            item = self.res_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def _open_fullscreen(self, path, title):
        viewer = ImageViewer(path, title, self)
        viewer.exec_() 

    @QtCore.pyqtSlot(str)
    def show_error(self, msg):
        self.pbar.setVisible(False)
        QMessageBox.critical(self, "Error", msg)


class ConfigTab(QWidget):
    """Configuration interface"""
    model_config_changed = pyqtSignal()
    def __init__(self, mgr, parent=None):
        super().__init__(parent)
        self.mgr = mgr
        layout = QVBoxLayout(self)
        
        # Model configuration group
        gb_model = QGroupBox("Model Path & Parameter Configuration")
        grid = QGridLayout(gb_model)
        self.edits = {}
        
        keys = ["cfg_file", "crop_weight", "petal_weights_file", "diease_weights_file"]
        row = 0
        for k in keys:
            grid.addWidget(QLabel(k + ":"), row, 0)
            edit = QLineEdit()
            self.edits[k] = edit
            grid.addWidget(edit, row, 1)
            btn = QPushButton("Browse")
            btn.clicked.connect(lambda _, e=edit: self._browse_file(e))
            grid.addWidget(btn, row, 2)
            row += 1
            
        btn_save = QPushButton("Save Configuration")
        btn_save.clicked.connect(self._save_config)
        grid.addWidget(btn_save, row, 0, 1, 3)
        layout.addWidget(gb_model)
        
        # Color configuration group
        gb_color = QGroupBox("Display Color Configuration")
        h_layout = QHBoxLayout(gb_color)
        self.btn_p_color = QPushButton("Petal Color")
        self.btn_p_color.clicked.connect(lambda: self._pick_color('petal_color', self.btn_p_color))
        self.btn_d_color = QPushButton("Disease Color")
        self.btn_d_color.clicked.connect(lambda: self._pick_color('disease_color', self.btn_d_color))
        
        h_layout.addWidget(self.btn_p_color)
        h_layout.addWidget(self.btn_d_color)
        
        layout.addWidget(gb_color)
        layout.addStretch()
        
        self._load_ui_values()

    def _load_ui_values(self):
        c = self.mgr.get_model_config()
        for k, e in self.edits.items():
            e.setText(str(c.get(k, "")))
        
        col = self.mgr.get_colors_config()
        self.btn_p_color.setStyleSheet(f"background-color: {col.get('petal_color', '#00FF00')}")
        self.btn_d_color.setStyleSheet(f"background-color: {col.get('disease_color', '#FF0000')}")

    def _browse_file(self, edit):
        f, _ = QFileDialog.getOpenFileName(self, "Select File")
        if f: edit.setText(f)

    def _pick_color(self, key, btn):
        color = QColorDialog.getColor()
        if color.isValid():
            hex_c = color.name()
            btn.setStyleSheet(f"background-color: {hex_c}")
            self.mgr.update_colors_config({key: hex_c})
            self.mgr.save_colors_config()

    def _save_config(self):
        new_conf = {k: v.text() for k, v in self.edits.items()}
        self.mgr.update_model_config(new_conf)
        self.mgr.save_model_config()
        self.model_config_changed.emit()
        QMessageBox.information(self, "Success", "Configuration saved successfully")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PetalSpot Pro - Intelligent Petal Disease Analysis System")
        self.resize(1280, 960) 
        
        if os.path.exists(LOGO_FILE):
            self.setWindowIcon(QIcon(LOGO_FILE))
        
        if os.path.exists(STYLE_FILE):
            try:
                with open(STYLE_FILE, 'r', encoding='utf-8') as f:
                    self.setStyleSheet(f.read())
            except Exception as e:
                logging.error(f"Failed to load style sheet (UnicodeError fixed): {e}")

        self.config_mgr = ConfigManager(MODEL_CONFIG_FILE, COLORS_CONFIG_FILE)
        self.inference_engine = None
        self._init_model()
        
        self.tabs = QTabWidget()
        self.home_tab = HomeTab()
        self.config_tab = ConfigTab(self.config_mgr)
        
        self.tabs.addTab(self.home_tab, "Analysis Home")
        self.tabs.addTab(self.config_tab, "System Settings")
        self.setCentralWidget(self.tabs)
        
        # 信号连接保持不变，但 home_tab.start_processing_requested 现在携带两个参数
        self.home_tab.start_processing_requested.connect(self._start_processing)
        self.config_tab.model_config_changed.connect(self._init_model)

    def _init_model(self):
        c = self.config_mgr.get_model_config()
        try:
            self.inference_engine = InferenceRoseDisease(
                cfg_file=c["cfg_file"], 
                petal_encoder_name=c["petal_encoder_name"],
                petal_decoder_name=c["petal_decoder_name"], 
                disease_encoder_name=c["disease_encoder_name"],
                disease_decoder_name=c["disease_decoder_name"], 
                crop_weight=c["crop_weight"],
                petal_weights_file=c["petal_weights_file"], 
                diease_weights_file=c["diease_weights_file"],
                box_threshold=float(c.get("box_threshold", 0.5)), 
                mask_threshold=float(c.get("mask_threshold", 0.5)),
                device=c["device"]
            )
            self.statusBar().showMessage("Model loaded successfully", 3000)
        except Exception as e:
            logging.error(f"Model initialization failed: {e}")
            QMessageBox.critical(self, "Error", f"Model loading failed, please check configuration.\n{e}")

    # 修改: 接收 mode 参数
    def _start_processing(self, path, mode):
        """Start worker thread"""
        if not self.inference_engine:
            QMessageBox.warning(self, "Warning", "Model not initialized, cannot process.")
            return
            
        self.worker = ProcessingWorker(
            self.inference_engine, 
            self.config_mgr.get_colors_config(),
            file_path=path if os.path.isfile(path) else None,
            directory=path if os.path.isdir(path) else None,
            mode=mode # 传递 mode 给 Worker
        )
        self.worker.update_progress.connect(self.home_tab.update_progress)
        self.worker.finished.connect(self.home_tab.display_results)
        self.worker.error.connect(self.home_tab.show_error)
        self.worker.start()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set font to prevent Chinese character issues (optional)
    font = QtGui.QFont("Microsoft YaHei", 9)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())