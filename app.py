import sys
import os
import cv2
import numpy as np
import torch
import json
import csv
import logging
import traceback
from typing import Dict, List, Optional, Any, Tuple

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QScrollArea,
    QGridLayout, QGroupBox, QLineEdit, QSlider, QColorDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QProgressBar, QSpinBox, QDoubleSpinBox
)
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QPixmap, QImage, QColor, QPalette, QFontDatabase, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize

# Assuming these exist and work as expected
try:
    from inference import InferenceRoseDisease, ImageVisualizer
except ImportError:
    # Provide dummy classes if the originals are not found
    # This allows the GUI code to run for review/testing purposes
    print("WARNING: 'inference' module not found. Using dummy classes.")
    class InferenceRoseDisease:
        def __init__(self, *args, **kwargs):
            print("Dummy InferenceRoseDisease initialized.")
        def step1_crop_img(self, *args, **kwargs):
            print("Dummy step1_crop_img called.")
            # Simulate creating dummy cropped files if save_dir is provided
            save_dir = kwargs.get('save_dir')
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                # Create a dummy jpg file
                dummy_path = os.path.join(save_dir, "dummy_crop_0.jpg")
                cv2.imwrite(dummy_path, np.zeros((100, 100, 3), dtype=np.uint8))

        def step2_seg_petal(self, *args, **kwargs):
            print("Dummy step2_seg_petal called.")
             # Simulate creating dummy mask file if save_dir is provided
            save_dir = kwargs.get('save_dir')
            img_file = kwargs.get('img_file')
            if save_dir and img_file:
                os.makedirs(save_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(img_file))[0]
                dummy_path = os.path.join(save_dir, f"{base_name}_petal_mask.png")
                cv2.imwrite(dummy_path, np.zeros((100, 100), dtype=np.uint8)) # Black mask


        def step3_seg_disease(self, *args, **kwargs):
            print("Dummy step3_seg_disease called.")
            # Simulate creating dummy mask file if save_dir is provided
            save_dir = kwargs.get('save_dir')
            img_file = kwargs.get('img_file')
            if save_dir and img_file:
                os.makedirs(save_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(img_file))[0]
                dummy_path = os.path.join(save_dir, f"{base_name}_disease_mask.png")
                cv2.imwrite(dummy_path, np.zeros((100, 100), dtype=np.uint8)) # Black mask

    class ImageVisualizer:
         def __init__(self, *args, **kwargs):
            print("Dummy ImageVisualizer initialized.")
# --- Constants ---

# File Paths
CONFIG_DIR = "configs/app_config"
MODEL_CONFIG_FILE = os.path.join(CONFIG_DIR, "model_config.json")
COLORS_CONFIG_FILE = os.path.join(CONFIG_DIR, "colors_config.json")
STYLE_FILE = r"configs\app_config\style.qss"
FONT_FILE = "msyh.ttc" # Example font file
LOGO_FILE = r'configs\app_config\logo.png'
RESULTS_DIR = 'demo'

# Default Configurations
DEFAULT_MODEL_CONFIG = {
    "cfg_file": r"configs/Base-TridentNet-Fast-C4.yaml",
    "crop_weight": r'weights_from_online/STEP1_PETAL_CROP_TRIDENTENT.pth',
    "petal_encoder_name": "timm-regnetx_032",
    "petal_decoder_name": "unet",
    "disease_encoder_name": "mobilenet_v2",
    "disease_decoder_name": "unet++",
    "petal_weights_file": r'weights_from_online/STEP2_PETAL_SEG_REGNETX_032_UNET.ckpt',
    "diease_weights_file": r'logs/EX-2025-03-16-step3_mobilenetv2_uvnet++/version_0/checkpoints/epoch=48-step=17738.ckpt',
    "box_threshold": 0.5,
    "mask_threshold": 0.5
}

DEFAULT_COLORS_CONFIG = {
    "petal_color": "#00FF00",  # Green
    "disease_color": "#FF0000",  # Red
    "alpha": 0.4
}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration Management ---

class ConfigManager:
    """Handles loading and saving of configuration files."""
    def __init__(self, model_config_path: str, colors_config_path: str):
        self.model_config_path = model_config_path
        self.colors_config_path = colors_config_path
        os.makedirs(os.path.dirname(model_config_path), exist_ok=True)
        os.makedirs(os.path.dirname(colors_config_path), exist_ok=True)
        self.model_config = self._load_json(model_config_path, DEFAULT_MODEL_CONFIG)
        self.colors_config = self._load_json(colors_config_path, DEFAULT_COLORS_CONFIG)

    def _load_json(self, file_path: str, default_data: Dict) -> Dict:
        """Loads data from a JSON file or returns default data."""
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON from {file_path}: {e}")
                return default_data.copy()
            except Exception as e:
                logging.error(f"Failed to load {file_path}: {e}")
                return default_data.copy()
        else:
            logging.warning(f"Config file not found: {file_path}. Using defaults.")
            # Optionally save the default config if the file doesn't exist
            # self._save_json(file_path, default_data)
            return default_data.copy()

    def _save_json(self, file_path: str, data: Dict) -> bool:
        """Saves data to a JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
            return True
        except Exception as e:
            logging.error(f"Failed to save config to {file_path}: {e}")
            return False

    def get_model_config(self) -> Dict:
        return self.model_config

    def get_colors_config(self) -> Dict:
        return self.colors_config

    def save_model_config(self) -> bool:
        logging.info(f"Saving model configuration to {self.model_config_path}")
        return self._save_json(self.model_config_path, self.model_config)

    def save_colors_config(self) -> bool:
        logging.info(f"Saving colors configuration to {self.colors_config_path}")
        return self._save_json(self.colors_config_path, self.colors_config)

    def update_model_config(self, new_config: Dict):
        self.model_config.update(new_config)

    def update_colors_config(self, new_config: Dict):
        self.colors_config.update(new_config)

# --- Worker Thread ---

class ProcessingWorker(QThread):
    """Handles long-running image processing tasks in a separate thread."""
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, infer_instance: InferenceRoseDisease, colors: Dict,
                 file_path: Optional[str] = None, directory: Optional[str] = None):
        super().__init__()
        if not file_path and not directory:
            raise ValueError("Either file_path or directory must be provided.")
        if file_path and directory:
            raise ValueError("Provide either file_path or directory, not both.")

        self.infer = infer_instance
        self.colors = colors
        self.file_path = file_path
        self.directory = directory
        self.batch_mode = bool(directory)
        self.result_folder = RESULTS_DIR
        os.makedirs(self.result_folder, exist_ok=True)

    def run(self):
        """Executes the processing task."""
        try:
            if self.batch_mode:
                results = self._process_batch()
            else:
                results = self._process_single_file()
            self.finished.emit(results)
        except Exception as e:
            logging.error(f"Error during processing: {e}\n{traceback.format_exc()}")
            self.error.emit(f"处理失败: {e}")

    def _process_single_file(self) -> List[Dict]:
        """Processes a single image file."""
        filename = os.path.basename(self.file_path)
        base_name = os.path.splitext(filename)[0]
        result_dir = os.path.join(self.result_folder, base_name)
        os.makedirs(result_dir, exist_ok=True)

        logging.info(f"Processing single file: {self.file_path}")
        logging.info(f"Results will be saved in: {result_dir}")

        # Step 1: Crop Image
        self.update_progress.emit(10)
        logging.info("Step 1: Cropping image...")
        self.infer.step1_crop_img(img_file=self.file_path, save_dir=result_dir)

        cropped_image_files = [f for f in os.listdir(result_dir) if f.lower().endswith('.jpg')]
        if not cropped_image_files:
            logging.warning("No cropped images found after step 1.")
            self.update_progress.emit(100)
            return [] # Return empty list if no crops found

        logging.info(f"Found {len(cropped_image_files)} cropped images.")

        # Step 2 & 3: Segmentation
        petal_dir = os.path.join(result_dir, 'petal_seg')
        disease_dir = os.path.join(result_dir, 'petal_disease')
        merged_dir = os.path.join(result_dir, 'merged')
        os.makedirs(petal_dir, exist_ok=True)
        os.makedirs(disease_dir, exist_ok=True)
        os.makedirs(merged_dir, exist_ok=True)

        total_steps = len(cropped_image_files) * 2
        current_step = 0
        results_data = []

        logging.info("Starting Step 2 (Petal Seg) and Step 3 (Disease Seg)...")
        for i, img_file in enumerate(cropped_image_files):
            img_path = os.path.join(result_dir, img_file)
            crop_base_name = os.path.splitext(img_file)[0]

            # Step 2
            self.infer.step2_seg_petal(img_file=img_path, save_dir=petal_dir)
            current_step += 1
            self.update_progress.emit(10 + int(40 * current_step / total_steps))

            # Step 3
            self.infer.step3_seg_disease(img_file=img_path, save_dir=disease_dir)
            current_step += 1
            self.update_progress.emit(10 + int(40 * current_step / total_steps))

            # Process results for this crop
            petal_mask_file = f"{crop_base_name}_petal_mask.png"
            petal_mask_path = os.path.join(petal_dir, petal_mask_file)
            disease_mask_file = f"{crop_base_name}_disease_mask.png"
            disease_mask_path = os.path.join(disease_dir, disease_mask_file)

            # Check if mask files were created successfully
            if not os.path.exists(petal_mask_path):
                 logging.warning(f"Petal mask not found: {petal_mask_path}")
                 continue # Skip if mask is missing
            if not os.path.exists(disease_mask_path):
                 logging.warning(f"Disease mask not found: {disease_mask_path}")
                 continue # Skip if mask is missing


            petal_mask = cv2.imread(petal_mask_path, cv2.IMREAD_GRAYSCALE)
            disease_mask = cv2.imread(disease_mask_path, cv2.IMREAD_GRAYSCALE)

            petal_area = np.count_nonzero(petal_mask)
            disease_area = np.count_nonzero(disease_mask)
            disease_ratio = (disease_area / petal_area * 100) if petal_area > 0 else 0

            # Merge images
            progress_offset = 50
            progress_range = 40 # Progress for merging steps (50% to 90%)
            merge_progress = progress_offset + int(progress_range * (i + 1) / len(cropped_image_files))
            self.update_progress.emit(merge_progress)


            petal_merged_path = os.path.join(merged_dir, f"{crop_base_name}_petal_merged.png")
            try:
                 petal_merged_img = self._merge_image_with_mask(img_path, petal_mask_path, self.colors["petal_color"], self.colors["alpha"])
                 cv2.imwrite(petal_merged_path, petal_merged_img)
            except Exception as merge_err:
                 logging.error(f"Error merging petal mask for {img_file}: {merge_err}")
                 petal_merged_path = None # Indicate merge failure


            disease_merged_path = os.path.join(merged_dir, f"{crop_base_name}_disease_merged.png")
            try:
                 disease_merged_img = self._merge_image_with_mask(img_path, disease_mask_path, self.colors["disease_color"], self.colors["alpha"])
                 cv2.imwrite(disease_merged_path, disease_merged_img)
            except Exception as merge_err:
                 logging.error(f"Error merging disease mask for {img_file}: {merge_err}")
                 disease_merged_path = None # Indicate merge failure


            results_data.append({
                'id': i + 1,
                'image_path': img_path,
                'petal_mask_path': petal_mask_path,
                'disease_mask_path': disease_mask_path,
                'petal_merged_path': petal_merged_path,
                'disease_merged_path': disease_merged_path,
                'disease_ratio': disease_ratio,
                'petal_area': petal_area,
                'disease_area': disease_area,
                'source_image': filename, # Add original source for context
                'result_dir': result_dir # Add result dir for reference
            })

        # Save CSV report
        self.update_progress.emit(95)
        csv_file = os.path.join(result_dir, f"{base_name}_disease_report.csv")
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['花朵ID', '病害面积比例(%)', '花瓣面积(像素)', '病害面积(像素)'])
                for result in results_data:
                    writer.writerow([
                        result['id'],
                        f"{result['disease_ratio']:.2f}",
                        result['petal_area'],
                        result['disease_area']
                    ])
            logging.info(f"Disease report saved to: {csv_file}")
        except Exception as e:
            logging.error(f"Failed to save CSV report: {e}")

        self.update_progress.emit(100)
        logging.info(f"Single file processing finished for: {self.file_path}")
        return results_data

    def _process_batch(self) -> List[Dict]:
        """Processes all images in a directory."""
        logging.info(f"Starting batch processing for directory: {self.directory}")
        image_files = [f for f in os.listdir(self.directory)
                       if os.path.isfile(os.path.join(self.directory, f)) and
                       f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not image_files:
             logging.warning(f"No image files found in directory: {self.directory}")
             self.update_progress.emit(100)
             return []

        batch_base_name = 'batch_' + os.path.basename(self.directory)
        batch_result_dir = os.path.join(self.result_folder, batch_base_name)
        os.makedirs(batch_result_dir, exist_ok=True)
        logging.info(f"Batch results will be saved in: {batch_result_dir}")

        results_summary = []
        total_files = len(image_files)

        for i, img_file in enumerate(image_files):
            progress = int((i / total_files) * 100)
            self.update_progress.emit(progress)
            logging.info(f"Processing batch file {i+1}/{total_files}: {img_file}")

            img_path = os.path.join(self.directory, img_file)
            img_base_name = os.path.splitext(img_file)[0]
            img_result_dir = os.path.join(batch_result_dir, img_base_name)
            # Don't fail if dir exists from previous run
            os.makedirs(img_result_dir, exist_ok=True)

            # Step 1: Crop
            self.infer.step1_crop_img(img_file=img_path, save_dir=img_result_dir)
            cropped_images = [f for f in os.listdir(img_result_dir) if f.lower().endswith('.jpg')]

            if not cropped_images:
                logging.warning(f"No crops found for image: {img_file}. Skipping.")
                results_summary.append({
                    'image': img_file,
                    'crops_count': 0,
                    'disease_ratio': 0,
                    'petal_area': 0,
                    'disease_area': 0,
                    'error': 'No flowers detected/cropped'
                })
                continue

            # Step 2 & 3: Segmentation for each crop
            petal_dir = os.path.join(img_result_dir, 'petal_seg')
            disease_dir = os.path.join(img_result_dir, 'petal_disease')
            os.makedirs(petal_dir, exist_ok=True)
            os.makedirs(disease_dir, exist_ok=True)

            total_petal_area = 0
            total_disease_area = 0

            for crop_img in cropped_images:
                crop_path = os.path.join(img_result_dir, crop_img)
                crop_base = os.path.splitext(crop_img)[0]

                self.infer.step2_seg_petal(img_file=crop_path, save_dir=petal_dir)
                self.infer.step3_seg_disease(img_file=crop_path, save_dir=disease_dir)

                petal_mask_path = os.path.join(petal_dir, f"{crop_base}_petal_mask.png")
                disease_mask_path = os.path.join(disease_dir, f"{crop_base}_disease_mask.png")

                if os.path.exists(petal_mask_path) and os.path.exists(disease_mask_path):
                    petal_mask = cv2.imread(petal_mask_path, cv2.IMREAD_GRAYSCALE)
                    disease_mask = cv2.imread(disease_mask_path, cv2.IMREAD_GRAYSCALE)
                    total_petal_area += np.count_nonzero(petal_mask)
                    total_disease_area += np.count_nonzero(disease_mask)
                else:
                    logging.warning(f"Masks missing for crop {crop_img} of image {img_file}")


            disease_ratio = (total_disease_area / total_petal_area * 100) if total_petal_area > 0 else 0

            results_summary.append({
                'image': img_file,
                'crops_count': len(cropped_images),
                'disease_ratio': disease_ratio,
                'petal_area': total_petal_area,
                'disease_area': total_disease_area,
                'result_dir': img_result_dir # Add for reference
            })

        # Save batch summary CSV
        self.update_progress.emit(95) # Indicate saving CSV
        csv_file = os.path.join(batch_result_dir, f"batch_summary_report.csv")
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['图像名称', '检测到的花朵数量', '总花瓣面积(像素)', '总病害面积(像素)', '总体病害比例(%)', '错误信息'])
                for result in results_summary:
                    writer.writerow([
                        result['image'],
                        result['crops_count'],
                        result['petal_area'],
                        result['disease_area'],
                        f"{result['disease_ratio']:.2f}",
                        result.get('error', '') # Include error if present
                    ])
            logging.info(f"Batch summary report saved to: {csv_file}")
        except Exception as e:
            logging.error(f"Failed to save batch summary CSV: {e}")

        self.update_progress.emit(100)
        logging.info(f"Batch processing finished for directory: {self.directory}")
        # Add the overall batch result directory to the summary for easy access
        for res in results_summary:
            res['batch_result_dir'] = batch_result_dir
        return results_summary # Return the list of dicts

    def _hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        """Converts a HEX color string to a BGR tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0)) # B, G, R

    def _merge_image_with_mask(self, img_path: str, mask_path: str, color_hex: str, alpha: float) -> np.ndarray:
        """Merges an image with a mask using a specified color and alpha."""
        img = cv2.imread(img_path)
        if img is None:
            raise IOError(f"Could not read image file: {img_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Could not read mask file: {mask_path}")

        # Ensure mask is binary (0 or 255)
        _, mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

        # Resize mask to match image dimensions ONLY if necessary
        if mask_binary.shape[:2] != img.shape[:2]:
             logging.warning(f"Resizing mask {os.path.basename(mask_path)} to match image {os.path.basename(img_path)}")
             mask_binary = cv2.resize(mask_binary, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)


        colored_mask_layer = np.zeros_like(img)
        bgr_color = self._hex_to_bgr(color_hex)
        colored_mask_layer[mask_binary > 0] = bgr_color

        # Blend using cv2.addWeighted
        blended_img = cv2.addWeighted(colored_mask_layer, alpha, img, 1 - alpha, 0)
        return blended_img

# --- UI Components ---

class HomeTab(QWidget):
    """Widget for the main image/directory processing tab."""
    # Signal to request processing start, passing file path or directory
    start_processing_requested = pyqtSignal(object) # object can be str (path) or None

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_input_path: Optional[str] = None # Store path for file or dir
        self.is_batch_mode: bool = False
        self._init_ui()

    def _init_ui(self):
        """Initializes the UI elements for this tab."""
        main_layout = QHBoxLayout(self)

        # --- Left Control Panel ---
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setAlignment(Qt.AlignTop)

        title_label = QLabel("玫瑰病害检测与分析")
        title_label.setProperty("title", "true")
        title_label.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title_label)

        # Input Selection Group
        input_group = QGroupBox("输入选择")
        input_layout = QVBoxLayout(input_group)

        file_btn = self._create_button(
            "选择单个图片文件",
            "fileBtn",
            QApplication.style().standardIcon(QApplication.style().SP_DialogOpenButton),
            self._select_image_file
        )
        input_layout.addWidget(file_btn)

        dir_btn = self._create_button(
            "选择图片文件夹 (批量处理)",
            "dirBtn",
            QApplication.style().standardIcon(QApplication.style().SP_DirOpenIcon),
            self._select_directory
        )
        input_layout.addWidget(dir_btn)

        self.path_label = QLabel("未选择文件或文件夹")
        self.path_label.setObjectName("pathLabel")
        self.path_label.setWordWrap(True)
        input_layout.addWidget(self.path_label)

        process_btn = self._create_button(
            "开始处理",
            "processBtn",
            QApplication.style().standardIcon(QApplication.style().SP_MediaPlay),
            self._request_processing
        )
        input_layout.addWidget(process_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("progressBar")
        self.progress_bar.setVisible(False)
        input_layout.addWidget(self.progress_bar)

        control_layout.addWidget(input_group)
        control_layout.addStretch(1) # Push elements to top

        main_layout.addWidget(control_panel)

        # --- Right Preview Panel ---
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)

        preview_title = QLabel("处理结果预览")
        preview_title.setProperty("title", "true")
        preview_title.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(preview_title)

        self.preview_scroll = QScrollArea()
        self.preview_scroll.setObjectName("previewScroll")
        self.preview_scroll.setWidgetResizable(True)
        self.preview_widget = QWidget() # Content widget for scroll area
        self.preview_widget.setObjectName("previewWidget")
        self.results_layout = QVBoxLayout(self.preview_widget) # Layout for results
        self.results_layout.setAlignment(Qt.AlignTop)
        self.preview_scroll.setWidget(self.preview_widget)
        preview_layout.addWidget(self.preview_scroll)

        main_layout.addWidget(preview_panel, stretch=3) # Give preview more space

    def _create_button(self, text: str, obj_name: str, icon: Optional[QIcon] = None, callback: Optional[callable] = None) -> QPushButton:
        """Helper to create a QPushButton."""
        btn = QPushButton(text)
        btn.setObjectName(obj_name)
        if icon:
            btn.setIcon(icon)
        if callback:
            btn.clicked.connect(callback)
        return btn

    def _select_image_file(self):
        """Handles selection of a single image file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片文件", "", "图片文件 (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.current_input_path = file_path
            self.is_batch_mode = False
            self.path_label.setText(f"文件: {os.path.basename(file_path)}")
            self._clear_results() # Clear previous results
            logging.info(f"Selected file: {file_path}")

    def _select_directory(self):
        """Handles selection of an image directory."""
        directory = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if directory:
            self.current_input_path = directory
            self.is_batch_mode = True
            self.path_label.setText(f"文件夹: {os.path.basename(directory)}")
            self._clear_results() # Clear previous results
            logging.info(f"Selected directory: {directory}")

    def _request_processing(self):
        """Emits a signal to the main window to start processing."""
        if not self.current_input_path:
            QMessageBox.warning(self, "输入缺失", "请先选择一个图片文件或文件夹。")
            return

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self._clear_results()
        self.start_processing_requested.emit(self.current_input_path) # Emit path

    @QtCore.pyqtSlot(int)
    def update_progress(self, value: int):
        """Updates the progress bar."""
        self.progress_bar.setValue(value)

    @QtCore.pyqtSlot(list)
    def display_results(self, results: List[Dict]):
        """Displays the processing results in the preview area."""
        self.progress_bar.setVisible(False)
        self._clear_results() # Clear previous results before adding new ones

        if not results:
            self.results_layout.addWidget(QLabel("处理完成，但未找到有效结果。"))
            return

        if self.is_batch_mode:
            self._display_batch_summary(results)
        else:
            self._display_single_file_details(results)

    @QtCore.pyqtSlot(str)
    def show_error(self, error_msg: str):
        """Shows an error message."""
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "处理错误", error_msg)
        self._clear_results() # Clear potentially partial results on error
        self.results_layout.addWidget(QLabel(f"处理失败: {error_msg}"))


    def _clear_results(self):
        """Clears the results display area."""
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _display_single_file_details(self, results: List[Dict]):
        """Displays detailed results for each cropped flower from a single file."""
        if not results: return

        source_image = results[0].get('source_image', '未知图片')
        result_dir = results[0].get('result_dir', RESULTS_DIR) # Get dir from first result

        overall_info_label = QLabel(f"图片 '{source_image}' 的处理结果 (检测到 {len(results)} 个花朵):")
        overall_info_label.setProperty("subtitle", "true")
        self.results_layout.addWidget(overall_info_label)


        # Add Export CSV button for single file results
        export_btn = QPushButton("导出此图片的结果 (CSV)")
        export_btn.setIcon(QApplication.style().standardIcon(QApplication.style().SP_DialogSaveButton))
        export_btn.clicked.connect(lambda: self._export_single_csv(results, source_image))
        self.results_layout.addWidget(export_btn)


        for result in results:
            group_box = QGroupBox(f"花朵 #{result['id']}")
            grid_layout = QGridLayout(group_box)

            # Row 0: Labels
            grid_layout.addWidget(QLabel("原始裁剪"), 0, 0, Qt.AlignCenter)
            grid_layout.addWidget(QLabel("花瓣分割"), 0, 1, Qt.AlignCenter)
            grid_layout.addWidget(QLabel("病害分割"), 0, 2, Qt.AlignCenter)

            # Row 1: Images
            max_size = 200 # Max display size for thumbnails
            grid_layout.addWidget(self._create_image_label(result['image_path'], max_size), 1, 0, Qt.AlignCenter)
            grid_layout.addWidget(self._create_image_label(result['petal_merged_path'], max_size), 1, 1, Qt.AlignCenter)
            grid_layout.addWidget(self._create_image_label(result['disease_merged_path'], max_size), 1, 2, Qt.AlignCenter)


            # Row 2: Disease Ratio Info
            ratio_label = QLabel(f"病害面积比例: {result['disease_ratio']:.2f}%")
            ratio_label.setProperty("highlight", "true")
            ratio_label.setAlignment(Qt.AlignCenter)
            grid_layout.addWidget(ratio_label, 2, 0, 1, 3) # Span across columns

            self.results_layout.addWidget(group_box)


        # Add saved location info
        save_info = QLabel(f"详细结果 (包括掩码和合并图像) 已保存到:\n{result_dir}")
        save_info.setWordWrap(True)
        self.results_layout.addWidget(save_info)

    def _display_batch_summary(self, results: List[Dict]):
        """Displays a summary table for batch processing results."""
        if not results: return

        batch_result_dir = results[0].get('batch_result_dir', RESULTS_DIR) # Get batch dir

        summary_label = QLabel(f"批量处理完成 ({len(results)} 个图片):")
        summary_label.setProperty("subtitle", "true")
        self.results_layout.addWidget(summary_label)

        # Add Export CSV Button for batch summary
        export_btn = QPushButton("导出批量处理摘要 (CSV)")
        export_btn.setIcon(QApplication.style().standardIcon(QApplication.style().SP_DialogSaveButton))
        export_btn.clicked.connect(lambda: self._export_batch_csv(results, batch_result_dir))
        self.results_layout.addWidget(export_btn)


        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["图片名称", "检测花朵数", "总花瓣面积", "总病害面积", "总体病害比例(%)"])
        table.setRowCount(len(results))

        for i, result in enumerate(results):
            table.setItem(i, 0, QTableWidgetItem(result['image']))
            table.setItem(i, 1, QTableWidgetItem(str(result['crops_count'])))
            table.setItem(i, 2, QTableWidgetItem(str(result['petal_area'])))
            table.setItem(i, 3, QTableWidgetItem(str(result['disease_area'])))
            ratio_item = QTableWidgetItem(f"{result['disease_ratio']:.2f}")
            # Maybe color code based on ratio?
            # if result['disease_ratio'] > 10: # Example threshold
            #     ratio_item.setBackground(QColor("lightcoral"))
            table.setItem(i, 4, ratio_item)

        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        table.setEditTriggers(QTableWidget.NoEditTriggers) # Make table read-only
        table.setMinimumHeight(300) # Ensure table has some visible height
        self.results_layout.addWidget(table)


        # Add saved location info
        save_info = QLabel(f"每个图片的详细结果已保存到子文件夹中:\n{batch_result_dir}")
        save_info.setWordWrap(True)
        self.results_layout.addWidget(save_info)


    def _create_image_label(self, image_path: Optional[str], max_size: int) -> QLabel:
        """Creates a QLabel with a scaled pixmap from an image file."""
        label = QLabel()
        label.setFixedSize(max_size, max_size) # Fixed size for grid alignment
        label.setAlignment(Qt.AlignCenter)
        if image_path and os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                 scaled_pixmap = pixmap.scaled(max_size, max_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                 label.setPixmap(scaled_pixmap)
            else:
                 label.setText("无法加载图片")
                 logging.warning(f"Failed to load pixmap for: {image_path}")

        else:
            label.setText("图片未生成\n或未找到")
            label.setStyleSheet("color: red;")
        return label


    def _export_single_csv(self, results: List[Dict], source_image_name: str):
        """Exports the detailed results for a single image to a CSV file."""
        if not results: return

        base_name = os.path.splitext(source_image_name)[0]
        default_filename = os.path.join(RESULTS_DIR, f"{base_name}_disease_report_exported.csv")

        file_path, _ = QFileDialog.getSaveFileName(self, "保存单图片结果CSV", default_filename, "CSV 文件 (*.csv)")

        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['花朵ID', '病害面积比例(%)', '花瓣面积(像素)', '病害面积(像素)'])
                    for result in results:
                         writer.writerow([
                             result['id'],
                             f"{result['disease_ratio']:.2f}",
                             result['petal_area'],
                             result['disease_area']
                         ])
                QMessageBox.information(self, "导出成功", f"结果已保存到:\n{file_path}")
                logging.info(f"Single image results exported to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"无法保存CSV文件:\n{e}")
                logging.error(f"Failed to export single CSV to {file_path}: {e}")


    def _export_batch_csv(self, results: List[Dict], batch_result_dir: str):
        """Exports the batch summary results to a CSV file."""
        if not results: return

        default_filename = os.path.join(batch_result_dir, f"batch_summary_report_exported.csv")
        file_path, _ = QFileDialog.getSaveFileName(self, "保存批量处理摘要CSV", default_filename, "CSV 文件 (*.csv)")

        if file_path:
            try:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['图像名称', '检测到的花朵数量', '总花瓣面积(像素)', '总病害面积(像素)', '总体病害比例(%)', '错误信息'])
                    for result in results:
                         writer.writerow([
                             result['image'],
                             result['crops_count'],
                             result['petal_area'],
                             result['disease_area'],
                             f"{result['disease_ratio']:.2f}",
                             result.get('error', '')
                         ])
                QMessageBox.information(self, "导出成功", f"批量处理摘要已保存到:\n{file_path}")
                logging.info(f"Batch summary exported to: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "导出失败", f"无法保存CSV文件:\n{e}")
                logging.error(f"Failed to export batch CSV to {file_path}: {e}")


class ConfigTab(QWidget):
    """Widget for the configuration settings tab."""
    # Signal emitted when model configuration might require a model reload
    model_config_changed = pyqtSignal()

    def __init__(self, config_manager: ConfigManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.config_manager = config_manager
        self._init_ui()
        self._load_settings() # Load initial values into UI

    def _init_ui(self):
        """Initializes the UI elements for this tab."""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignTop)

        title_label = QLabel("系统配置")
        title_label.setProperty("title", "true")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # --- Color Settings ---
        color_group = QGroupBox("显示颜色设置")
        color_layout = QGridLayout(color_group)

        # Petal Color
        color_layout.addWidget(QLabel("花瓣掩码颜色:"), 0, 0)
        self.petal_color_btn = QPushButton()
        self.petal_color_btn.setFixedSize(QSize(50, 25))
        self.petal_color_btn.clicked.connect(lambda: self._select_color('petal_color', self.petal_color_btn))
        color_layout.addWidget(self.petal_color_btn, 0, 1)

        # Disease Color
        color_layout.addWidget(QLabel("病害掩码颜色:"), 1, 0)
        self.disease_color_btn = QPushButton()
        self.disease_color_btn.setFixedSize(QSize(50, 25))
        self.disease_color_btn.clicked.connect(lambda: self._select_color('disease_color', self.disease_color_btn))
        color_layout.addWidget(self.disease_color_btn, 1, 1)

        # Alpha (Transparency)
        color_layout.addWidget(QLabel("掩码透明度:"), 2, 0)
        self.alpha_slider = QSlider(Qt.Horizontal)
        self.alpha_slider.setMinimum(1) # 0.1
        self.alpha_slider.setMaximum(9) # 0.9
        self.alpha_slider.setSingleStep(1)
        self.alpha_slider.setTickPosition(QSlider.TicksBelow)
        self.alpha_slider.setTickInterval(1)
        self.alpha_slider.valueChanged.connect(self._update_alpha_label)
        color_layout.addWidget(self.alpha_slider, 2, 1, 1, 2) # Span slider

        self.alpha_value_label = QLabel("当前值: 0.0") # Placeholder
        self.alpha_value_label.setObjectName("alphaValueLabel")
        color_layout.addWidget(self.alpha_value_label, 3, 1, Qt.AlignCenter) # Below slider

        save_colors_btn = QPushButton("保存颜色设置")
        save_colors_btn.setIcon(QApplication.style().standardIcon(QApplication.style().SP_DialogSaveButton))
        save_colors_btn.clicked.connect(self._save_color_settings)
        color_layout.addWidget(save_colors_btn, 4, 0, 1, 3) # Span button

        layout.addWidget(color_group)

        # --- Model Configuration ---
        model_group = QGroupBox("模型与路径配置 (修改后可能需要重启生效)")
        model_layout = QGridLayout(model_group)
        model_layout.setColumnStretch(1, 1) # Allow path edits to expand

        self.model_config_edits: Dict[str, QWidget] = {} # Store widgets for easy access

        # Helper to add a config row
        def add_config_row(label_text: str, key: str, widget: QWidget, row: int):
            model_layout.addWidget(QLabel(label_text), row, 0)
            model_layout.addWidget(widget, row, 1)
            # Add browse button for file paths
            if "file" in key or "weight" in key or "cfg" in key:
                 browse_btn = QPushButton("浏览...")
                 browse_btn.clicked.connect(lambda checked, w=widget, is_dir="dir" in key : self._browse_path(w, is_dir))
                 model_layout.addWidget(browse_btn, row, 2)

            self.model_config_edits[key] = widget

        # Add configuration fields
        add_config_row("配置文件路径:", "cfg_file", QLineEdit(), 0)
        add_config_row("裁剪模型权重:", "crop_weight", QLineEdit(), 1)
        add_config_row("花瓣分割模型权重:", "petal_weights_file", QLineEdit(), 2)
        add_config_row("病害分割模型权重:", "diease_weights_file", QLineEdit(), 3) # Typo in original key kept for consistency
        add_config_row("花瓣编码器:", "petal_encoder_name", QLineEdit(), 4)
        add_config_row("花瓣解码器:", "petal_decoder_name", QLineEdit(), 5)
        add_config_row("病害编码器:", "disease_encoder_name", QLineEdit(), 6)
        add_config_row("病害解码器:", "disease_decoder_name", QLineEdit(), 7)

        # Thresholds (SpinBoxes)
        self.box_thresh_spin = QDoubleSpinBox(minimum=0.0, maximum=1.0, singleStep=0.05, decimals=2)
        self.mask_thresh_spin = QDoubleSpinBox(minimum=0.0, maximum=1.0, singleStep=0.05, decimals=2)
        add_config_row("检测框置信度阈值:", "box_threshold", self.box_thresh_spin, 8)
        add_config_row("掩码二值化阈值:", "mask_threshold", self.mask_thresh_spin, 9)

        save_model_btn = QPushButton("保存模型配置")
        save_model_btn.setIcon(QApplication.style().standardIcon(QApplication.style().SP_DialogSaveButton))
        save_model_btn.clicked.connect(self._save_model_settings)
        model_layout.addWidget(save_model_btn, 10, 0, 1, 3) # Span across

        layout.addWidget(model_group)
        layout.addStretch(1) # Push content up

    def _load_settings(self):
        """Loads current config values into the UI elements."""
        # Load Colors
        colors = self.config_manager.get_colors_config()
        petal_color = colors.get('petal_color', DEFAULT_COLORS_CONFIG['petal_color'])
        disease_color = colors.get('disease_color', DEFAULT_COLORS_CONFIG['disease_color'])
        alpha = colors.get('alpha', DEFAULT_COLORS_CONFIG['alpha'])

        self.petal_color_btn.setStyleSheet(f"background-color: {petal_color};")
        self.disease_color_btn.setStyleSheet(f"background-color: {disease_color};")
        self.alpha_slider.setValue(int(alpha * 10))
        self._update_alpha_label() # Set initial label text

        # Load Model Config
        model_config = self.config_manager.get_model_config()
        for key, widget in self.model_config_edits.items():
            value = model_config.get(key, DEFAULT_MODEL_CONFIG.get(key)) # Fallback
            if isinstance(widget, QLineEdit):
                widget.setText(str(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QSpinBox): # Example if int spinbox was used
                 widget.setValue(int(value))

    def _select_color(self, config_key: str, button: QPushButton):
        """Opens a color dialog and updates the button background."""
        current_color_hex = self.config_manager.get_colors_config().get(config_key)
        current_qcolor = QColor(current_color_hex) if current_color_hex else Qt.white

        color = QColorDialog.getColor(current_qcolor, self, "选择颜色")
        if color.isValid():
            hex_color = color.name()
            button.setStyleSheet(f"background-color: {hex_color};")
            # Store temporarily, save happens on button click
            # self.config_manager.update_colors_config({config_key: hex_color})

    def _update_alpha_label(self):
        """Updates the label showing the current alpha value."""
        alpha = self.alpha_slider.value() / 10.0
        self.alpha_value_label.setText(f"当前值: {alpha:.1f}")

    def _save_color_settings(self):
        """Saves the selected colors and alpha to the config file."""
        try:
            # Extract color from button style (safer than assuming it was stored)
            petal_style = self.petal_color_btn.styleSheet()
            disease_style = self.disease_color_btn.styleSheet()

            petal_color = petal_style.split(":")[-1].strip().rstrip(';')
            disease_color = disease_style.split(":")[-1].strip().rstrip(';')
            alpha = self.alpha_slider.value() / 10.0

            self.config_manager.update_colors_config({
                'petal_color': petal_color,
                'disease_color': disease_color,
                'alpha': alpha
            })
            if self.config_manager.save_colors_config():
                 QMessageBox.information(self, "成功", "颜色设置已保存。")
                 logging.info("Color settings saved.")
            else:
                 QMessageBox.warning(self, "失败", "无法保存颜色设置。请检查日志。")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存颜色设置时出错: {e}")
            logging.error(f"Error saving color settings: {e}")

    def _browse_path(self, line_edit_widget: QLineEdit, is_directory: bool = False):
        """Opens a file or directory dialog to select a path."""
        current_path = line_edit_widget.text()
        parent_dir = os.path.dirname(current_path) if os.path.isfile(current_path) else current_path

        if is_directory:
            path = QFileDialog.getExistingDirectory(self, "选择文件夹", parent_dir)
        else:
            # Adjust filter based on expected file type if needed
            if "weight" in line_edit_widget.objectName().lower() or ".pth" in current_path or ".ckpt" in current_path:
                filter_str = "权重文件 (*.pth *.ckpt);;所有文件 (*)"
            elif "cfg" in line_edit_widget.objectName().lower() or ".yaml" in current_path:
                 filter_str = "配置文件 (*.yaml);;所有文件 (*)"
            else:
                 filter_str = "所有文件 (*)"
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", parent_dir, filter_str)

        if path:
            line_edit_widget.setText(path)


    def _save_model_settings(self):
        """Saves the model configuration settings."""
        try:
            new_config = {}
            for key, widget in self.model_config_edits.items():
                if isinstance(widget, QLineEdit):
                    new_config[key] = widget.text()
                elif isinstance(widget, QDoubleSpinBox):
                    new_config[key] = widget.value()
                elif isinstance(widget, QSpinBox):
                     new_config[key] = widget.value()

            self.config_manager.update_model_config(new_config)
            if self.config_manager.save_model_config():
                QMessageBox.information(self, "成功", "模型配置已保存。\n部分更改可能需要重启应用程序才能完全生效。")
                logging.info("Model settings saved.")
                self.model_config_changed.emit() # Notify main window
            else:
                QMessageBox.warning(self, "失败", "无法保存模型配置。请检查日志。")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存模型配置时出错: {e}")
            logging.error(f"Error saving model settings: {e}")


# --- Main Application Window ---

class MainWindow(QMainWindow):
    """Main application window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("玫瑰花瓣病害检测系统")
        self.setMinimumSize(1200, 800)
        self.setObjectName("mainWindow")
        if os.path.exists(LOGO_FILE):
            self.setWindowIcon(QIcon(LOGO_FILE))

        self.config_manager = ConfigManager(MODEL_CONFIG_FILE, COLORS_CONFIG_FILE)
        self.infer_instance: Optional[InferenceRoseDisease] = None
        self.worker: Optional[ProcessingWorker] = None # Hold the current worker thread

        if not self._init_model():
            # Optionally disable processing features if model init fails critically
            # For now, just log and show message. App will continue.
             logging.critical("Model initialization failed. Processing may not work.")
             # QMessageBox.critical(self, "初始化失败", "无法初始化核心模型，请检查配置和权重文件。")


        self._init_ui()
        self._load_styles()


    def _init_model(self) -> bool:
        """Initializes the core InferenceRoseDisease model."""
        logging.info("Initializing inference model...")
        config = self.config_manager.get_model_config()
        required_files = [
            config["cfg_file"],
            config["crop_weight"],
            config["petal_weights_file"],
            config["diease_weights_file"]
        ]

        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            error_msg = f"模型初始化失败：以下必要文件不存在:\n" + "\n".join(missing_files)
            logging.error(error_msg)
            QMessageBox.critical(self, "文件缺失", error_msg)
            return False

        try:
            self.infer_instance = InferenceRoseDisease(
                cfg_file=config["cfg_file"],
                petal_encoder_name=config["petal_encoder_name"],
                petal_decoder_name=config["petal_decoder_name"],
                disease_encoder_name=config["disease_encoder_name"],
                disease_decoder_name=config["disease_decoder_name"],
                crop_weight=config["crop_weight"],
                petal_weights_file=config["petal_weights_file"],
                diease_weights_file=config["diease_weights_file"],
                box_threshold=config["box_threshold"],
                mask_threshold=config["mask_threshold"]
            )
            logging.info("Inference model initialized successfully.")
            self.statusBar().showMessage("模型已加载", 3000)
            return True
        except Exception as e:
            error_msg = f"模型初始化时发生意外错误: {e}\n{traceback.format_exc()}"
            logging.error(error_msg)
            QMessageBox.critical(self, "初始化错误", f"模型初始化失败，请检查配置或日志。\n错误: {e}")
            return False

    def _reload_model(self):
        """Reloads the model after configuration changes."""
        logging.info("Reloading model due to configuration change...")
        self.statusBar().showMessage("正在重新加载模型...")
        QApplication.processEvents() # Update UI
        if self._init_model():
             self.statusBar().showMessage("模型已根据新配置重新加载", 5000)
        else:
             self.statusBar().showMessage("模型重新加载失败，请检查配置", 5000)


    def _init_ui(self):
        """Initializes the main UI structure (tabs, status bar)."""
        self.tabs = QTabWidget()
        self.tabs.setObjectName("mainTabs")
        self.setCentralWidget(self.tabs)

        # Create Tab Widgets
        self.home_tab = HomeTab()
        self.config_tab = ConfigTab(self.config_manager)

        # Add Tabs
        self.tabs.addTab(self.home_tab, "图片处理与分析")
        self.tabs.addTab(self.config_tab, "系统配置")

        # Status Bar
        self.statusBar().showMessage("就绪")
        self.statusBar().setObjectName("statusBar")

        # Connect Signals
        self.home_tab.start_processing_requested.connect(self._start_processing)
        self.config_tab.model_config_changed.connect(self._reload_model)


    def _load_styles(self):
        """Loads fonts and stylesheets."""
        # Load Font
        if os.path.exists(FONT_FILE):
             font_id = QFontDatabase.addApplicationFont(FONT_FILE)
             if font_id != -1:
                  font_families = QFontDatabase.applicationFontFamilies(font_id)
                  if font_families:
                       # Apply the first loaded font family
                       self.setFont(QtGui.QFont(font_families[0]))
                       logging.info(f"Loaded and applied font: {font_families[0]}")
             else:
                  logging.warning(f"Could not load font file: {FONT_FILE}")

        # Load Stylesheet
        if os.path.exists(STYLE_FILE):
            try:
                with open(STYLE_FILE, "r", encoding="utf-8") as f:
                    self.setStyleSheet(f.read())
                logging.info(f"Loaded stylesheet: {STYLE_FILE}")
            except Exception as e:
                logging.error(f"Failed to load stylesheet {STYLE_FILE}: {e}")


    @QtCore.pyqtSlot(object) # Accepts the path (str) from HomeTab signal
    def _start_processing(self, input_path: str):
        """Starts the processing worker thread."""
        if not self.infer_instance:
            QMessageBox.critical(self, "模型未就绪", "无法开始处理，因为模型未能成功初始化。")
            self.home_tab.show_error("模型未初始化") # Update HomeTab UI state
            return

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "处理中", "当前有任务正在处理，请等待其完成后再开始新的任务。")
            return

        is_batch = os.path.isdir(input_path)
        file_arg = input_path if not is_batch else None
        dir_arg = input_path if is_batch else None

        self.statusBar().showMessage("开始处理...")
        logging.info(f"Starting processing: {'Batch' if is_batch else 'Single file'} - {input_path}")

        # Get current color config for the worker
        current_colors = self.config_manager.get_colors_config()

        # Create and start the worker
        self.worker = ProcessingWorker(
            infer_instance=self.infer_instance,
            colors=current_colors,
            file_path=file_arg,
            directory=dir_arg
        )

        # Connect worker signals to HomeTab slots
        self.worker.update_progress.connect(self.home_tab.update_progress)
        self.worker.finished.connect(self.home_tab.display_results)
        self.worker.error.connect(self.home_tab.show_error)
        # Also connect finished/error to update status bar
        self.worker.finished.connect(lambda: self.statusBar().showMessage("处理完成", 5000))
        self.worker.error.connect(lambda msg: self.statusBar().showMessage(f"处理失败: {msg[:50]}...", 5000))


        self.worker.start()

    def closeEvent(self, event: QtGui.QCloseEvent):
        """Ensures worker thread is stopped before closing."""
        if self.worker and self.worker.isRunning():
             reply = QMessageBox.question(self, '退出确认',
                                          "当前有处理任务正在进行中，确定要退出吗？",
                                          QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
             if reply == QMessageBox.Yes:
                  logging.info("Terminating worker thread on close.")
                  self.worker.quit() # Ask thread to stop nicely
                  self.worker.wait(2000) # Wait up to 2 seconds for it to finish
                  if self.worker.isRunning(): # Force terminate if still running
                     logging.warning("Worker thread did not quit gracefully, terminating.")
                     self.worker.terminate()
                     self.worker.wait() # Wait for termination
                  event.accept()
             else:
                  event.ignore()
        else:
             event.accept()


# --- Main Execution ---

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())