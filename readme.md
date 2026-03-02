# PetalSpot Pro - Intelligent Petal Disease Analysis System

## 📖 Project Overview

**PetalSpot Pro** is an intelligent petal disease analysis system based on deep learning, designed for the detection and quantitative analysis of rose petal diseases. Combining object detection (TridentNet) and semantic segmentation technologies, this system can automatically identify petals from images, segment disease areas, and accurately calculate the disease area ratio, providing a powerful tool for plant pathology research and agricultural production.

## ✨ Key Features

- **Intelligent Detection & Cropping**: Utilizes the TridentNet object detection model to automatically locate and crop petal regions in images.
- **High-Precision Segmentation**: Employs advanced segmentation networks (RegNet + U-Net/Manet) to extract petal masks and disease area masks respectively.
- **Quantitative Analysis**:
  - Automatic calculation of petal area (pixels).
  - Automatic calculation of disease area (pixels).
  - Precise output of disease ratio (%).
- **Batch Processing**: Supports single image analysis and batch folder processing, significantly improving efficiency.
- **Visualized Results**: Provides result images with detection boxes, segmentation masks, and data annotations.
- **Data Export**: Supports exporting analysis results to CSV tables for subsequent statistical analysis.
- **User-Friendly Interface**: A graphical interface developed based on PyQt5, supporting full-screen viewing, zooming, panning, and other operations.

## 🖼️ Software Screenshot

![Software Interface](pic/software.png)

*The main interface shows image loading, analysis control, and result preview functions.*

## 🧠 Model Architecture

This project adopts a cascaded deep learning processing flow:
1. **Stage 1 (Object Detection)**: Uses TridentNet to quickly locate petal positions.
2. **Stage 2 (Semantic Segmentation)**: Performs fine segmentation on the cropped petals to extract petal contours and disease areas.

![Model Architecture](pic/model.png)

## 🛠️ Installation & Usage

### Requirements
- Python 3.8+
- PyTorch (GPU version recommended)
- PyQt5
- OpenCV
- Detectron2 (Included in the project or needs separate installation)

### Installation
Ensure the following core dependencies are installed:
```bash
pip install torch torchvision
pip install opencv-python
pip install PyQt5
pip install pyyaml
# Other project-specific dependencies...
```

### Usage
Run `app.py` in the project root directory to start the application:

```bash
python app.py
```

## 📂 Directory Structure

```
f:\2025\Petal_diseasa-master\
├── app.py              # Application entry point (GUI main program)
├── configs/            # Model and application configuration files
├── det/                # TridentNet related detection code
├── detectron2/         # Detectron2 core library
├── pic/                # Documentation image resources
│   ├── model.png       # Model architecture diagram
│   └── software.png    # Software screenshot
├── utils/              # Utility scripts (e.g., inference class InferenceRoseDisease)
└── readme.md           # Project documentation
```

## 📝 Configuration
The system supports customizing model paths and display colors via the interface or configuration files:
- **Model Configuration**: Modify model weight paths and threshold parameters in the "System Settings" tab.
- **Display Configuration**: Customize the display color and transparency of petal and disease masks.

---
*PetalSpot Pro - Empowering Smart Agriculture and Plant Disease Research*
