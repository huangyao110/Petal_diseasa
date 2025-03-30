import os
import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, Response, session
from werkzeug.utils import secure_filename
from inference import InferenceRoseDisease, ImageVisualizer
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rose_petal_disease_detection'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传文件大小为16MB
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['CONFIG_FILE'] = 'model_config.json'
app.config['COLORS_FILE'] = 'colors_config.json'

# 确保上传和结果目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

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

# 加载配置
def load_config():
    if os.path.exists(app.config['CONFIG_FILE']):
        try:
            with open(app.config['CONFIG_FILE'], 'r') as f:
                return json.load(f)
        except:
            return DEFAULT_CONFIG

# 加载颜色配置
def load_colors():
    if os.path.exists(app.config['COLORS_FILE']):
        try:
            with open(app.config['COLORS_FILE'], 'r') as f:
                return json.load(f)
        except:
            return DEFAULT_COLORS
    return DEFAULT_COLORS

# 保存配置
def save_config(config):
    with open(app.config['CONFIG_FILE'], 'w') as f:
        json.dump(config, f, indent=4)

# 保存颜色配置
def save_colors(colors):
    with open(app.config['COLORS_FILE'], 'w') as f:
        json.dump(colors, f, indent=4)

# 初始化模型
def init_model():
    config = load_config()
    return InferenceRoseDisease(
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

# 全局变量存储模型实例
infer = init_model()
visualizer = ImageVisualizer()

# 将十六进制颜色转换为BGR格式
def hex_to_bgr(hex_color):
    """将十六进制颜色转换为BGR格式
    
    Args:
        hex_color: 十六进制颜色字符串，例如"#FF0000"表示红色
        
    Returns:
        BGR格式的颜色元组，例如(0, 0, 255)表示红色
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV使用BGR格式

# 合并图像和掩码的函数
def merge_image_with_mask(image_path, mask_path, color_hex, alpha=0.4):
    """将图像与掩码合并，创建半透明叠加效果
    
    Args:
        image_path: 原始图像路径
        mask_path: 掩码图像路径
        color_hex: 掩码颜色，十六进制格式的字符串，例如"#FF0000"表示红色
        alpha: 透明度，默认0.4
        
    Returns:
        合并后的图像数据的base64编码
    """
    # 读取原始图像和掩码
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # 调整掩码大小以匹配图像
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # 将十六进制颜色转换为BGR格式
    color = hex_to_bgr(color_hex)
    
    # 创建彩色掩码
    colored_mask = np.zeros_like(img)
    colored_mask[mask > 0] = color
    
    # 合并图像和掩码
    result = cv2.addWeighted(colored_mask, alpha, img, 1 - alpha, 0)
    
    # 将结果转换为base64编码
    _, buffer = cv2.imencode('.png', result)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    return img_str

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/config')
def config():
    # 加载当前配置
    config = load_config()
    colors = load_colors()
    return render_template('config.html', config=config, colors=colors)

@app.route('/save_config', methods=['POST'])
def save_config_route():
    # 获取表单数据
    config = {
        "cfg_file": request.form.get('cfg_file'),
        "crop_weight": request.form.get('crop_weight'),
        "petal_encoder_name": request.form.get('petal_encoder_name'),
        "petal_decoder_name": request.form.get('petal_decoder_name'),
        "disease_encoder_name": request.form.get('disease_encoder_name'),
        "disease_decoder_name": request.form.get('disease_decoder_name'),
        "petal_weights_file": request.form.get('petal_weights_file'),
        "diease_weights_file": request.form.get('diease_weights_file'),
        "box_threshold": float(request.form.get('box_threshold')),
        "mask_threshold": float(request.form.get('mask_threshold'))
    }
    
    # 保存配置
    save_config(config)
    flash('模型配置已保存，重启应用后生效')
    return redirect(url_for('config'))

@app.route('/save_colors', methods=['POST'])
def save_colors_route():
    # 获取表单数据
    colors = {
        "petal_color": request.form.get('petal_color'),
        "disease_color": request.form.get('disease_color'),
        "alpha": float(request.form.get('alpha'))
    }
    
    # 保存颜色配置
    save_colors(colors)
    flash('颜色设置已保存')
    return redirect(url_for('config'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('没有文件部分')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('没有选择文件')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        # 加载颜色配置
        colors = load_colors()
        petal_color = colors.get('petal_color', DEFAULT_COLORS['petal_color'])
        disease_color = colors.get('disease_color', DEFAULT_COLORS['disease_color'])
        alpha = float(colors.get('alpha', DEFAULT_COLORS['alpha']))
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # 创建结果目录
        result_dir = os.path.join(app.config['RESULT_FOLDER'], os.path.splitext(filename)[0])
        os.makedirs(result_dir, exist_ok=True)
        
        # 执行步骤1：裁剪图像
        infer.step1_crop_img(img_file=file_path, save_dir=result_dir)
        
        # 获取裁剪后的图像列表
        cropped_images = [f for f in os.listdir(result_dir) if f.endswith('.jpg')]
        
        # 执行步骤2和3：花瓣分割和病害分割
        petal_dir = os.path.join(result_dir, 'petal_seg')
        disease_dir = os.path.join(result_dir, 'petal_disease')
        os.makedirs(petal_dir, exist_ok=True)
        os.makedirs(disease_dir, exist_ok=True)
        
        for img_file in cropped_images:
            img_path = os.path.join(result_dir, img_file)
            infer.step2_seg_petal(img_file=img_path, save_dir=petal_dir)
            infer.step3_seg_disease(img_file=img_path, save_dir=disease_dir)
        
        # 创建合并图像的目录
        merged_dir = os.path.join(result_dir, 'merged')
        os.makedirs(merged_dir, exist_ok=True)
        
        # 准备结果数据
        results = []
        for i, img_file in enumerate(cropped_images):
            # 原始裁剪图像
            img_path = os.path.join(result_dir, img_file)
            with open(img_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
            
            # 花瓣分割结果
            petal_mask_file = f"{os.path.splitext(img_file)[0]}_petal_mask.png"
            petal_mask_path = os.path.join(petal_dir, petal_mask_file)
            with open(petal_mask_path, 'rb') as f:
                petal_mask_data = base64.b64encode(f.read()).decode('utf-8')
            
            # 病害分割结果
            disease_mask_file = f"{os.path.splitext(img_file)[0]}_disease_mask.png"
            disease_mask_path = os.path.join(disease_dir, disease_mask_file)
            with open(disease_mask_path, 'rb') as f:
                disease_mask_data = base64.b64encode(f.read()).decode('utf-8')
            
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
            petal_merged_data = merge_image_with_mask(img_path, petal_mask_path, petal_color, alpha)  # 使用自定义颜色
            
            # 保存合并后的花瓣图像
            petal_merged_img = base64.b64decode(petal_merged_data)
            with open(petal_merged_path, 'wb') as f:
                f.write(petal_merged_img)
            
            # 合并病害掩码和原图
            disease_merged_file = f"{os.path.splitext(img_file)[0]}_disease_merged.png"
            disease_merged_path = os.path.join(merged_dir, disease_merged_file)
            disease_merged_data = merge_image_with_mask(img_path, disease_mask_path, disease_color, alpha)  # 使用自定义颜色
            
            # 保存合并后的病害图像
            disease_merged_img = base64.b64decode(disease_merged_data)
            with open(disease_merged_path, 'wb') as f:
                f.write(disease_merged_img)
            
            # 添加结果数据
            results.append({
                'id': i + 1,
                'image': img_data,
                'petal_mask': petal_mask_data,
                'disease_mask': disease_mask_data,
                'petal_merged': petal_merged_data,
                'disease_merged': disease_merged_data,
                'petal_merged_path': petal_merged_path,
                'disease_merged_path': disease_merged_path,
                'disease_ratio': f"{disease_ratio:.2f}%",
                'disease_ratio_raw': disease_ratio
            })
        
        # 保存病害比例数据到CSV文件
        csv_file = os.path.join(result_dir, f"{os.path.splitext(filename)[0]}_disease_ratio.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['花朵ID', '病害面积比例(%)', '花瓣面积(像素)', '病害面积(像素)'])
            for i, result in enumerate(results):
                writer.writerow([i + 1, f"{result['disease_ratio_raw']:.2f}", 
                                petal_area, disease_area])
        
        # 将当前使用的颜色配置保存到会话中，以便在结果页面显示
        session['petal_color'] = petal_color
        session['disease_color'] = disease_color
        session['alpha'] = alpha
        
        # 在return前确保所有数据已添加到results中
        # results.append()代码应该在前面的处理逻辑中，而不是在return后面
        return render_template('results.html', 
                              filename=filename, 
                              results=results, 
                              petal_color=petal_color,
                              disease_color=disease_color,
                              alpha=alpha)
    
    flash('不允许的文件类型')
    return redirect(request.url)

@app.route('/batch', methods=['GET', 'POST'])
def batch_process():
    if request.method == 'POST':
        if 'directory' not in request.form:
            flash('没有指定目录')
            return redirect(request.url)
        
        directory = request.form['directory']
        if not os.path.isdir(directory):
            flash('指定的目录不存在')
            return redirect(request.url)
        
        # 获取目录中的所有图像文件
        image_files = [f for f in os.listdir(directory) 
                      if os.path.isfile(os.path.join(directory, f)) and 
                      f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 创建批处理结果目录
        batch_result_dir = os.path.join(app.config['RESULT_FOLDER'], 'batch_' + os.path.basename(directory))
        os.makedirs(batch_result_dir, exist_ok=True)
        
        # 处理每个图像
        results_summary = []
        for img_file in image_files:
            img_path = os.path.join(directory, img_file)
            img_result_dir = os.path.join(batch_result_dir, os.path.splitext(img_file)[0])
            os.makedirs(img_result_dir, exist_ok=True)
            
            # 执行步骤1：裁剪图像
            infer.step1_crop_img(img_file=img_path, save_dir=img_result_dir)
            
            # 获取裁剪后的图像列表
            cropped_images = [f for f in os.listdir(img_result_dir) if f.endswith('.jpg')]
            
            # 执行步骤2和3：花瓣分割和病害分割
            petal_dir = os.path.join(img_result_dir, 'petal_seg')
            disease_dir = os.path.join(img_result_dir, 'petal_disease')
            os.makedirs(petal_dir, exist_ok=True)
            os.makedirs(disease_dir, exist_ok=True)
            
            for crop_img in cropped_images:
                crop_path = os.path.join(img_result_dir, crop_img)
                infer.step2_seg_petal(img_file=crop_path, save_dir=petal_dir)
                infer.step3_seg_disease(img_file=crop_path, save_dir=disease_dir)
            
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
                'disease_ratio': f"{disease_ratio:.2f}%"
            })
        
        return render_template('batch_results.html', results=results_summary, directory=directory)
    
    return render_template('batch.html')

@app.route('/download_image/<path:image_path>')
def download_image(image_path):
    """下载指定路径的图像"""
    try:
        return send_file(image_path, as_attachment=True)
    except Exception as e:
        flash(f'下载图像失败: {str(e)}')
        return redirect(url_for('index'))

@app.route('/download_csv/<filename>')
def download_csv(filename):
    """下载病害指数CSV文件"""
    try:
        # 获取结果目录
        result_dir = os.path.join(app.config['RESULT_FOLDER'], os.path.splitext(filename)[0])
        
        # 创建临时文件
        temp_file = os.path.join(tempfile.gettempdir(), f"{filename}_disease_index.csv")
        
        # 获取所有裁剪图像
        cropped_images = [f for f in os.listdir(result_dir) if f.endswith('.jpg')]
        petal_dir = os.path.join(result_dir, 'petal_seg')
        disease_dir = os.path.join(result_dir, 'petal_disease')
        
        # 写入CSV文件
        with open(temp_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['花朵ID', '花瓣面积', '病害面积', '病害比例(%)'])
            
            for img_file in cropped_images:
                base_name = os.path.splitext(img_file)[0]
                petal_mask_path = os.path.join(petal_dir, f"{base_name}_petal_mask.png")
                disease_mask_path = os.path.join(disease_dir, f"{base_name}_disease_mask.png")
                
                if os.path.exists(petal_mask_path) and os.path.exists(disease_mask_path):
                    petal_mask = cv2.imread(petal_mask_path, cv2.IMREAD_GRAYSCALE)
                    disease_mask = cv2.imread(disease_mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    petal_area = np.sum(petal_mask > 0)
                    disease_area = np.sum(disease_mask > 0)
                    
                    if petal_area > 0:
                        disease_ratio = disease_area / petal_area * 100
                    else:
                        disease_ratio = 0
                    
                    writer.writerow([base_name, petal_area, disease_area, f"{disease_ratio:.2f}"])
        
        return send_file(temp_file, as_attachment=True, download_name=f"{filename}_disease_index.csv")
    except Exception as e:
        flash(f'下载CSV文件失败: {str(e)}')
        return redirect(url_for('index'))

@app.route('/download_batch_csv/<directory>')
def download_batch_csv(directory):
    """下载批处理结果的CSV文件"""
    try:
        # 获取批处理结果目录
        batch_result_dir = os.path.join(app.config['RESULT_FOLDER'], 'batch_' + directory)
        
        # 创建临时文件
        temp_file = os.path.join(tempfile.gettempdir(), f"batch_{directory}_disease_index.csv")
        
        # 获取所有图像目录
        image_dirs = [d for d in os.listdir(batch_result_dir) if os.path.isdir(os.path.join(batch_result_dir, d))]
        
        # 写入CSV文件
        with open(temp_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['图像名称', '花朵数量', '总花瓣面积', '总病害面积', '病害比例(%)'])
            
            for img_dir in image_dirs:
                img_result_dir = os.path.join(batch_result_dir, img_dir)
                cropped_images = [f for f in os.listdir(img_result_dir) if f.endswith('.jpg')]
                petal_dir = os.path.join(img_result_dir, 'petal_seg')
                disease_dir = os.path.join(img_result_dir, 'petal_disease')
                
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
                
                writer.writerow([img_dir, len(cropped_images), total_petal_area, total_disease_area, f"{disease_ratio:.2f}"])
        
        return send_file(temp_file, as_attachment=True, download_name=f"batch_{directory}_disease_index.csv")
    except Exception as e:
        flash(f'下载CSV文件失败: {str(e)}')
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)