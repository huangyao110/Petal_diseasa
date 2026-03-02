import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def hex_to_bgr(hex_color):
    """将十六进制颜色转换为BGR格式
    
    参数:
        hex_color: 十六进制颜色字符串，例如 "#FF0000" 表示红色
        
    返回:
        BGR颜色元组，例如 (0, 0, 255) 表示红色
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV使用BGR格式

def _process_image_mask(img, mask, color_hex, alpha=0.8):
    """处理单个图像和掩码的通用函数
    
    参数:
        img: 原始图像
        mask: 掩码图像
        color_hex: 十六进制颜色字符串
        alpha: 透明度，默认0.8
        
    返回:
        处理后的图像
    """
    # 调整掩码大小以匹配图像
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    # 将十六进制颜色转换为BGR格式
    color = hex_to_bgr(color_hex)
    
    # 创建彩色掩码
    colored_mask = np.zeros_like(img)
    colored_mask[mask > 0] = color
    
    # 合并图像和掩码
    result = cv2.addWeighted(colored_mask, alpha, img, 1 - alpha, 0)
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

def merge_image_with_mask(image_dir=None, 
                        mask_dir=None, 
                        color_hex="#FF0000",
                        alpha=0.8,
                        image_file=None,
                        mask_file=None,
                        save_dir=None):
    """合并图像与掩码创建半透明覆盖效果
    
    参数:
        image_dir: 原始图像目录路径 demo\demo1\DSC01133_4.jpg
        mask_dir: 掩码图像目录路径  demo\demo1\disease\DSC01133_1_disease_mask.png
        color_hex: 十六进制格式的掩码颜色，例如 "#FF0000" 表示红色
        alpha: 透明度，默认0.8
        image_file: 单个图像文件路径 
        mask_file: 单个掩码文件路径
        
    返回:
        无，直接显示处理后的图像
    """
    # 检查参数有效性
    if (image_dir is None and image_file is None) or (mask_dir is None and mask_file is None):
        raise ValueError("必须提供图像和掩码的路径（目录或文件）")
    
    # 处理单个文件的情况
    if image_file is not None and mask_file is not None:
        # 读取原始图像和掩码
        img = cv2.imread(image_file)
        if img is None:
            raise ValueError(f"无法读取图像文件: {image_file}")
        
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取掩码文件: {mask_file}")
        
        # 处理图像和掩码
        result = _process_image_mask(img, mask, color_hex, alpha)
        
        # 显示原始图像和结果
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title('original_image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(result)
        plt.title('after_mask')
        plt.axis('off')
        
        plt.show()
        return
    
    # 处理目录的情况
    # 获取目录下所有图片
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # 根据文件名中的数字进行排序
    try:
        img_files_sorted = sorted(image_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
        mask_files_sorted = sorted(mask_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    except (IndexError, ValueError):
        # 如果排序失败，使用原始排序
        img_files_sorted = image_files
        mask_files_sorted = mask_files
    
    # 确保图像和掩码数量匹配
    if len(img_files_sorted) != len(mask_files_sorted):
        raise ValueError(f"图像数量({len(img_files_sorted)})和掩码数量({len(mask_files_sorted)})不匹配")
    
    # 创建子图，行数为图片数量
    fig, axes = plt.subplots(len(img_files_sorted), 2, figsize=(12, 6*len(img_files_sorted)))
    
    # 处理单张图片的情况
    if len(img_files_sorted) == 1:
        axes = np.array([axes])
    
    for idx, (img_file, mask_file) in enumerate(zip(img_files_sorted, mask_files_sorted)):
        # 读取原始图像和掩码
        img = cv2.imread(os.path.join(image_dir, img_file))
        if img is None:
            raise ValueError(f"无法读取图像文件: {img_file}")
            
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取掩码文件: {mask_file}")
        
        # 处理图像和掩码
        result = _process_image_mask(img, mask, color_hex, alpha)
        # 保存结果
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"merged_{idx+1}.png")
            cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        
        # 显示原始图像和结果
        axes[idx, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f'original_image_{idx+1}')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(result)
        axes[idx, 1].set_title(f'after_mask_{idx+1}')
        axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.show()