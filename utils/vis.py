import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
def hex_to_bgr(hex_color):
    """Convert hexadecimal color to BGR format
    
    Args:
        hex_color: Hexadecimal color string, e.g. "#FF0000" for red
        
    Returns:
        BGR color tuple, e.g. (0, 0, 255) for red
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV使用BGR格式

def merge_image_with_mask(image_dir, 
                        mask_dir, 
                        color_hex,
                        alpha=0.8):
    """Merge image with mask to create semi-transparent overlay effect
    
    Args:
        image_dir: Path to the original image directory
        mask_dir: Path to the mask image directory
        color_hex: Mask color in hexadecimal format, e.g. "#FF0000" for red
        alpha: Transparency, default 0.8
        
    Returns:
        Base64 encoded string of the merged image
    """
    # 获取目录下所有图片
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    # 根据文件名中的数字进行排序
    img_files_sorted = sorted(image_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    mask_files_sorted = sorted(mask_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    # Ensure image and mask counts match
    assert len(image_files) == len(mask_files), "Number of images and masks do not match"
    
    # 创建子图，行数为图片数量
    fig, axes = plt.subplots(len(image_files), 2, figsize=(12, 6*len(image_files)))
    
    # 处理单张图片的情况
    if len(image_files) == 1:
        axes = axes.reshape(1, -1)
    
    # 将十六进制颜色转换为BGR格式
    color = hex_to_bgr(color_hex)
    
    for idx, (img_file, mask_file) in enumerate(zip(img_files_sorted, mask_files_sorted)):
        # 读取原始图像和掩码
        img = cv2.imread(os.path.join(image_dir, img_file))
        if img is None:
            raise ValueError(f"无法读取图像文件: {img_file}")
            
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"无法读取掩码文件: {mask_file}")
        
        # 调整掩码大小以匹配图像
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # 创建彩色掩码
        colored_mask = np.zeros_like(img)
        colored_mask[mask > 0] = color
        
        # 合并图像和掩码
        result = cv2.addWeighted(colored_mask, alpha, img, 1 - alpha, 0)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        
        # 显示原始图像和结果
        axes[idx, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx, 0].set_title(f'original_image_{idx+1}')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(result)
        axes[idx, 1].set_title(f'after_mask_{idx+1}')
        axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.show()