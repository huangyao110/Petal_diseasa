import os
from pathlib import Path

def get_per_img_path(directory, suffix=None):
    if suffix is None:
        img_extensions = ('.jpg', '.jpeg', '.png')
    else:
        img_extensions = (suffix,)
    return [os.path.join(directory, file) for file in os.listdir(directory) 
            if file.lower().endswith(img_extensions)]

def sort_by_name(img_paths, mask_paths):
    sorted_masks = []
    unmatched_imgs = []
    
    for img_path in img_paths:
        base_name = os.path.splitext(os.path.basename(img_path))[0].split("_disease")[0]
        matched = False
        
        for mask_path in mask_paths:
            if base_name in mask_path:
                sorted_masks.append(mask_path)
                matched = True
                break
                
        if not matched:
            unmatched_imgs.append(img_path)
    if unmatched_imgs:
        print(f"警告: {len(unmatched_imgs)} 个图像没有找到对应的掩码")
    matched_imgs = [img for img in img_paths if img not in unmatched_imgs]
    return matched_imgs, sorted_masks
                
def return_sorted_img_path(img_dir, mask_dir):
    if not os.path.isdir(img_dir):
        raise ValueError(f"图像目录不存在: {img_dir}")
    if not os.path.isdir(mask_dir):
        raise ValueError(f"掩码目录不存在: {mask_dir}")
    img_paths = get_per_img_path(img_dir)
    mask_paths = get_per_img_path(mask_dir)
    if not img_paths:
        raise ValueError(f"在目录 {img_dir} 中没有找到图像文件")
    if not mask_paths:
        raise ValueError(f"在目录 {mask_dir} 中没有找到掩码文件")
    return sort_by_name(img_paths, mask_paths)


def get_all_image_from_dir(img_dir):
    """递归获取目录下所有图片文件"""
    
    # 支持的图片格式
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.JPG')
    
    # 存储所有图片路径
    all_images = []
    
    # 递归遍历目录
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            # 检查文件扩展名
            if file.lower().endswith(img_extensions):
                # 使用Path构建完整路径并添加到列表
                img_path = str(Path(root) / file)
                all_images.append(img_path)
                
    return all_images


def move_sec2dir(img_lst, target_dir):
    for img_path in img_lst:
        # 获取文件名
        file_name = os.path.basename(img_path)
        
        # 构建目标路径
        target_path = os.path.join(target_dir, file_name)
        
        # 如果目标文件已存在则跳过
        if os.path.exists(target_path):
            print(f"Skip {img_path}, target file already exists: {target_path}")
            continue
            
        # 移动文件
        os.rename(img_path, target_path)
        
        print(f"Moved {img_path} to {target_path}")