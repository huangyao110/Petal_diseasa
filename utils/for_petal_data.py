import os
import json
import cv2
import numpy as np
from PIL import Image


def move_file(src, dst):
    json_files = [f for f in os.listdir(src) if f.endswith('.json')]
    png_files = [f for f in os.listdir(src) if f.endswith('.png') and \
                 f'{f.split(".")[0]}.json' in json_files]
    
    for j,m in zip(json_files, png_files):
        os.rename(f'{src}/{j}', f'{dst}/{j}')
        os.rename(f'{src}/{m}', f'{dst}/{m}')


def get_reference_mask(json_file, ref_image, label, pixel_number):
    # Create a copy of the reference image to avoid modifying the original
    mask_init = np.zeros((ref_image.size[1], ref_image.size[0], 3), dtype=np.uint8)
    with open(json_file) as f:
        anna_mask = json.load(f)
    
    # 检查json文件中是否包含指定的label
    labels = [shape['label'] for shape in anna_mask['shapes']]
    if label not in labels:
        return mask_init
    
    points = [np.array(shape['points'], dtype=np.int32) 
             for shape in anna_mask['shapes'] 
             if shape['label'] == label]
    
    # Create separate masks for disease and petal
    disease_mask = np.zeros_like(mask_init)
    if points:
        cv2.fillPoly(disease_mask, points, (pixel_number, pixel_number, pixel_number))

    return disease_mask

def resize_mask_img(img, target_size):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    return img.resize(target_size, Image.Resampling.BILINEAR)

def save_mask_img(img_path,
                label, 
                pixel_number, 
                resize=True, 
                maskorimg='image',
                json_file=None, 
                dir='data/Images'):
    """Save processed image or mask to specified directory.
    
    Args:
        img_path (str): Path to input image
        resize (bool): Whether to resize output
        maskorimg (str): 'mask' or 'image' to specify output type
        json_file (str): Path to annotation file (required for mask)
        dir (str): Output directory path
    """
    # Create output directory if it doesn't exist
    os.makedirs(dir, exist_ok=True)
    
    # Load and process image
    with Image.open(img_path) as img:
        target_size = (640, 640) if resize else img.size
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        
        if maskorimg == 'mask':
            if not json_file:
                raise ValueError("json_file is required when generating mask")
                
            # Generate and process mask
            mask = get_reference_mask(json_file, img, label=label, pixel_number=pixel_number)
            if isinstance(mask, np.ndarray):
                mask = Image.fromarray(mask)
            if np.max(mask) == 0:
                print(f"Warnning: file {json_file} not find label '{label}', \
                    the file {json_file} not save.")
                return
            processed_img = resize_mask_img(mask, target_size)
            output_path = os.path.join(dir, f"{file_name}.png")
        else:
            # Process regular image
            processed_img = resize_mask_img(img, target_size)
            output_path = os.path.join(dir, f"{file_name}.jpg")
            
        # Save processed image
        processed_img.save(output_path)


def split_dataset(train_dir, val_dir, val_number=200):
    """Split dataset into train, val, and test sets.

    Args:
        img_dir (str): Path to the image directory
        json_dir (str): Path to the JSON annotations directory
        save_dir (str): Path to save the split datasets
        img_suffix (str): Image file suffix
    """
    # Create output directories if they don't exist
    t_img = os.path.join(train_dir, 'img')
    t_gt = os.path.join(train_dir, 'gt')
    v_img = os.path.join(val_dir, 'img')
    v_gt = os.path.join(val_dir, 'gt')
    
    # 确保目标目录存在
    os.makedirs(v_img, exist_ok=True)
    os.makedirs(v_gt, exist_ok=True)
    
    # 获取训练集图片和标注文件
    t_img_files = [os.path.join(t_img, f) for f in os.listdir(t_img) if f.endswith('.jpg')]
    t_gt_files = [os.path.join(t_gt, f) for f in os.listdir(t_gt) if f.endswith('.png')]
    
    # 检查文件数量
    if len(t_img_files) < val_number:
        raise ValueError(f"训练集图片数量({len(t_img_files)})小于要求的验证集数量({val_number})")
    
    # 随机打乱并选择验证集图片
    np.random.seed(42)  # 设置随机种子保证可重复性
    t_img_files = np.array(t_img_files)
    np.random.shuffle(t_img_files)
    choose_v_img = t_img_files[:val_number]
    
    # 获取对应的标注文件
    choose_v_gt = []
    for img_path in choose_v_img:
        gt_path = img_path.replace(t_img, t_gt).replace('.jpg', '.png')
        if gt_path in t_gt_files:
            choose_v_gt.append(gt_path)
        else:
            print(f"警告: 未找到图片{img_path}对应的标注文件")
            
    # 移动文件到验证集目录
    for v_im, v_g in zip(choose_v_img, choose_v_gt):
        try:
            os.rename(v_im, v_im.replace(t_img, v_img))
            os.rename(v_g, v_g.replace(t_gt, v_gt))
        except OSError as e:
            print(f"移动文件失败: {e}")

if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Process images and generate masks or split dataset')
    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-commands')

    # Subcommand: process
    process_parser = subparsers.add_parser('process', help='Process images and generate masks')
    process_parser.add_argument('--img_path', type=str, default='train', required=True,
                        help='Path to the image directory')
    process_parser.add_argument('--json_path', type=str, default='train',
                        help='Path to the JSON annotations directory')
    process_parser.add_argument('--save_path', type=str, default='petal_data/Annotations', required=True,
                        help='Path to save the generated masks')
    process_parser.add_argument('--maskorimg', type=str, default='mask', required=True,
                        help='Whether to generate masks or images')
    process_parser.add_argument('--resize', type=bool, default=False, required=True,
                        help='Whether to resize masks or images')
    process_parser.add_argument('--label', type=str, default='petal', required=True,
                        help='label to mask ')
    process_parser.add_argument('--pixel_number', type=float, default=255, required=True,
                        help='pixel number to mask ')
    process_parser.add_argument('--img_suffix', type=str, default='png', required=True,
                        help='Whether to move files')

    # Subcommand: split
    split_parser = subparsers.add_parser('split', help='Split dataset into train/val')
    split_parser.add_argument('--train_dir', type=str, required=True, help='Path to train directory')
    split_parser.add_argument('--val_dir', type=str, required=True, help='Path to val directory')
    split_parser.add_argument('--val_number', type=int, default=200, help='Number of images for validation set')

    args = parser.parse_args()

    if args.command == 'process':
        img_suffix = args.img_suffix

        os.makedirs(args.save_path, exist_ok=True)
        json_files = [os.path.join(args.json_path, f) for f in os.listdir(args.json_path) if f.endswith('.json')]
        img_files = []
        for f in json_files[:]:  # 使用切片创建副本以避免在迭代时修改列表
            img_path = os.path.join(args.img_path, f"{os.path.basename(f).split('.')[0]}.{img_suffix}")
            if not os.path.exists(img_path):
                print(f'Image {img_path} does not exist, remove {f}')
                json_files.remove(f)
            else:
                img_files.append(img_path)
        print(f'Number of images: {len(img_files)}')
        print(f'Number of JSON files: {len(json_files)}')

        if args.maskorimg == 'mask':
            assert len(img_files) == len(json_files), f'Number of images {len(img_files)} and JSON files {len(json_files)} must be the same'

            for img_file, json_file in zip(img_files, json_files):
                try:
                    save_mask_img(img_file, maskorimg=args.maskorimg, resize=args.resize,
                                json_file=json_file, dir=args.save_path, label=args.label,
                                pixel_number=args.pixel_number)
                except Exception as e:
                    print(f'Error processing {img_file}: {e}')
        else:
            mask_dir = args.save_path.replace('img', 'gt')
            for img_file in img_files:
                if os.path.basename(img_file).split('.')[0] not in [i.split('.')[0] for i in os.listdir(mask_dir)]:
                    print(f'Image {img_file} does not have corresponding mask, remove {img_file}')
                else:
                    save_mask_img(img_file, maskorimg=args.maskorimg, 
                                resize=args.resize, dir=args.save_path, label=args.label,
                                pixel_number=args.pixel_number)
        print('Done!')

    elif args.command == 'split':
        split_dataset(args.train_dir, args.val_dir, val_number=args.val_number)
        print('Dataset split done!')
