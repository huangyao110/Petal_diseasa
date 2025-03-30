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
            processed_img = resize_mask_img(mask, target_size)
            output_path = os.path.join(dir, f"{file_name}.png")
        else:
            # Process regular image
            processed_img = resize_mask_img(img, target_size)
            output_path = os.path.join(dir, f"{file_name}.jpg")
            
        # Save processed image
        processed_img.save(output_path)


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser(description='Process images and generate masks')
    parser.add_argument('--img_path', type=str, default='train', required=True,
                        help='Path to the image directory')
    parser.add_argument('--json_path', type=str, default='train',
                        help='Path to the JSON annotations directory')
    parser.add_argument('--save_path', type=str, default='petal_data/Annotations', required=True,
                        help='Path to save the generated masks')
    parser.add_argument('--maskorimg', type=str, default='mask', required=True,
                        help='Whether to generate masks or images')
    parser.add_argument('--resize', type=bool, default=False, required=True,
                        help='Whether to resize masks or images')
    parser.add_argument('--label', type=str, default='petal', required=True,
                        help='label to mask ')
    parser.add_argument('--pixel_number', type=float, default=255, required=True,
                        help='pixel number to mask ')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    json_files = [os.path.join(args.json_path, f) for f in os.listdir(args.json_path) if f.endswith('.json')]
    img_files = [os.path.join(args.json_path, f"{os.path.basename(f).split('.')[0]}.png") for f in json_files]

    if args.maskorimg == 'mask':
        assert len(img_files) == len(json_files), f'Number of images {len(img_files)} and JSON files {len(json_files)} must be the same'

        for img_file, json_file in zip(img_files, json_files):
            save_mask_img(img_file, maskorimg=args.maskorimg, resize=args.resize,
                        json_file=json_file, dir=args.save_path, label=args.label,
                        pixel_number=args.pixel_number)
    else:
        for img_file in img_files:
            save_mask_img(img_file, maskorimg=args.maskorimg, 
                        resize=args.resize, dir=args.save_path, label=args.label,
                        pixel_number=args.pixel_number)
    print('Done!')
