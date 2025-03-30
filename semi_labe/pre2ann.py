## 人工微调代码
### 这部分代码实现对mask-rcnn预测的数据进行处理，以形成相应的ann的josn文件，用于anylabeling工具进行微调
#### Code written by hy, on 2023/10/19, in Suzhou, China

import cv2
import numpy as np
import json
import os
from tqdm import tqdm
import ipdb

def mask2json(mask):
    """
    Convert single mask prediction to JSON-compatible edge points for annotation.
    
    Args:
        img: Input image array (H,W,C) - used only for shape reference
        mask: Binary mask array (H,W) where 1 indicates target region
        
    Returns:
        List of edge point arrays for each mask contour
    """
    # Convert mask to binary (0 or 1)

    binary_mask = (mask == 1).astype(np.uint8)
    
    # Find contours in the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Parameters
    NUM_SAMPLES = 20  # Number of points to sample per contour
    MIN_CONTOUR_POINTS = NUM_SAMPLES  # Minimum points required to process a contour
    
    all_edge_points = []
    
    # Process each contour
    for contour in contours:
        # Skip small contours
        if len(contour) < MIN_CONTOUR_POINTS:
            continue
            
        # Calculate contour length
        contour_length = cv2.arcLength(contour, True)
        
        # Calculate sampling interval
        sample_interval = contour_length / NUM_SAMPLES
        
        # Sample points evenly along contour
        sampled_points = []
        for i in range(NUM_SAMPLES):
            # Calculate target arc length
            target_length = i * sample_interval
            
            # Find closest point
            cumulative_length = 0
            for j in range(1, len(contour)):
                segment_length = np.linalg.norm(contour[j] - contour[j-1])
                cumulative_length += segment_length
                if cumulative_length >= target_length:
                    sampled_points.append(contour[j][0].tolist())
                    break
                    
        all_edge_points.append(sampled_points)
        
    return all_edge_points

def samplepointwrite2json(sample_point, 
                          label,
                          shape_type,  
                          img_path, 
                          save_path):
    """Convert sampled points to JSON annotation format and save to file.
    
    Args:
        sample_point: List of sampled points for each object
        img_path: Path to source image
        save_path: Directory to save JSON annotation
    """
    # Read image and get metadata
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # Convert shape type to LabelMe compatible format
    shape_type = 'rectangle' if shape_type == 'box' else 'polygon'

    img_name = os.path.basename(img_path)

    # Create annotation dictionary
    json_dict = {
        'version': '0.3.3',
        'flags': {},
        'shapes': [],
        'imagePath': img_name,
        'imageData': None,
        'imageHeight': h,
        'imageWidth': w,
        'text': ''
    }

    # Process each object's points
    for obj_points in sample_point:
        if shape_type == 'rectangle':
            # For boxes, convert to [xmin, ymin, xmax, ymax] format
            x_coords = [obj_points[0], obj_points[2]]
            y_coords = [obj_points[1], obj_points[3]]
            points = [[min(x_coords), min(y_coords)], [max(x_coords), max(y_coords)]]
        else:
            # For polygons, keep original points
            points = [point for point in obj_points]
            
        shape_data = {
            'label': label,
            'text': '',
            'points': points,
            'group_id': None,
            'shape_type': shape_type,
            'flags': {}
        }
        json_dict['shapes'].append(shape_data)

    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Save JSON file
    json_filename = os.path.splitext(img_name)[0] + '.json'
    json_path = os.path.join(save_path, json_filename)
    
    with open(json_path, 'w') as f:
        json.dump(json_dict, f, indent=2, separators=(',', ': '))

def main_pred(pred_call, 
              img_dir, 
              shape_type, 
              label, 
              save_json_dir=None):
    """Main prediction function to process images and save annotations.
    
    Args:
        pred_call: Prediction function that takes image path and returns predictions
        img_dir: Directory containing images to process
        shape_type: Type of annotation shape ('mask' or 'box')
        label: Label to assign to annotations
        save_json_dir: Directory to save JSON annotations (defaults to img_dir)
    """
    # Get list of images
    ims = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    no_mask = []
    no_box = []
    
    if shape_type == 'mask':
        # Process each image for mask predictions
        for img_file in tqdm(ims):
            img_path = os.path.join(img_dir, img_file)
            
            # Get predictions from model
            predicted_mask, resized_img = pred_call(img_path, shape_type)
            
            if predicted_mask[predicted_mask == 1].sum() >= 100:
                sample_points = mask2json(predicted_mask)
                save_dir = save_json_dir if save_json_dir else img_dir
                samplepointwrite2json(sample_points, 
                                     label, 
                                     shape_type, 
                                     img_path, 
                                     save_dir)
                
                # Save resized image
                img_filename = os.path.splitext(img_file)[0] + '.png'
                img_save_path = os.path.join(save_dir, img_filename)
                resized_img = resized_img.squeeze().permute(1, 2, 0).cpu().numpy()
                resized_img = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(img_save_path, resized_img)
            else:
                no_mask.append(img_file)
                print(f'No mask detected in image: {img_file}')
                
        print(f'Mask processing complete. {len(no_mask)} images had no predicted masks.')
        
    elif shape_type == 'box':
        # Process each image for box predictions
        for img_file in tqdm(ims):
            img_path = os.path.join(img_dir, img_file)
            

            # Get predicted boxes from model output
            model_output = pred_call(img_path, shape_type)
            predicted_boxes = [[int(coord) for coord in box] for box in model_output['instances'].pred_boxes.tensor.tolist()]

            
            if len(predicted_boxes) > 0:
                save_dir = save_json_dir if save_json_dir else img_dir
                samplepointwrite2json(predicted_boxes,
                                     label,
                                     shape_type,
                                     img_path,
                                     save_dir)
            else:
                no_box.append(img_file)
                print(f'No boxes detected in image: {img_file}')
                
        print(f'Box processing complete. {len(no_box)} images had no predicted boxes.')
    else:
        raise ValueError(f"Unsupported shape_type: {shape_type}. Must be 'mask' or 'box'")
            