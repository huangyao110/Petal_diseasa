import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .file import get_per_img_path
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_labelme_annotation(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_mask_from_polygon(shapes, img_height, img_width, class_name=None):
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    class_list = []
    
    for shape in shapes:
        if class_name is not None and shape['label'] != class_name:
            continue
        if shape['label'] not in class_list:
            class_list.append(shape['label'])
        if shape['shape_type'] == 'polygon':
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
    return mask, class_list

def calculate_area(mask):
    return np.count_nonzero(mask)

def calculate_class_area(json_file, class_name=None):
    data = read_labelme_annotation(json_file)
    img_height = data['imageHeight']
    img_width = data['imageWidth']
    mask, class_list = get_mask_from_polygon(data['shapes'], img_height, img_width, class_name)
    area = calculate_area(mask)
    if class_name is not None:
        return {class_name: area}, area
    class_areas = {}
    for cls in class_list:
        cls_mask, _ = get_mask_from_polygon(data['shapes'], img_height, img_width, cls)
        cls_area = calculate_area(cls_mask)
        class_areas[cls] = cls_area
    
    return class_areas, area

def visualize_annotation(json_file, class_name=None, save_path=None):
    # 这个是可视化labelme标注的
    data = read_labelme_annotation(json_file)
    img_path = os.path.join(os.path.dirname(json_file), data['imagePath'])
    if not os.path.exists(img_path):
        img_path = os.path.join(os.path.dirname(json_file), os.path.basename(data['imagePath']))
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.zeros((data['imageHeight'], data['imageWidth'], 3), dtype=np.uint8)
    mask, _ = get_mask_from_polygon(data['shapes'], data['imageHeight'], data['imageWidth'], class_name)
    colored_mask = np.zeros_like(img)
    colored_mask[mask > 0] = [255, 0, 0]  # 红色掩码
    alpha = 0.5
    result = cv2.addWeighted(colored_mask, alpha, img, 1 - alpha, 0)
    class_areas, total_area = calculate_class_area(json_file, class_name)
    
    # 显示图像和面积信息
    plt.figure(figsize=(12, 8))
    plt.imshow(result)
    
    # 添加面积信息
    area_text = f"Total: {total_area} pixel\n"
    for cls, area in class_areas.items():
        area_text += f"{cls}: {area} pixel\n"
    
    plt.title(area_text)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

def batch_calculate_areas(json_dir, class_name=None, output_file=None):
    # 这个是批量计算labelme标注的相对面积
    json_files = get_per_img_path(json_dir, 'json')
    results = {}
    for json_file in json_files:
        try:
            class_areas, total_area = calculate_class_area(json_file, class_name)
            file_name = os.path.basename(json_file)
            results[file_name] = {
                "class_areas": class_areas,
                "total_area": total_area
            }
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    else:
        for file_name, result in results.items():
            print(f"文件: {file_name}")
            print(f"总面积: {result['total_area']} 像素")
            for cls, area in result['class_areas'].items():
                print(f"  {cls}: {area} 像素")
            print()
    
    return results

def calculate_area_percentage(json_file, class_name):
    data = read_labelme_annotation(json_file)
    img_height = data['imageHeight']
    img_width = data['imageWidth']
    total_img_area = img_height * img_width
    mask, _ = get_mask_from_polygon(data['shapes'], img_height, img_width, class_name)
    class_area = calculate_area(mask)
    percentage = (class_area / total_img_area) * 100
    return percentage


def calculate_relative_index(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = []
    for file_name, areas in data.items():
        disease_area = areas['class_areas'].get('disease', 0)
        petal_area = areas['class_areas'].get('petal', 0)
        total_area = areas['total_area']
        if petal_area > 0:
            relative_index = disease_area / petal_area
            disease_percentage = disease_area / total_area * 100 if total_area > 0 else 0
            petal_percentage = petal_area / total_area * 100 if total_area > 0 else 0
            
            results.append({
                'file_name': file_name.split('.')[0],
                'disease_area': disease_area,
                'petal_area': petal_area,
                'total_area': total_area,
                'relative_index': relative_index,
                'disease_percentage': disease_percentage,
                'petal_percentage': petal_percentage
            })
    return pd.DataFrame(results).set_index('file_name')

def calculate_area_percentage_from_mask(disease_mask_path, petal_mask_path):
    petal_mask = plt.imread(petal_mask_path)
    disease_mask = plt.imread(disease_mask_path)
    petal_mask = petal_mask // 255 if np.max(petal_mask) > 1 else petal_mask
    disease_mask = disease_mask // 255 if np.max(disease_mask) > 1 else disease_mask
    petal_area = np.sum(petal_mask == 1)
    disease_area = np.sum(disease_mask == 1)
    return disease_area / petal_area

def batch_calculate_relative_index_from_mask(disease_mask_dir, petal_mask_dir):
    disease_paths = get_per_img_path(disease_mask_dir, 'png')
    petal_paths = get_per_img_path(petal_mask_dir, 'png')
    results = {os.path.basename(p).split('_petal')[0]: calculate_area_percentage_from_mask(d, p) 
              for d, p in zip(disease_paths, petal_paths)}             
    return pd.DataFrame(results, index=['pred_mask']).T

def bacth_calculate_relative_index_from_fit_ellipse(disease_mask_dir, petal_mask_dir, fun_call):
    disease_paths = get_per_img_path(disease_mask_dir, 'png')
    petal_paths = get_per_img_path(petal_mask_dir, 'png')
    res = {}
    for i, j in zip(disease_paths, petal_paths):
        _, m_area = fun_call(i, None)
        _, petal_area = fun_call(j, None)
        if m_area is None:
            s = 0
        else:
            s = m_area/petal_area
        res[os.path.basename(j).split('_petal')[0]] = s
    return pd.DataFrame(res, index=['fit_ellipse']).T
    

