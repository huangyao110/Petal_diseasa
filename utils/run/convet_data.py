# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode
import os
import json
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
import shutil
import logging

__all__ = ["register_qk_dataset"]

# 初始化日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def split_qk_jk(img_dir: str, out_dir: str): 
    ratio = 0.8
    files = os.listdir(img_dir)
    files_img = [i for i in files if i.endswith(".jpg")]
    files_json = [i for i in files if i.endswith(".json")]
    if len(files_json) != len(files_img):
        logger.info("The number of images and json files are not equal.")
        i = [o.split('.')[0] for o in files_img]
        j = [n.split('.')[0] for n in files_json]
        diff_files =  list(set(i) - set(j))
        logger.info(f"Files not in json: {diff_files}")
    if diff_files is not None:
        for i in diff_files:
            files_img.remove(f'{i}.jpg')
    files_img, files_json = sorted(files_img), sorted(files_json)
    assert len(files_img) == len(files_json), "The number of images and json files are not equal."
    train_files_img, val_files_img = files_img[:int(ratio*len(files_img))], files_img[int(ratio*len(files_img)):]
    train_files_json, val_files_json = files_json[:int(ratio*len(files_json))], files_json[int(ratio*len(files_json)):]
    dst_train = out_dir + "train"
    dst_val = out_dir + "val"
    if not os.path.exists(dst_train):
        os.makedirs(dst_train, exist_ok=True) 
    if not os.path.exists(dst_val):
        os.makedirs(dst_val, exist_ok=True)
    for i,j in zip(train_files_img, train_files_json):
        try:  
            img = os.path.join(img_dir, i)
            json = os.path.join(img_dir, j)
            shutil.move(img, dst_train)
            shutil.move(json, dst_train)
        except Exception as e:
            logger.error(f"An error occurred when moving the {img} to {dst_train}: {e}")
            continue
    for i,j in zip(val_files_img, val_files_json):
        try:  
            img = os.path.join(img_dir, i)
            json = os.path.join(img_dir, j)
            shutil.move(img, dst_val)
            shutil.move(json, dst_val)
        except Exception as e:
            logger.error(f"An error occurred when moving the {img} to {dst_train}: {e}")
            continue

def get_qk_dicts(img_dir):
    dataset_dicts = []
    try:
        for idx, v in enumerate(os.listdir(img_dir)):
            record = {}
            if v.endswith(".json"):
                logger.info(f"Processing file: {v}")
                json_file = os.path.join(img_dir, v)
                try:
                    with open(json_file, 'r') as f:
                        imgs_anns = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to read {json_file}: {e}")
                    continue

                filename = os.path.join(img_dir, imgs_anns["imagePath"])
                record["file_name"] = filename
                record["image_id"] = idx
                record["height"] = imgs_anns['imageHeight']
                record["width"] = imgs_anns['imageWidth']

                try:
                    annos = imgs_anns["shapes"]
                    objs = []
                    for _, anno in enumerate(annos):
                        pxy = anno["points"]
                        px, py = [], []
                        for x, y in pxy:
                            px.append(x)
                            py.append(y)
                        obj = {
                            "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": 0,
                        }
                        objs.append(obj)
                    record["annotations"] = objs
                except KeyError as e:
                    logger.error(f"Missing key in annotation data: {e}")
                    continue
                dataset_dicts.append(record)
                logger.debug(f"Added record for image {idx}: {record}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

    return dataset_dicts

# 确保在调用函数之前设置好日志级别和格式
logging.basicConfig(level=logging.INFO)

def register_qk_dataset(img_dir):
    for d in ["train", "val"]:
        DatasetCatalog.register("qk_dataset_" + d, lambda d=d: get_qk_dicts(os.path.join(img_dir, d)))
        MetadataCatalog.get("qk_dataset_" + d).set(thing_classes=["petal"])
        MetadataCatalog.get("qk_dataset_" + d).set(evaluator_type="coco")

img_dir = r"D:\2025\data\obj_dectetion\data_path_for_obj_dect"
register_qk_dataset(img_dir)
# qk_data_metadata = MetadataCatalog.get('qk_dataset_train')