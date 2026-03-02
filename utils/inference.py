import torch
from detectron2.engine import DefaultPredictor
from semantic_seg.model import PDSEG
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import os
import numpy as np
import logging
from det import add_tridentnet_config
# from det.trident_backbone import build_trident_resnet_backbone

class InferenceRoseDisease:
    def __init__(self,
                cfg_file,
                petal_encoder_name,
                petal_decoder_name,
                disease_encoder_name,
                disease_decoder_name,
                crop_weight,
                petal_weights_file,
                diease_weights_file,
                device="cuda:0",
                box_threshold=0.5,
                crop_img_size=(512, 512),
                mask_threshold=0.2,
                img_dir=None, 
                img_file=None) -> None:
        if img_dir is not None and img_file is not None:
            raise ValueError("不能同时指定 img_dir 和 img_file")
        if img_dir is not None:
            if not isinstance(img_dir, str):
                raise TypeError("img_dir 必须是字符串")
        elif img_file is not None:
            if not isinstance(img_file, str):
                raise TypeError("img_file 必须是字符串")
        if img_dir:
            assert os.path.isdir(img_dir), f"{img_dir} 不是一个有效的路径"
            self.img_files = [os.path.join(img_dir, file) for file in os.listdir(img_dir) if file.endswith(".jpg") or file.endswith(".png")]
        elif img_file:
            assert os.path.isfile(img_file), f"{img_file} 不是一个有效的路径"
            self.img_files = [img_file]
        else:
            self.img_files = None

        self.device = torch.device(device)
        print(f"device: {self.device}")
        self.crop_img_size = crop_img_size
        self.box_threshold = box_threshold
        self.mask_threshold = mask_threshold
        
        # delopy the object detection model.
        from detectron2.config import get_cfg
        cfg = get_cfg()
        cfg.set_new_allowed(True)
        add_tridentnet_config(cfg)
        cfg.merge_from_file(cfg_file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = crop_weight
        cfg.MODEL.DEVICE = device
        
        self.step1_model = DefaultPredictor(cfg)

        # deploy the petal segmentation model.
        self.step2_models =  PDSEG(
            decoder_name=petal_decoder_name,
            encoder_name=petal_encoder_name,
            pretrained_model_path=petal_weights_file,
            encoder_weights=None,
            freeze_layers=[],
            in_channels=3,
            out_classes=1,
            T_MAX=2,
        )
        self.step2_models.to(self.device)
        self.step2_models.eval()
        
        # delopy the diease segmentation model.
        self.step3_models =  PDSEG(
            decoder_name=disease_decoder_name,
            encoder_name=disease_encoder_name,
            pretrained_model_path=diease_weights_file,
            freeze_layers=[],
            encoder_weights=None,
            in_channels=3,
            out_classes=1,
            T_MAX=2,
        )

        self.step3_models.to(self.device)
        self.step3_models.eval()
        
        
    def _preprocess_image(self, img_path):
        """预处理图像的通用函数"""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _create_tensor(self, img):
        """将图像转换为张量的通用函数"""
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        return img_tensor

    def step1_crop_img(self, img_file, save_dir=None):
        img_tensor = cv2.imread(img_file)
        
        crop_imgs = []
        res = self.step1_model(img_tensor)
        if len(res["instances"]) == 0:
            return None
            
        bbox = res["instances"].pred_boxes.tensor.tolist()
        score = res["instances"].scores.tolist()
        print(len(score))
        keep_boxes = [
            box for box, score in zip(bbox, score) if score > self.box_threshold
        ]
        if len(keep_boxes) == 0:
            return None
        print(len(keep_boxes))
            
        for box in keep_boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cropped = img_tensor[y1:y2, x1:x2]
            crop_imgs.append(cropped)
            
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            for idx, crop_img in enumerate(crop_imgs):
                resized = cv2.resize(crop_img, self.crop_img_size, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(os.path.join(save_dir, f"{os.path.basename(img_file).split('.')[0]}_{idx}.jpg"), resized)
                # 保存box坐标
                # 将box坐标和图像信息一起保存
                # 将所有图片的box信息保存到同一个文件中
        np.save(os.path.join(save_dir, f"{os.path.basename(img_file).split('.')[0]}_boxes_info.npy"), np.array(keep_boxes))
        

    def _segment_and_save_mask(self, img_file, save_dir, model, mask_type):
        """通用的分割和保存掩码函数"""
        img = self._preprocess_image(img_file)
        
        if img.shape[:2] != self.crop_img_size:
            img = cv2.resize(img, self.crop_img_size, interpolation=cv2.INTER_LINEAR)
            logging.info(f"{img_file} is resized to {self.crop_img_size}")
            
        img_tensor = self._create_tensor(img)
        logits = model(img_tensor)
        pr_mask = torch.sigmoid(logits)
        pr_mask_np = pr_mask.cpu().detach().numpy().squeeze()
        
        # Apply threshold with hysteresis for better edge preservation
        high_thresh = self.mask_threshold
        strong_mask = pr_mask_np >= high_thresh
        
        # Final mask
        final_mask = np.zeros_like(pr_mask_np, dtype=np.uint8) # Ensure mask is uint8
        final_mask[strong_mask] = 1

        # Smooth the mask edges
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smoothed_mask = np.zeros_like(final_mask)
        if contours:
            # Optional: Filter small contours if needed
            # contours = [cnt for cnt in contours if cv2.contourArea(cnt) > some_threshold]

            # Approximate contours to smooth them
            smoothed_contours = []
            for cnt in contours:
                epsilon = 0.005 * cv2.arcLength(cnt, True) # Adjust epsilon for desired smoothness
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                smoothed_contours.append(approx)

            # Draw smoothed contours to create the final smoothed mask
            cv2.drawContours(smoothed_mask, smoothed_contours, -1, (1), thickness=cv2.FILLED)
        else:
            # If no contours found, use the original final_mask
            smoothed_mask = final_mask
        
        # 后处理掩码
        smoothed_mask = self.postprocess_mask(smoothed_mask)

        if save_dir is None:
            save_dir = os.path.dirname(img_file)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        else:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
        # save the smoothed mask
        cv2.imwrite(os.path.join(save_dir, f"{os.path.splitext(os.path.basename(img_file))[0]}_{mask_type}_mask.png"), smoothed_mask * 255)
        logging.info(f"{img_file} is saved in {save_dir}")


    def postprocess_mask(self, mask):
        """后处理掩码,删除小面积的掩码区域"""
        # 确保掩码是二值图像
        # if not isinstance(mask, np.ndarray):
        #     mask = np.array(mask)
        # binary_mask = (mask > 0).astype(np.uint8)
        
        # # 寻找所有连通区域
        # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        # if num_labels <= 2:  # 包含背景,所以实际掩码数量要减1
        #     return binary_mask
            
        # # 获取各个连通区域的面积(像素数),不包括背景(index 0)
        # areas = stats[1:, cv2.CC_STAT_AREA]
        # # 创建新掩码,只保留符合条件的区域
        # processed_mask = np.zeros_like(binary_mask)
        # h, w = binary_mask.shape
        
        # # 计算图像中心区域的范围(中心1/4区域)
        # center_x_min = w * 0.375 * 2 # 图像宽度的3/8
        # center_x_max = w * 0.625 * 2 # 图像宽度的5/8
        # center_y_min = h * 0.375 * 2 # 图像高度的3/8
        # center_y_max = h * 0.625 * 2 # 图像高度的5/8
        
        # for i in range(1, num_labels):  # 从1开始以跳过背景
        #     area = stats[i, cv2.CC_STAT_AREA]
        #     center_x, center_y = centroids[i]
            
        #     # 判断连通区域的中心点是否在图像中心区域内
        #     is_center = (center_x_min <= center_x <= center_x_max and 
        #                 center_y_min <= center_y <= center_y_max)
            
        #     # 只保留面积大于最大面积一半且中心点在图像中心区域的连通区域
        #     if is_center:
        #         processed_mask[labels == i] = 1
                
        return mask
        

    def step2_seg_petal(self, img_file, save_dir=None):
        """花瓣分割函数"""
        self._segment_and_save_mask(img_file, save_dir, self.step2_models, "petal")

    def step3_seg_disease(self, img_file, save_dir=None):
        """病害分割函数"""
        self._segment_and_save_mask(img_file, save_dir, self.step3_models, "disease")

class ImageVisualizer:
    def __init__(self):
        pass
        
    def show_img(self, img, show_type='box', box=None, mask=None):
        """显示图像的主入口函数"""
        if show_type not in ['box', 'mask']:
            raise ValueError("show_type 必须是 'box' 或者'mask'")
        if show_type == 'box':
            if box is None:
                raise ValueError("显示box时必须提供box参数")
            self.show_box(img, box)
        elif show_type == 'mask':
            if mask is None:
                raise ValueError("显示mask时必须提供mask参数")
            self.show_mask(img, mask)

    def _check_image_input(self, img):
        """处理不同格式的输入图像"""
        if isinstance(img, str):
            return self._preprocess_image(img)
        elif isinstance(img, np.ndarray):
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img, Image.Image):
            img_array = np.array(img)
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError("img 必须是字符串、numpy.ndarray 或 PIL.Image")
            
    def _preprocess_image(self, img_path):
        """预处理图像的通用函数"""
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def show_box(self, img, box):
        """显示带有边界框的图像"""
        img = self._check_image_input(img)
    def show_box(self, img, boxes):
        """显示带有边界框的图像"""
        img = self._check_image_input(img)
        
        # 处理单个box和多个box的情况
        if not isinstance(boxes[0], (list, tuple)):
            boxes = [boxes]  # 将单个box转换为列表
            
        # 绘制所有边界框
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # 在边界框上方添加编号
            label = str(idx + 1)  # 从1开始编号
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            thickness = 2
            # 获取文本大小以调整位置
            (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
            # 确保文本位置在图像范围内
            text_x = max(x1, 0)
            text_y = max(y1 - 10, text_height)  # 在框上方10个像素
            # 绘制文本
            cv2.putText(img, label, (text_x, text_y), font, font_scale, (255, 0, 0), thickness)
            
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.title("检测到的目标")
        plt.axis('off')
        plt.show()

    def show_mask(self, img, mask):
        """显示带有遮罩的图像"""
        # 处理输入图像
        img = self._check_image_input(img)

        # 处理遮罩
        if isinstance(mask, str):
            mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        elif isinstance(mask, np.ndarray):
            mask = mask.astype(np.uint8)
        else:
            raise ValueError("mask 必须是字符串路径或 numpy.ndarray")

        # 调整遮罩大小以匹配图像
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

        # 创建叠加效果
        overlay = img.copy()
        alpha = 0.4  # 透明度
        
        # 创建彩色遮罩
        colored_mask = np.zeros_like(img)
        colored_mask[mask > 0] = [193, 4, 45]  # BGR 格式的红色
        
        # 合并图像和遮罩
        result = cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0)
        
        # 显示结果
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("原始图像")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title("遮罩结果")
        plt.axis('off')
        
        plt.show()