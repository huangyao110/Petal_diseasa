from semantic_seg.model import PDSEG

import torch
import os
import cv2
import numpy as np

from semi_labe.pre2ann import main_pred 
from detectron2.engine import DefaultPredictor

from backbone import add_tridentnet_config

class semi(object):
    def __init__(self, 
                 img_dir,
                 shape_type,
                 label,
                 cfg=None,
                 model=None,
                 encoder_name=None,
                 weights_file=None):
        self.shape_type = shape_type
        assert shape_type in ['box', 'mask'], '...'
        self.label = label
        if shape_type == 'box':
            if cfg is None:
                raise ValueError("cfg must be provided when shape_type is 'box'")
            self.model = DefaultPredictor(cfg)
        elif shape_type == 'mask':
            if model is None or encoder_name is None or weights_file is None:
                raise ValueError("model, encoder_name and weights_file must be provided when shape_type is 'mask'")
            weights = torch.load(weights_file)
            model = model(encoder_name=encoder_name, in_channels=3, out_classes=1)
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.load_state_dict(weights['state_dict'])
            self.model = model.to(self.device)
            self.model.eval()
        else:
            raise ValueError(f"Invalid shape_type: {shape_type}. Must be either 'box' or 'mask'")
        self.img_dir = img_dir


    def predict(self, img_file, shape_type):
        # Read and preprocess image
        img = cv2.imread(img_file)
        if shape_type == 'mask':
            if img is None:
                raise ValueError(f"Unable to read image file: {img_file}")
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            img_tensor = torch.nn.functional.interpolate(img_tensor, 
                                                        size=(640, 640), 
                                                        mode='bilinear', 
                                                        align_corners=False)
            img_copy = img_tensor.clone()
            
            # Model inference
            with torch.no_grad():
                logits = self.model(img_tensor)
                
            # Process output
            pr_mask = torch.sigmoid(logits)
            pr_mask_np = pr_mask.cpu().numpy().squeeze()
            
            # Apply threshold with hysteresis for better edge preservation
            high_thresh = 0.8
            strong_mask = pr_mask_np >= high_thresh
            
            # Final mask
            final_mask = np.zeros_like(pr_mask_np)
            final_mask[strong_mask] = 1
            
            return final_mask, img_copy
        elif shape_type == 'box':
            res = self.model(img)
            return res

    
    def mask2json(self, save_json):
        main_pred(self.predict, 
                  self.img_dir, 
                  shape_type=self.shape_type,
                  label=self.label,
                  save_json_dir=save_json)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run semi-supervised labeling')
    parser.add_argument('--encoder', type=str, default=None, required=False,
                        help='Encoder name for the model')
    parser.add_argument('--weights', type=str, required=False,
                        default=r'logs\step_obj_dect\model_final.pth',
                        help='Path to model weights file')
    parser.add_argument('--img_dir', type=str, required=False,
                        default= r'D:\2025\data\petal_form_jrh\5_9',
                        help='Directory containing images to process')
    parser.add_argument('--save_dir', type=str, required=False,
                        default=r'D:\2025\data\petal_form_jrh\5_9',
                        help='Directory to save JSON annotations')
    parser.add_argument('--cfg', type=str, required=False,
                        default=r'configs\Base-TridentNet-Fast-C4.yaml',
                        help='cfg file for box generator')
    parser.add_argument('--shape_type', type=str, required=False,
                        default='box',
                        help='Annotations type')
    parser.add_argument('--label', type=str, required=False,
                        default='petal',
                        help='Annotatuons label')
    args = parser.parse_args()
    
    
    if args.shape_type =='box':
        from detectron2.config import get_cfg
        cfg = get_cfg()
        cfg.set_new_allowed(True)
        add_tridentnet_config(cfg)
        cfg.merge_from_file(args.cfg)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = args.weights
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        semi_model = semi(args.img_dir, args.shape_type, args.label, cfg)
        
        
    elif args.shape_type =='mask':
        model = PDSEG
        semi_model = semi(args.img_dir, args.shape_type, args.label, model, args.encoder, args.weights)
        
    semi_model.mask2json(args.save_dir) 