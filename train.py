import torch
import pytorch_lightning as pl
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl
from semi_labe.mapper import get_training_augmentation, get_validation_augmentation
from semi_labe.dataloader import Dataset
from semantic_seg.fpn import FPN
from semantic_seg.model import PDSEG
import argparse

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Disease Segmentation Training Script')
    
    # 添加命令行参数
    parser.add_argument('--log_dir', type=str, default='logs/', help='日志保存目录')
    parser.add_argument('--log_name', type=str, default='EX-2025-03-16-step3(disease seg)_mobilenetv2_uvnet++', help='实验名称')
    parser.add_argument('--train_img_dir', type=str, default=r"D:\2025\data\obj_for_per_sam\seg_diease\train\img", help='训练图像目录')
    parser.add_argument('--train_mask_dir', type=str, default=r"D:\2025\data\obj_for_per_sam\seg_diease\train\gt", help='训练掩码目录')
    parser.add_argument('--val_img_dir', type=str, default=r"D:\2025\data\obj_for_per_sam\seg_diease\val\img", help='验证图像目录')
    parser.add_argument('--val_mask_dir', type=str, default=r"D:\2025\data\obj_for_per_sam\seg_diease\val\gt", help='验证掩码目录')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--epochs', type=int, default=60, help='训练轮数')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--encoder_name', type=str, default='mobilenet_v2', help='编码器名称')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', help='编码器预训练权重')
    parser.add_argument('--device', type=int, default=1, help='GPU设备ID')
    
    args = parser.parse_args()

    # 设置logger
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.log_name)
    
    # 创建数据集
    train_dataset = Dataset(
        args.train_img_dir,
        args.train_mask_dir,
        augmentation=get_training_augmentation(t_size=(512, 512)),
    )

    valid_dataset = Dataset(
        args.val_img_dir,
        args.val_mask_dir,
        augmentation=get_validation_augmentation(t_size=(512, 512)),
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                shuffle=True, num_workers=args.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)

    # 计算T_MAX
    T_MAX = args.epochs * len(train_loader)
    OUT_CLASSES = 1

    # 创建模型
    model = PDSEG(
        encoder_name=args.encoder_name,
        in_channels=3,
        encoder_weights=args.encoder_weights,
        out_classes=OUT_CLASSES,
        T_MAX=T_MAX
    )

    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        log_every_n_steps=1,
        logger=logger,
        devices=[args.device],
        num_nodes=1,
    )

    # 开始训练
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=valid_loader,
    )

if __name__ == '__main__':
    main()
