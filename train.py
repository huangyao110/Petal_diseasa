import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from semi_labe.mapper import get_training_augmentation, get_validation_augmentation
from semi_labe.dataloader import Dataset
from semantic_seg.model import PDSEG
import argparse

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Disease Segmentation Training Script')
    
    # 添加命令行参数
    parser.add_argument('--log_dir', type=str, default='logs/', help='日志保存目录')
    parser.add_argument('--log_name', type=str, 
                        default='EX-2025-04-12-step3(disease_seg)_resnet50_unet', help='实验名称')
    parser.add_argument('--train_img_dir', type=str, 
                        default=r"D:\2025\data\obj_for_per_sam\seg_diease\train\img", help='训练图像目录')
    parser.add_argument('--train_mask_dir', type=str, 
                        default=r"D:\2025\data\obj_for_per_sam\seg_diease\train\gt", help='训练掩码目录')
    parser.add_argument('--val_img_dir', type=str, 
                        default=r"D:\2025\data\obj_for_per_sam\seg_diease\val\img", help='验证图像目录')
    parser.add_argument('--val_mask_dir', type=str, 
                        default=r"D:\2025\data\obj_for_per_sam\seg_diease\val\gt", help='验证掩码目录')
    parser.add_argument('--batch_size', type=int, default=14, help='批次大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--encoder_name', type=str, default='timm-regnetx_064', help='编码器名称')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', help='编码器预训练权重')
    parser.add_argument('--pretrained', type=str, 
                        default=r'logs\EX-2025-04-12-step3(disease_seg)_resnet50_unet\version_1\checkpoints\epoch=99-step=26200.ckpt', help='预训练权重路径')
    parser.add_argument('--decoder_name', type=str, default='manet', help='解码器名称')
    parser.add_argument('--freeze_layers', default=['encoder'], help='冻结层的名称')
    parser.add_argument('--device', type=int, default=1, help='GPU设备ID')
    
    args = parser.parse_args()

    # 设置logger
    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.log_name)
    
    # 创建数据集
    train_dataset = Dataset(
        args.train_img_dir,
        args.train_mask_dir,
        augmentation=get_training_augmentation(t_size=(256, 256)),
    )

    valid_dataset = Dataset(
        args.val_img_dir,
        args.val_mask_dir,
        augmentation=get_validation_augmentation(t_size=(256, 256)),
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
        decoder_name=args.decoder_name,
        encoder_name=args.encoder_name,
        in_channels=3,
        encoder_weights=args.encoder_weights,
        out_classes=OUT_CLASSES,
        T_MAX=T_MAX,
        pretrained_model_path=args.pretrained,
        freeze_layers=args.freeze_layers,
    )
    model.to(f'cuda:{args.device}')

    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        log_every_n_steps=1,
        logger=logger,
        devices=1,  # 使用双GPU模式
        accelerator='gpu',
        strategy='auto',  # 使用DataParallel替代DDP，不需要NCCL支持
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
