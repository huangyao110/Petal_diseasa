import torch
import pytorch_lightning as pl
import os
import torch
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl

from segmentation_models_pytorch.decoders.unet import Unet 
from segmentation_models_pytorch.decoders.unetplusplus import UnetPlusPlus

__all__ = ['PDSEG']

class PDSEG(pl.LightningModule):

    def __init__(self, 
                encoder_name, 
                decoder_name,
                in_channels, 
                out_classes, 
                T_MAX, 
                encoder_weights, 
                pretrained_model_path=None,
                freeze_layers = None,
                **kwargs):
        super().__init__()
        # self.model = FPN.FPN(
        #     encoder_name = encoder_name,
        #     encoder_depth = 5,
        #     encoder_weights = encoder_weights,
        #     decoder_pyramid_channels = 256,
        #     decoder_segmentation_channels= 128,
        #     decoder_merge_policy = "add",
        #     decoder_dropout = 0.1,
        #     in_channels = in_channels,
        #     classes = out_classes,
        #     activation = None,
        #     upsampling = 4,
        #     aux_params = None,
        # )
        if decoder_name == 'unet':
            self.model = Unet(
                        encoder_name = encoder_name, 
                        encoder_depth = 5,
                        encoder_weights =encoder_weights,
                        decoder_use_batchnorm = True,
                        decoder_channels = (256, 128, 64, 32, 16),
                        decoder_attention_type= 'scse',
                        in_channels = in_channels,
                        classes = out_classes,
            )
        elif decoder_name == 'unet++':
            self.model = UnetPlusPlus(
                encoder_name = encoder_name,
                encoder_depth = 5,
                encoder_weights = encoder_weights,
                decoder_use_batchnorm = True,
                decoder_channels = (256, 128, 64, 32, 16),
                decoder_attention_type = 'scse',
                in_channels = in_channels,
                classes = out_classes,
            )
        elif decoder_name == 'deeplabv3':
            self.model = smp.DeepLabV3(
                encoder_name = encoder_name,
                encoder_depth = 5,
                encoder_weights = encoder_weights,  
                in_channels = in_channels,
                classes = out_classes,
            )
        elif decoder_name == 'manet':
            self.model = smp.MAnet(
                encoder_name = encoder_name,
                encoder_depth = 5,
                encoder_weights = encoder_weights,
                in_channels = in_channels,
                classes = out_classes,
            )
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name=encoder_name)
        self.T_MAX = T_MAX
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        
        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        
        # initialize step metics
        if pretrained_model_path is not None:
            self._load_pretrained_model(pretrained_model_path)
        if freeze_layers is not None:
            assert type(freeze_layers) in [str, list], "freeze_layers must be a list or str."
            self.freeze(freeze_layers)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def _load_pretrained_model(self, pretrained_model_path):
        """加载预训练模型"""
        if os.path.exists(pretrained_model_path):
            checkpoint = torch.load(pretrained_model_path)
            # 使用strict=False允许加载部分权重
            self.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            raise FileNotFoundError(f"预训练模型文件 {pretrained_model_path} 不存在")

    
    def freeze(self, which_freeze: str) -> None:
        """
        冻结模型的某些层
        :param which_freeze: 要冻结的层的名称或索引
        """
        for i in which_freeze:
            assert hasattr(self.model, i), f"模型中没有名为 {i} 的层"
            assert i in ['encoder', 'decoder', 'segmentation_head'], f"模型中没有名为 {i} 的层"
            layer = getattr(self.model, i)
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):

        image, mask = batch
        
        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4
        
        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0
        
        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        # 添加调试断点
    
        loss = self.loss_fn(logits_mask, mask)
        
        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        
        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        
        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return 

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return 

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return 

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5, 
                                     weight_decay=1e-5)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_MAX, eta_min=1e-5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 2
            }
        }
        return 