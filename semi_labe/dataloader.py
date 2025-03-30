import os
import cv2
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.optim import lr_scheduler
import segmentation_models_pytorch as smp
import pytorch_lightning as pl


class Dataset(BaseDataset):
    """CamVid Dataset. Read images, apply augmentation transformations.
    
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
            
    """
    
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
    ):
        self.ids = os.listdir(images_dir)
        self.gt_ids = [i.replace('jpg', 'png') for i in self.ids]
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.gt_ids]
        self._check_data(self.images_fps, self.masks_fps)
        
        # convert str names to class values on masks
        self.class_values = [1]
        
        self.augmentation = augmentation

    def _check_data(self, img_list, label_list):
        img_len = len(img_list)
        label_len = len(label_list)
        assert img_len == label_len, "the number of labels must be equal to the number of images."
        for i, j in zip(img_list, label_list):
            i_name = os.path.basename(i).split('.')[0]
            j_name = os.path.basename(j).split('.')[0]
            assert i_name == j_name, 'image and label are not the same.'

    
    def __getitem__(self, i):
        image = cv2.imread(self.images_fps[i])
        # BGR-->RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_fps[i], 0)
        
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        return image.transpose(2, 0, 1), mask.transpose(2, 0, 1)
        
    def __len__(self):
        return len(self.ids)
    
# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        if name == 'image':
            plt.imshow(image.transpose(1, 2, 0))
        else:
            plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    from mapper import get_training_augmentation
    img_dir = r"D:\rose-flower\data\seg\val\images"
    mask_dir = r"D:\rose-flower\data\seg\val\labels"
    dataset = Dataset(img_dir, mask_dir, classes=['disease'], augmentation=get_training_augmentation())
    
    for i in range(3):
        img, labels = dataset[10]
        visualize(image=img, disease_mask = labels.squeeze())
