# training set images augmentation
import albumentations as A
def get_training_augmentation(t_size=None):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        
        A.PadIfNeeded(min_height=320, min_width=320, always_apply=True),
        A.RandomCrop(height=320, width=320, always_apply=True),
        
        A.GaussNoise(p=0.2),
        A.Perspective(p=0.5),
        
        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightnessContrast(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.Sharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        A.OneOf(
            [
                A.RandomBrightnessContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Resize(height=t_size[0], width=t_size[1], p=1)
    ]
    return A.Compose(train_transform)

def get_validation_augmentation(t_size=None):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(384, 480),
        A.Resize(height=t_size[0], width=t_size[1], p=1)
    ]
    return A.Compose(test_transform)