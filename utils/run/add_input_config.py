from detectron2.config import CfgNode as CN

def add_aug_config(cfg):
    """
    Add config for augmentation
    """
    cfg.INPUT = CN()

    cfg.INPUT.SIZE_DIVISIBILITY = 16
    cfg.INPUT.FORMAT = 'BGR'
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'choice'
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.MAX_SIZE_TRAIN = 800
    cfg.INPUT.MIN_SIZE_TEST = 1333
    cfg.INPUT.MIN_SIZE_TRAIN = 800
    cfg.INPUT.CROP = CN()
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.CROP.SIZE = [.9, .9]
    cfg.INPUT.CROP.TYPE = 'relative_range'
    cfg.INPUT.COLOR_AUG_SSD =  False
    cfg.INPUT.MASK_FORMAT = 'polygon'