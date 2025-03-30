
from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN
from detectron2.modeling.backbone import fpn
from .scafm2backbone import backbone_with_scafm
from .lvt import lvt2mutilout


# @ BACKBONE_REGISTRY.register()
def bulid_backbone_scafm_fpn(cfg, backbone):
    backbone = backbone_with_scafm(cfg, backbone)
    model = FPN(
        bottom_up=backbone,
        in_features=['s_1','s_2','s_3','s_4'],
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm="",
        top_block=fpn.LastLevelMaxPool(), 
        fuse_type="sum",
        square_pad=0,
    )
    return model

@ BACKBONE_REGISTRY.register()
def bulid_lvt_scafm_fpn(cfg,  input_shape=None):
    backbone = backbone_with_scafm(cfg, lvt2mutilout)
    model = FPN(
        bottom_up=backbone,
        in_features=['s_1','s_2','s_3','s_4'],
        out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
        norm="",
        top_block=fpn.LastLevelMaxPool(), 
        fuse_type="sum",
        square_pad=0,
    )
    return model

# @ BACKBONE_REGISTRY.register()



if __name__ == '__main__':
    # from thop import profile
    # from thop import clever_format
    # from detectron2.config import get_cfg
    # config_file = r'work\config\lvt.yaml'
    # cfg = get_cfg()
    # cfg.set_new_allowed(True)
    # cfg.merge_from_file(config_file)
    # input = torch.randn(1, 3, 224, 224)
    # model = bulid_lvt_scafm_fpn(cfg)
    # x = torch.randn(1, 3, 224, 224)
    # out = model(x)
    # for i in out:
    #     print(out[i].size())  # 打印每一阶段的输出����
    # macs, params = profile(model, inputs=(input, ))
    # flops = macs * 2
    # b, params = clever_format([flops, params], "%.3f") 
    # print(b, params)
    pass