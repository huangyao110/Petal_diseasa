import torch.nn as nn
from torch.nn import functional as F
from detectron2.modeling import Backbone
from ..semantic_seg.fpn.scafm import SCAFM

__all__ = ['backbone_with_scafm']

class backbone_with_scafm(Backbone):
    def __init__(self, cfg, backbone):
        super(backbone_with_scafm, self).__init__()
        self.backbone = backbone(cfg)
        scafm_list = []
        layers = cfg.MODEL.BACKBONE.embed_dims # [64, 64, 128, 256]
        for s_i in range(len(layers)):
            if s_i == 3:
                scafm_list.append(nn.Identity())
            else:
                scafm_list.append(
                SCAFM(layers[s_i+1],layers[s_i],mode='dl',out_c=layers[s_i]) 
                )
        self.scafm_list = nn.ModuleList(scafm_list)

    def forward(self, x):
        out = self.backbone(x)
        outputs = {}
        length = len(out)
        for idx, ((k, v), layer) in enumerate(zip(out.items(), self.scafm_list)):
            if idx == int(length-1):
                outputs[k] = layer(v)
            else:
                na = f's_{idx+2}'
                next_v = out[na]
                outputs[k] = layer(next_v,v)
        return outputs

    def output_shape(self):
        return self.backbone.output_shape()
    

