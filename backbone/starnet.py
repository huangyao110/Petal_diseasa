# The StarNet implementation in PyTorch, 
# and reference: https://github.com/ma-xu/Rewrite-the-Stars/blob/main/2D_visual/demonet_2d.py
# @author: HuangYao, email: 1647721078@qq.com
# @date: 2024, july 12.

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from detectron2.modeling import Backbone, BACKBONE_REGISTRY
from detectron2.layers import CNNBlockBase, ShapeSpec, get_norm
from detectron2.modeling import FPN
from detectron2.modeling.backbone.fpn import LastLevelMaxPool, LastLevelP6P7


class Block(nn.Module):
    def __init__(self, dim, mode="mul", act=nn.ReLU):

        super().__init__()
        self.mode=mode
        self.norm = nn.LayerNorm(dim)
        self.f = nn.Conv1d(in_channels=dim, out_channels=int(6 * dim), kernel_size=1, stride=1, padding=0) 
        self.act = act()
        self.g = nn.Linear(3*dim, dim)

    def forward(self, x):
        input = x
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C).contiguous()
        x = self.norm(x).transpose(2,1)
        x = self.f(x).transpose(2,1)
        dim = x.shape[-1]
        x1, x2 = torch.chunk(x.reshape(B, 2*H*W, int(dim//2)), chunks=2, dim=1)
        x = self.act(x1)+x2 if self.mode == "sum" else self.act(x1)*x2
        x = self.g(x)
        x = x.permute(0, 2, 1).contiguous().view_as(input)
        x = input + x
        return x
    
class Stem(nn.Module):
    def __init__(self, in_chans=3, out_chans=100):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        self.norm = nn.BatchNorm2d(out_chans)
        self.act  = nn.ReLU6()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


@ BACKBONE_REGISTRY.register()
class StarNet(Backbone):
    def __init__(self, in_chans=3, depth=4, dim=100, layer_num = [2,2,8,3], act=nn.ReLU,
                 mode="sum", num_classes=None, **kwargs):
        super().__init__()
        assert mode in ["sum", "mul"]
        self.stem = Stem(in_chans, dim)
        self.depth = depth
        out_c = dim
        blocks = nn.ModuleList()

        self._out_features         = []
        self._out_feature_channels = {}
        self._out_feature_strides  = {}

        for i in range(depth):
            in_c  = out_c
            out_c = 2 * in_c
            conv1 = nn.Conv2d(in_c, out_c, stride=2, kernel_size=3, padding=1) # downsampling ratio = 2
            dw1   = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, groups=out_c)
            blocks = [Block(dim=out_c, act=act, mode=mode) for i in range(layer_num[i])]
            dw2   = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, groups=out_c)
            blocks.append(nn.Sequential(conv1, dw1, *blocks, dw2))
            self.add_module(f'stage_{i+1}', blocks[-1])
            self._out_features.append(f'stage_{i+1}')
            self._out_feature_channels[f'stage_{i+1}'] = out_c
            self._out_feature_strides[f'stage_{i+1}']  = int(2**(i+2))
        self.norm = nn.LayerNorm(out_c) # final norm layer
        if num_classes is not None:
            self.head = nn.Linear(out_c, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)


    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    def forward(self, x):
        output = {}
        x = self.stem(x)
        stages = range(1, self.depth + 1)  # 创建一个范围，表示所有阶段的编号
        for stage_num in stages:
            stage_name = f'stage_{stage_num}'
            # 使用getattr获取模块，然后调用它
            x = getattr(self, stage_name)(x)
            output[stage_name] = x
        if hasattr(self, 'head'):
            b, s, h, w = x.size()
            x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, x.size(1)).contiguous()
            x = self.norm(x)
            x = self.head(x)
            return x
        return output
    

@BACKBONE_REGISTRY.register()
def bulid_StarNet_with_fpn(cfg, input_shape=None):
    depth = cfg.MODEL.BACKBONE.DEPTH
    dim   = cfg.MODEL.BACKBONE.DIM
    mod = cfg.MODEL.BACKBONE.MODEL
    out = StarNet(depth=depth, dim=dim, act=nn.ReLU, mode=mod, num_classes=None)
    model = FPN(bottom_up=out,
                in_features=cfg.MODEL.FPN.IN_FEATURES,
                out_channels=cfg.MODEL.FPN.OUT_CHANNELS,
                top_block=LastLevelMaxPool(),
                )
    return model

if __name__ == "__main__":
    model = StarNet(depth=4, dim=24, act=nn.ReLU, mode="mul")
    x = torch.randn(1, 3, 224, 224)
    from thop import profile
    from thop import clever_format
    input = torch.randn(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input, ))
    flops = macs * 2
    b, params = clever_format([flops, params], "%.3f") 
    print(b, params)
