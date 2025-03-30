import torch
import torch.nn as nn
from torch.nn import functional as F

class SCAFM(nn.Module):
    def __init__(self, in_ch_high, in_ch_low, mode, out_c):
        """
        mode: is the method of upsample. In this class we supported 2 different methods 
        1) bilinear interpolation
        2) deep convolution -> DUpsampling, which inspired the "https://en.wikipedia.org/wiki/"
        Args: 'bilinear' or 'dl', and default was bilinear interpolation
        """
        super().__init__()
        if mode == 'bilinear':
            self.upx2_1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.upx2_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        if mode == 'dl':
            assert in_ch_high == out_c or in_ch_high == int(2*out_c),'man must is this.'
            self.upx2_1 = DUpsampling(inplanes=in_ch_high, out_c=out_c, scale=2)
            self.upx2_2 = DUpsampling(inplanes=in_ch_high, out_c=out_c, scale=2)

        self.convx1_in_high = nn.Conv2d(in_ch_high, in_ch_high, 1, 1)
        self.convx1_in_low  = nn.Conv2d(in_ch_low, in_ch_low, 1, 1) 

        self.sam = SAM()
        self.cam = CAM(out_c=in_ch_low)
        
        in_c_at_in = int(in_ch_low*4)
        out_c = int(in_ch_low)
        self.in_conv = nn.Conv2d(in_c_at_in, out_c, 1, 1)

        in_c_at_out = int(in_ch_low*3)
        self.out_conv = nn.Conv2d(in_c_at_out, out_c, 1, 1) 

    def forward(self, f_h, f_l):
        # for f_h
        fh = self.upx2_1(self.convx1_in_high(f_h))
        fup = self.upx2_2(f_h)

        # for f_l
        fconv = self.convx1_in_low(f_l)
        fl = f_l

        asp, neg_asp = self.sam(fup, fconv)
        fsp = (fup * asp) + (fconv * neg_asp)
        
        ach, neg_ach = self.cam(fup, fconv)
        fch = (fup * ach) + (fconv * neg_ach)
        
        fsc = self.in_conv(torch.cat([fh, fl, fsp, fch], dim=1))
        fout = self.out_conv(torch.cat([fh, fl, fsc], dim=1))
        return fout


class SAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 1, 1, 1)
        self.act   = nn.Sigmoid()

    def forward(self, f_up, f_conv):
        f_up_mean      = torch.mean(f_up, dim=1, keepdim=True) # w,h,c -> w,h,1
        f_up_max, _    = torch.max(f_up, dim=1, keepdim=True)  # w,h,c -> w,h,1
        f_conv_mean    = torch.mean(f_conv, dim=1, keepdim=True) # w,h,c -> w,h,1
        f_conv_max, _  = torch.max(f_conv, dim=1, keepdim=True)  # w,h,c -> w,h,1
        cat_out = torch.cat([f_up_mean, f_up_max, f_conv_mean, f_conv_max], dim=1)
        out = self.act(self.conv1(cat_out))
        return out, 1-out
        
class CAM(nn.Module):
    def __init__(self, out_c):
        super().__init__()
        in_c = int(4*out_c)
        self.conv1 = nn.Conv2d(in_c, out_c, 1, 1)
        self.act   = nn.Sigmoid()

    def forward(self, f_up, f_conv):
        f_up_mean   = F.adaptive_avg_pool2d(f_up,(1,1)) # w,h,c -> 1,1,c
        f_up_max    = F.adaptive_max_pool2d(f_up,(1,1)) # w,h,c -> 1,1,c
        f_conv_mean = F.adaptive_avg_pool2d(f_conv,(1,1)) # w,h,c -> 1,1,c
        f_conv_max  = F.adaptive_max_pool2d(f_conv,(1,1)) # w,h,c -> 1,1,c
        cat_out     = torch.cat([f_up_mean, f_up_max, f_conv_mean, f_conv_max], dim=1)
        out         = self.act(self.conv1(cat_out))
        return out, 1-out

class DUpsampling(nn.Module):
    # which was implemented in https://github.com/haochange/DUpsampling/blob/master/models/dunet.py
    def __init__(self, inplanes, scale, out_c, pad=0):
        super(DUpsampling, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 
                            int(out_c * scale * scale), 
                            kernel_size=1, 
                            padding = pad,
                            bias=False)
        self.scale = scale
    
    def forward(self, x):
        x = self.conv1(x)
        N, C, H, W = x.size()

        # N, H, W, C
        x_permuted = x.permute(0, 2, 3, 1) 

        # N, H, W*scale, C/scale
        x_permuted = x_permuted.contiguous().view((N, H, int(W * self.scale), int(C / (self.scale))))

        # N, W*scale,H, C/scale
        x_permuted = x_permuted.permute(0, 2, 1, 3)
        # N, W*scale,H*scale, C/(scale**2)
        x_permuted = x_permuted.contiguous().view((N, int(W * self.scale), int(H * self.scale), int(C / (self.scale * self.scale))))

        # N,C/(scale**2),W*scale,H*scale
        x = x_permuted.permute(0, 3, 2, 1)
        
        return x

if __name__ == "__main__":
    # sam = CAM()
    in_ch_high, in_ch_low, mode = 12,6,'dl'
    out_c = 6
    scafm = SCAFM(in_ch_high, in_ch_low, mode, out_c)
    f_up = torch.randn(1, 12, 32, 32)
    f_conv = torch.randn(1, 6, 64, 64)
    # print(sam(f_up, f_conv).shape)
    print(scafm(f_up, f_conv).shape)