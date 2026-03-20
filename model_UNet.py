#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

def downsample2x(x):
    """2倍下采样函数"""
    return F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=False)


class UNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = self._double_conv(1, 16)
        self.enc2 = self._double_conv(16, 32)
        self.enc3 = self._double_conv(32, 64)
        self.enc4 = self._double_conv(64, 128)
        self.bottleneck = self._double_conv(128, 256)  # 简化为双层卷积
        self.dec4 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, stride=2),
            self._double_conv(256, 128)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, stride=2),
            self._double_conv(128, 64)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, stride=2),
            self._double_conv(64, 32)
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 2, stride=2),
            self._double_conv(32, 16)
        )
        
        # 添加最终的上采样层来实现2倍超分辨率
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose3d(16, 16, 2, stride=2),  # 2倍上采样
            nn.Conv3d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, 1)  # 输出通道为1
        )

    @staticmethod
    def _double_conv(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), 
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器
        c1 = self.enc1(x)                              # (B,16,128,128,64)
        c2 = self.enc2(F.max_pool3d(c1, 2))            # (B,32,64,64,32)
        c3 = self.enc3(F.max_pool3d(c2, 2))            # (B,64,32,32,16)
        c4 = self.enc4(F.max_pool3d(c3, 2))            # (B,128,16,16,8)
        bn = self.bottleneck(F.max_pool3d(c4, 2))      # (B,256,8,8,4)
        
        # 解码器
        up4 = self.dec4[0](bn)                         # (B,128,16,16,8)
        if up4.shape[2:] != c4.shape[2:]:
            up4 = F.interpolate(up4, size=c4.shape[2:], mode='trilinear', align_corners=False)
        d4 = self.dec4[1](torch.cat([up4, c4], dim=1)) # (B,128,16,16,8)
        
        up3 = self.dec3[0](d4)                         # (B,64,32,32,16)
        if up3.shape[2:] != c3.shape[2:]:
            up3 = F.interpolate(up3, size=c3.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.dec3[1](torch.cat([up3, c3], dim=1)) # (B,64,32,32,16)
        
        up2 = self.dec2[0](d3)                         # (B,32,64,64,32)
        if up2.shape[2:] != c2.shape[2:]:
            up2 = F.interpolate(up2, size=c2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2[1](torch.cat([up2, c2], dim=1)) # (B,32,64,64,32)
        
        up1 = self.dec1[0](d2)                         # (B,16,128,128,64)
        if up1.shape[2:] != c1.shape[2:]:
            up1 = F.interpolate(up1, size=c1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1[1](torch.cat([up1, c1], dim=1)) # (B,16,128,128,64)
        
        # 最终2倍上采样到目标尺寸 (B,1,256,256,128)
        output = self.final_upsample(d1)
        return output
