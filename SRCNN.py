#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

def downsample2x(x):
    """2倍下采样函数"""
    return F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=False)        

class SRCNN3D(nn.Module):
    """3D SRCNN，添加内置上采样以匹配你的UNet3D"""
    def __init__(self, num_channels=1):
        super().__init__()
        # 内置上采样层
        self.upsample = nn.ConvTranspose3d(num_channels, num_channels, kernel_size=2, stride=2)
        
        # SRCNN的三个卷积层
        self.conv1 = nn.Conv3d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv3d(32, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # 先上采样
        x_up = self.upsample(x)
        
        # SRCNN处理（学习残差）
        residual = self.relu(self.conv1(x_up))
        residual = self.relu(self.conv2(residual))
        residual = self.conv3(residual)
        
        # 残差连接
        return x_up + residual