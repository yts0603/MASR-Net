#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class BilinearInterpolation3D(nn.Module):
    """
    双线性插值模型 - 作为传统超分辨率方法的对比
    输入: (B, 1, 128, 128, 64)
    输出: (B, 1, 256, 256, 128)
    """
    
    def __init__(self, scale_factor=2):
        super(BilinearInterpolation3D, self).__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        
        # 存储每个深度切片的上采样结果
        xy_upsampled_slices = []
        
        # 对每个深度切片进行2D双线性插值
        for d in range(D):
            # 取出单个深度切片，形状: (B, C, H, W) - 需要4D输入
            slice_2d = x[:, :, d, :, :]
            
            # 在XY平面进行双线性插值，输入需要是4D
            upsampled_slice = F.interpolate(
                slice_2d, 
                scale_factor=(self.scale_factor, self.scale_factor), 
                mode='bilinear', 
                align_corners=False
            )  # 形状: (B, C, 2H, 2W)
            
            # 添加回深度维度
            upsampled_slice = upsampled_slice.unsqueeze(2)  # 形状: (B, C, 1, 2H, 2W)
            
            xy_upsampled_slices.append(upsampled_slice)
        
        # 将所有切片拼接起来
        xy_upsampled = torch.cat(xy_upsampled_slices, dim=2)  # 形状: (B, C, D, 2H, 2W)
        
        # 在Z方向进行最近邻插值
        output = F.interpolate(
            xy_upsampled, 
            scale_factor=(self.scale_factor, 1, 1), 
            mode='nearest'
        )  # 形状: (B, C, 2D, 2H, 2W)
        
        return output
    
    def __repr__(self):
        return f"BilinearInterpolation3D(scale_factor={self.scale_factor})"
        
        
def downsample2x(x):
    """2倍下采样函数"""
    return F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=False)