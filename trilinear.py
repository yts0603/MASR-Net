#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class TrilinearInterpolation3D(nn.Module):
    """
    三线性插值模型 - 作为传统超分辨率方法的对比
    输入: (B, 1, 128, 128, 64)
    输出: (B, 1, 256, 256, 128)
    
    注意：这是一个非学习的方法，没有可训练参数
    """
    
    def __init__(self, scale_factor=2):
        super(TrilinearInterpolation3D, self).__init__()
        self.scale_factor = scale_factor
        
    def forward(self, x):
        """
        前向传播 - 应用三线性插值
        
        参数:
            x: 输入张量, 形状 (B, 1, 128, 128, 64)
            
        返回:
            output: 输出张量, 形状 (B, 1, 256, 256, 128)
        """
        # 应用三线性插值进行2倍上采样
        output = F.interpolate(x, 
                             scale_factor=self.scale_factor, 
                             mode='trilinear', 
                             align_corners=False)
        return output
    
    def __repr__(self):
        return f"TrilinearInterpolation3D(scale_factor={self.scale_factor})"


# 辅助函数：下采样函数（与你的UNet代码保持一致）
def downsample2x(x):
    """2倍下采样函数"""
    return F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=False)