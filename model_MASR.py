#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba 

def downsample2x(x):
    """2倍下采样函数"""
    return F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=False)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv3d(in_channels, in_channels // 8, 1)
        self.key   = nn.Conv3d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv3d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, D, H, W = x.shape
        q = self.query(x).view(B, -1, D*H*W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, D*H*W)
        v = self.value(x).view(B, -1, D*H*W)
        attn = F.softmax(q @ k / (C**0.5), dim=-1)
        out = (v @ attn.permute(0, 2, 1)).view(B, C, D, H, W)
        return self.gamma * out + x


class MambaBlock3D(nn.Module):
    """3D Mamba块"""
    def __init__(self, in_channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.in_channels = in_channels
        self.d_inner = in_channels * expand
        
        # 归一化
        self.norm = nn.BatchNorm3d(in_channels)
        
        # Mamba核心
        self.mamba = Mamba(
            d_model=in_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=1
        )
        
        self.output_proj = nn.Identity() 
        # 激活函数
        self.activation = nn.ReLU()
        
    def forward(self, x):
        B, C, D, H, W = x.shape
        residual = x
        
        # 归一化
        x_norm = self.norm(x)
        
        # 将3D特征转换为序列
        x_flat = x_norm.reshape(B, C, -1).transpose(1, 2)  # (B, D*H*W, C)
        
        # Mamba处理
        x_mamba = self.mamba(x_flat)
        
        # 恢复3D形状
        x_3d = x_mamba.transpose(1, 2).reshape(B, -1, D, H, W)  
        #添加输出投影
        x_out = self.output_proj(x_3d)  # 投影回原始维度
        
        # 残差连接
        return self.activation(x_out + residual)
       

class MASR_Net(nn.Module):
    """混合Mamba和自注意力的UNet3D"""
    def __init__(self, mamba_config=None):
        super().__init__()
        
        # Mamba配置
        if mamba_config is None:
            mamba_config = {
                'd_state': 16,
                'd_conv': 4,
            }
        
        # 第一层：卷积 + Mamba
        self.enc1_conv = self._double_conv(1, 16)
        self.enc1_mamba = MambaBlock3D(
            16, 
            d_state=mamba_config['d_state'],
            d_conv=mamba_config['d_conv'],
        )
        
        # 第二层：卷积 + Mamba
        self.enc2_conv = self._double_conv(16, 32)
        self.enc2_mamba = MambaBlock3D(
            32,
            d_state=mamba_config['d_state'],
            d_conv=mamba_config['d_conv'],
        )
        
        # 第三层：传统卷积层
        self.enc3 = self._double_conv(32, 64)
        
        # 第四层：传统卷积层
        self.enc4 = self._double_conv(64, 128)
        
        # 瓶颈层：自注意力
        self.bottleneck = nn.Sequential(
            self._double_conv(128, 256),
            SelfAttention(256),  
            self._double_conv(256, 256)
        )
        
        # 解码器
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
        
        # 最终上采样层
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose3d(16, 16, 2, stride=2),
            nn.Conv3d(16, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, 1)
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
        # 第一层：卷积 -> Mamba
        c1 = self.enc1_conv(x)  # (B,16,128,128,64)
        c1 = self.enc1_mamba(c1)  # Mamba处理
    
        # 第二层：下采样 -> 卷积 -> Mamba
        p1 = F.max_pool3d(c1, 2)
        c2 = self.enc2_conv(p1)  # (B,32,64,64,32)
        c2 = self.enc2_mamba(c2)  # Mamba处理
        
        # 第三层：传统编码器
        c3 = self.enc3(F.max_pool3d(c2, 2))  # (B,64,32,32,16)
        
        # 第四层：传统编码器
        c4 = self.enc4(F.max_pool3d(c3, 2))  # (B,128,16,16,8)
        
        # 瓶颈层：自注意力
        bn = self.bottleneck(F.max_pool3d(c4, 2))  # (B,256,8,8,4)
        
        # 解码器（保持原始设计）
        up4 = self.dec4[0](bn)
        if up4.shape[2:] != c4.shape[2:]:
            up4 = F.interpolate(up4, size=c4.shape[2:], mode='trilinear', align_corners=False)
        d4 = self.dec4[1](torch.cat([up4, c4], dim=1))
        
        up3 = self.dec3[0](d4)
        if up3.shape[2:] != c3.shape[2:]:
            up3 = F.interpolate(up3, size=c3.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.dec3[1](torch.cat([up3, c3], dim=1))
        
        up2 = self.dec2[0](d3)
        if up2.shape[2:] != c2.shape[2:]:
            up2 = F.interpolate(up2, size=c2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2[1](torch.cat([up2, c2], dim=1))
        
        up1 = self.dec1[0](d2)
        if up1.shape[2:] != c1.shape[2:]:
            up1 = F.interpolate(up1, size=c1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1[1](torch.cat([up1, c1], dim=1))
        
        # 最终上采样
        output = self.final_upsample(d1)
        return output


