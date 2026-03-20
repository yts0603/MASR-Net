#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

def downsample2x(x):
    """2倍下采样函数"""
    return F.interpolate(x, scale_factor=0.5, mode='trilinear', align_corners=False)


class SimpleSwinBlock3d(nn.Module):
    """3D Swin Transformer块"""
    def __init__(self, dim, num_heads=4, window_size=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # 使用多头注意力
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        
        self.window_size = window_size
    
    def window_partition(self, x, window_size):
        B, C, D, H, W = x.shape
        x = x.view(B, C, D//window_size, window_size, H//window_size, window_size, W//window_size, window_size)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x = x.view(-1, window_size*window_size*window_size, C)
        return x
    
    def window_reverse(self, windows, window_size, D, H, W):
        B = int(windows.shape[0] / (D * H * W / window_size**3))
        x = windows.view(B, D//window_size, H//window_size, W//window_size, 
                         window_size, window_size, window_size, -1)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(B, -1, D, H, W)
        return x
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        shortcut = x
        
        # 窗口划分
        x_windows = self.window_partition(x, self.window_size)
        
        # LayerNorm + Attention
        x_windows_norm = self.norm1(x_windows)
        x_attn = self.attn(x_windows_norm, x_windows_norm, x_windows_norm)[0]
        x_windows = x_windows + x_attn
        
        # 窗口还原
        x = self.window_reverse(x_windows, self.window_size, D, H, W)
        
        # 重塑为MLP需要的形状
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = x.view(B, -1, C)
        
        # LayerNorm + MLP
        x = x + self.mlp(self.norm2(x))
        
        # 恢复3D形状
        x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        
        # 添加残差连接
        x = x + shortcut
        return x


class SwinUNet3D(nn.Module):
    """保持与原始UNet相同的架构"""
    def __init__(self):
        super().__init__()
        
        # 编码器 - 完全匹配原始UNet的架构
        self.enc1 = self._double_conv(1, 16)
        self.enc2 = self._double_conv(16, 32)
        self.enc3 = self._double_conv(32, 64)
        self.enc4 = self._double_conv(64, 128)
        
        # 在深层添加Swin Blocks（enc3和enc4）
        self.swin3 = SimpleSwinBlock3d(64)   # 在enc3后添加
        self.swin4 = SimpleSwinBlock3d(128)  # 在enc4后添加
        
        # 瓶颈层 - 用Swin代替原始的自注意力
        self.bottleneck = nn.Sequential(
            self._double_conv(128, 256),
            SimpleSwinBlock3d(256),  # 替代SelfAttention
            self._double_conv(256, 256)
        )
        
        # 解码器 
        self.dec4 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, 2, stride=2),
            self._double_conv(256, 128)  # 128(上采样) + 128(skip) = 256
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, 2, stride=2),
            self._double_conv(128, 64)  # 64 + 64 = 128
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 2, stride=2),
            self._double_conv(64, 32)  # 32 + 32 = 64
        )
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 2, stride=2),
            self._double_conv(32, 16)  # 16 + 16 = 32
        )
        
        # 最终上采样
        self.final_up = nn.Sequential(
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
        # 编码器
        c1 = self.enc1(x)                              # (B,16,128,128,64)
        p1 = F.max_pool3d(c1, 2)                       # (B,16,64,64,32)
        
        c2 = self.enc2(p1)                             # (B,32,64,64,32)
        p2 = F.max_pool3d(c2, 2)                       # (B,32,32,32,16)
        
        c3 = self.enc3(p2)                             # (B,64,32,32,16)
        c3 = self.swin3(c3)                            # Swin处理
        p3 = F.max_pool3d(c3, 2)                       # (B,64,16,16,8)
        
        c4 = self.enc4(p3)                             # (B,128,16,16,8)
        c4 = self.swin4(c4)                            # Swin处理
        p4 = F.max_pool3d(c4, 2)                       # (B,128,8,8,4)
        
        # 瓶颈层
        bn = self.bottleneck(p4)                       # (B,256,8,8,4)
        
        # 解码器
        up4 = self.dec4[0](bn)                         # (B,128,16,16,8)
        if up4.shape[2:] != c4.shape[2:]:
            up4 = F.interpolate(up4, size=c4.shape[2:], mode='trilinear', align_corners=False)
        d4 = self.dec4[1](torch.cat([up4, c4], dim=1)) # (B,256,16,16,8) -> (B,128,16,16,8)
        
        up3 = self.dec3[0](d4)                         # (B,64,32,32,16)
        if up3.shape[2:] != c3.shape[2:]:
            up3 = F.interpolate(up3, size=c3.shape[2:], mode='trilinear', align_corners=False)
        d3 = self.dec3[1](torch.cat([up3, c3], dim=1)) # (B,128,32,32,16) -> (B,64,32,32,16)
        
        up2 = self.dec2[0](d3)                         # (B,32,64,64,32)
        if up2.shape[2:] != c2.shape[2:]:
            up2 = F.interpolate(up2, size=c2.shape[2:], mode='trilinear', align_corners=False)
        d2 = self.dec2[1](torch.cat([up2, c2], dim=1)) # (B,64,64,64,32) -> (B,32,64,64,32)
        
        up1 = self.dec1[0](d2)                         # (B,16,128,128,64)
        if up1.shape[2:] != c1.shape[2:]:
            up1 = F.interpolate(up1, size=c1.shape[2:], mode='trilinear', align_corners=False)
        d1 = self.dec1[1](torch.cat([up1, c1], dim=1)) # (B,32,128,128,64) -> (B,16,128,128,64)
        
        # 最终上采样
        out = self.final_up(d1)                        # (B,1,256,256,128)
        return out




