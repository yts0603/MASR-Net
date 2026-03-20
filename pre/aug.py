#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import glob
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

def downsample2x(x):
    return F.interpolate(x, scale_factor=0.5, mode='trilinear',
                         align_corners=False)

def augment_and_save_images():
    """扩充图像并保存到指定目录"""
    # 输入和输出目录
    input_dir = "/data/yangtianshu/reconstruction/infant-data/train_normalized/"
    output_dir = "/data/yangtianshu/reconstruction/infant-data/train-ex/"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有原始图像文件
    original_files = sorted(glob.glob(os.path.join(input_dir, "*.nii*")))
    print(f"找到 {len(original_files)} 个原始训练文件")
    
    # 统计保存的图像数量
    saved_count = 0
    
    # 对每个原始图像进行5种扩充
    for file_idx, file_path in enumerate(original_files):
        # 加载原始图像
        hr_np = nib.load(file_path).get_fdata(dtype=np.float32)
        affine = nib.load(file_path).affine  # 保存仿射变换矩阵
        
        # 转换为torch张量 (1, D, H, W)
        hr = torch.from_numpy(hr_np).unsqueeze(0).float()
        
        # 生成5种扩充版本
        for aug_type in range(5):
            # 应用数据扩充
            if aug_type == 0:
                # 原始图像
                augmented_hr = hr
                aug_name = "original"
            elif aug_type == 1:
                # 左右翻转
                augmented_hr = torch.flip(hr, [3])
                aug_name = "flipped"
            elif aug_type == 2:
                # 横断面90度旋转
                augmented_hr = torch.rot90(hr, 1, [2, 3])
                aug_name = "rot90"
            elif aug_type == 3:
                # 横断面180度旋转
                augmented_hr = torch.rot90(hr, 2, [2, 3])
                aug_name = "rot180"
            elif aug_type == 4:
                # 横断面270度旋转
                augmented_hr = torch.rot90(hr, 3, [2, 3])
                aug_name = "rot270"
            
            # 转换回numpy数组
            augmented_np = augmented_hr.squeeze(0).numpy()
            
            # 生成输出文件名
            base_name = os.path.basename(file_path)
            name_without_ext = os.path.splitext(base_name)[0]
            if base_name.endswith('.gz'):
                name_without_ext = os.path.splitext(name_without_ext)[0]
            
            output_filename = f"{name_without_ext}_{aug_name}.nii.gz"
            output_path = os.path.join(output_dir, output_filename)
            
            # 保存扩充后的图像
            augmented_img = nib.Nifti1Image(augmented_np, affine)
            nib.save(augmented_img, output_path)
            
            saved_count += 1
            
            print(f"已保存: {output_filename}")
        
        # 显示进度
        if (file_idx + 1) % 10 == 0:
            print(f"已完成 {file_idx + 1}/{len(original_files)} 个原始文件的扩充")
    
    print(f"\n扩充完成!")
    print(f"原始图像数量: {len(original_files)}")
    print(f"扩充后图像数量: {saved_count}")
    print(f"输出目录: {output_dir}")

def verify_augmentation():
    """验证扩充结果"""
    input_dir = "/data/yangtianshu/reconstruction/infant-data/train/"
    output_dir = "/data/yangtianshu/reconstruction/infant-data/train-ex/"
    
    original_files = sorted(glob.glob(os.path.join(input_dir, "*.nii*")))
    augmented_files = sorted(glob.glob(os.path.join(output_dir, "*.nii*")))
    
    print(f"原始图像数量: {len(original_files)}")
    print(f"扩充后图像数量: {len(augmented_files)}")
    print(f"扩充倍数: {len(augmented_files) / len(original_files):.1f}x")
    
    # 检查文件命名
    print("\n前5个扩充文件示例:")
    for i in range(min(5, len(augmented_files))):
        print(f"  {os.path.basename(augmented_files[i])}")

if __name__ == "__main__":
    # 执行图像扩充
    augment_and_save_images()
    
    print("\n" + "="*50)
    
    # 验证扩充结果
    verify_augmentation()