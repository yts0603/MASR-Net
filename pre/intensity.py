# coding:utf-8
import numpy as np
import nibabel as nib
import glob
import os

def intensity_normalization(data, method='percentile'):
    """
    强度值归一化
    """
    if method == 'percentile':
        # 基于百分位数的归一化（推荐）
        p_low = np.percentile(data, 1)   # 1%分位数
        p_high = np.percentile(data, 99) # 99%分位数
        
        # 裁剪异常值
        data_clipped = np.clip(data, p_low, p_high)
        
        # 归一化到 [0, 1]
        if p_high > p_low:
            normalized = (data_clipped - p_low) / (p_high - p_low)
        else:
            normalized = data_clipped - p_low
            
    elif method == 'zscore':
        # Z-score 归一化
        mean = np.mean(data)
        std = np.std(data)
        if std > 0:
            normalized = (data - mean) / std
        else:
            normalized = data - mean
            
    elif method == 'minmax':
        # 最小-最大归一化
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            normalized = (data - data_min) / (data_max - data_min)
        else:
            normalized = data - data_min
    
    return normalized

    
def apply_intensity_normalization_batch(input_dir, output_dir, method='percentile'):
    """
    批量应用强度值归一化
    """
    os.makedirs(output_dir, exist_ok=True)
    
        
    for file_path in glob.glob(os.path.join(input_dir, "*.nii*")):
        try:
            # 加载图像
            img = nib.load(file_path)
            data = img.get_fdata()
            affine = img.affine
            
            # 应用强度值归一化
            normalized_data = intensity_normalization(data, method=method)
            
            
            # 保存结果normalized
            output_img = nib.Nifti1Image(normalized_data, affine)
            output_filename = os.path.basename(file_path)
            nib.save(output_img, os.path.join(output_dir, output_filename))
            
            print(f"强度值归一化完成: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"错误: {os.path.basename(file_path)} - {e}")

# 应用强度值归一化
apply_intensity_normalization_batch(
    "/data/yangtianshu/reconstruction/infant-data/T1w_resized/",
    "/data/yangtianshu/reconstruction/infant-data/T1w_normalized/",
    method='percentile'
)