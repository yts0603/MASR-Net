import os
import numpy as np
import nibabel as nib
from scipy import ndimage
from tqdm import tqdm

def resample_to_uniform_spacing(input_path, output_dir, target_spacing=(1.0, 1.0, 1.0), order=3):
    """
    将NIfTI图像重采样到统一的体素间距
    
    参数:
        input_path: 输入NIfTI文件或目录路径
        output_dir: 输出目录
        target_spacing: 目标体素间距(mm), 默认为(1.0, 1.0, 1.0)
        order: 插值阶数 (0=最近邻, 1=线性, 3=三次样条)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有NIfTI文件
    if os.path.isfile(input_path):
        nifti_files = [input_path]
    else:
        nifti_files = []
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.endswith('.nii') or f.endswith('.nii.gz'):
                    nifti_files.append(os.path.join(root, f))
    
    print(f"找到 {len(nifti_files)} 个NIfTI文件")
    
    # 处理每个文件
    for file_path in tqdm(nifti_files, desc="处理文件中"):
        try:
            # 加载NIfTI文件
            img = nib.load(file_path)
            data = img.get_fdata()
            affine = img.affine
            header = img.header
            
            # 获取原始体素间距
            original_spacing = np.array(header.get_zooms()[:3])
            print(f"\n处理文件: {os.path.basename(file_path)}")
            print(f"原始体素间距: {original_spacing} mm")
            print(f"原始数据形状: {data.shape}")
            
            # 计算重采样比例
            resize_factor = original_spacing / np.array(target_spacing)
            new_shape = np.round(data.shape * resize_factor).astype(int)
            print(f"重采样比例: {resize_factor}")
            print(f"新数据形状: {new_shape}")
            
            # 执行重采样
            resampled_data = ndimage.zoom(data, resize_factor, order=order)
            
            # 更新affine矩阵
            new_affine = affine.copy()
            new_affine[:3, :3] = affine[:3, :3] @ np.diag(1/resize_factor)
            
            # 保存结果
            output_filename = os.path.basename(file_path).replace('.nii', '_resampled.nii')
            output_path = os.path.join(output_dir, output_filename)
            
            new_img = nib.Nifti1Image(resampled_data, new_affine, header)
            nib.save(new_img, output_path)
            
            print(f"已保存到: {output_path}")
            print(f"处理后体素间距: {target_spacing} mm")
            print(f"处理后数据形状: {resampled_data.shape}")
            
        except Exception as e:
            print(f"处理 {file_path} 时出错: {str(e)}")
    
    print("\n所有文件处理完成！")

# 使用示例
if __name__ == "__main__":
    input_path = "/data/yangtianshu/reconstruction/infant-data/T1w/"
    output_dir = "/data/yangtianshu/reconstruction/infant-data/T1w_resampled/"
    
    # 设置目标体素间距为1mm各向同性
    target_spacing = (1.0, 1.0, 1.0)
    
    # 执行重采样 (使用三次样条插值)
    resample_to_uniform_spacing(input_path, output_dir, target_spacing, order=3)