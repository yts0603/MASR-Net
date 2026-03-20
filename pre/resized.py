# coding:utf-8
import numpy as np
import nibabel as nib
import glob
import os

def simple_resize_3d(input_dir, output_dir, target=(256, 256, 128)):
    os.makedirs(output_dir, exist_ok=True)
    
    for file_path in glob.glob(input_dir + "*.nii*"):
        try:
        
            img = nib.load(file_path)
            data, affine = img.get_fdata(), img.affine
            
           
            for i in range(3):
                curr, targ = data.shape[i], target[i]
                if curr > targ:  
                    start = (curr - targ) // 2
                    slices = [slice(None)] * 3
                    slices[i] = slice(start, start + targ)
                    data = data[tuple(slices)]
                elif curr < targ: 
                    pad = [(0, 0)] * 3
                    pad_before = (targ - curr) // 2
                    pad[i] = (pad_before, targ - curr - pad_before)
                    data = np.pad(data, pad, mode='constant')
          
            name = os.path.basename(file_path)
            out_name = name.replace('.nii', '_resized.nii')
            nib.save(nib.Nifti1Image(data, affine), os.path.join(output_dir, out_name))
            print(f"处理完成: {name} -> {data.shape}")
            
        except Exception as e:
            print(f"错误: {os.path.basename(file_path)} - {e}")


simple_resize_3d(
    "/data/yangtianshu/reconstruction/infant-data/T1w_resampled/",
    "/data/yangtianshu/reconstruction/infant-data/T1w_resized/"
)