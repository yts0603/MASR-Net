#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, glob, json, sys
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import pandas as pd
import time

# 添加训练代码所在目录到Python路径
sys.path.append("/data/yangtianshu/reconstruction")
from model import UNet3D, downsample2x  
model_name = "SwinUNet3D"
# ---------------- 测试配置 ----------------
test_cfg = {
    "test_data_dir": "/data/yangtianshu/reconstruction/infant-data/test_normalized/",
    "model_path":"/data/yangtianshu/reconstruction/train_result/noattention/epo30/model30/best.pth",
    "result_dir": "/data/yangtianshu/reconstruction/test-result/noattention/epo30/",
    "batch_size": 1,
    "device": "cuda:2" if torch.cuda.is_available() else "cpu",
    "background_threshold": 0.0005,  # 背景归零阈值
}


os.makedirs(os.path.join(test_cfg["result_dir"], "reconstructed"), exist_ok=True)
os.makedirs(os.path.join(test_cfg["result_dir"], "metrics"), exist_ok=True)

# ---------------- 测试数据集 ----------------
class TestDataset(Dataset):
    def __init__(self, test_dir):
        self.files = sorted(glob.glob(os.path.join(test_dir, "*.nii*")))
        print(f"找到 {len(self.files)} 个测试文件")
    
    def __getitem__(self, idx):
        hr_img   = nib.load(self.files[idx])
        hr_np    = hr_img.get_fdata(dtype=np.float32)
        affine   = hr_img.affine
        
        # 转换为PyTorch张量
        hr = torch.from_numpy(hr_np).unsqueeze(0).float()
        
        # 生成低分辨率输入（用于模型推理）
        lr = downsample2x(hr.unsqueeze(0)).squeeze(0)
        
        affine_tensor = torch.from_numpy(affine.astype(np.float32))
        
        return lr, hr, os.path.basename(self.files[idx]), affine_tensor
    
    def __len__(self): 
        return len(self.files)

def apply_background_zeroing(pred_np, target_np, threshold):
    """背景归零后处理"""
    # 创建背景掩码（基于真实图像）
    background_mask = target_np < threshold
    
    # 同时对两个图像应用相同的掩码
    pred_processed = pred_np.copy()
    target_processed = target_np.copy()
    
    pred_processed[background_mask] = 0.0
    target_processed[background_mask] = 0.0
    
    return pred_processed, target_processed

# ---------------- 评估指标计算 ----------------
def calculate_metrics(pred, target, bg_threshold):
    """计算评估指标，应用背景归零后处理"""
    pred_np = pred.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()
    
    # 应用背景归零后处理
    pred_np, target_np = apply_background_zeroing(pred_np, target_np, bg_threshold)
    
    # PSNR
    psnr_val = psnr(target_np, pred_np, data_range=1.0)

    # SSIM (3D)
    min_edge = min(target_np.shape)
    win = min(7, min_edge // 2 * 2 + 1)
    ssim_val = ssim(target_np, pred_np,
                    data_range=1.0,
                    win_size=win,
                    channel_axis=0)

    # MSE
    mse_val = np.mean((target_np - pred_np) ** 2)

    return {
        'psnr': float(psnr_val), 
        'ssim': float(ssim_val), 
        'mse': float(mse_val)
    }

def append_to_excel(overall_metrics, test_cfg):
    """每次测试追加一行到 Excel"""
    file_path = os.path.join(test_cfg["result_dir"], "metrics", "test_history.xlsx")
    
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": os.path.basename(test_cfg["model_path"]),
        "threshold": test_cfg["background_threshold"],
        "PSNR_mean": float(overall_metrics["psnr"]["mean"]),
        "PSNR_std": float(overall_metrics["psnr"]["std"]),
        "SSIM_mean": float(overall_metrics["ssim"]["mean"]),
        "SSIM_std": float(overall_metrics["ssim"]["std"]),
        "MSE_mean": float(overall_metrics["mse"]["mean"]),
        "MSE_std": float(overall_metrics["mse"]["std"]),
    }
    
    # 读取现有数据或创建新DataFrame
    if os.path.exists(file_path):
        df_old = pd.read_excel(file_path)
        
        # 查找是否有相同时间戳的记录
        mask = df_old["timestamp"] == row["timestamp"]
        if mask.any():
            # 更新现有记录
            for col, value in row.items():
                if col in df_old.columns:
                    df_old.loc[mask, col] = value
                else:
                    df_old.loc[mask, col] = value
            df_all = df_old
        else:
            # 添加新记录
            df_all = pd.concat([df_old, pd.DataFrame([row])], ignore_index=True)
    else:
        df_all = pd.DataFrame([row])
    
    df_all.to_excel(file_path, index=False, engine='openpyxl')
    print(f"已追加结果到: {file_path}")

# ---------------- 主要测试函数 ----------------
def test():
    """执行测试过程"""
    print("开始测试...")
    print(f"设备: {test_cfg['device']}")
    print(f"模型路径: {test_cfg['model_path']}")
    print(f"测试数据: {test_cfg['test_data_dir']}")
    print(f"结果保存: {test_cfg['result_dir']}")
    print(f"后处理阈值: {test_cfg['background_threshold']}")
    
    # 加载测试数据集
    test_dataset = TestDataset(test_cfg["test_data_dir"])
    test_loader = DataLoader(test_dataset, batch_size=test_cfg["batch_size"], 
                           shuffle=False, num_workers=2)
    
    # 直接使用训练代码中的模型定义
    model = UNet3D().to(test_cfg["device"])
    
    # 加载训练好的模型权重
    if os.path.exists(test_cfg["model_path"]):
        checkpoint = torch.load(test_cfg["model_path"], map_location=test_cfg["device"])
        model.load_state_dict(checkpoint["model"])
        print(f"成功加载模型权重，训练epoch: {checkpoint.get('epoch', '未知')}")
        print(f"最佳损失: {checkpoint.get('loss', '未知'):.6f}")
    else:
        raise FileNotFoundError(f"模型文件不存在: {test_cfg['model_path']}")
    
    # 设置为评估模式
    model.eval()
    
    # 存储所有测试结果
    all_metrics = {}
    overall_metrics = {'psnr': [], 'ssim': [], 'mse': []}
    
    print("\n开始推理并应用后处理...")
    with torch.no_grad():
        for batch_idx, (lr, hr, filename, affine) in enumerate(tqdm(test_loader, desc=f"测试进度 (共{len(test_loader)}个)")):
            lr = lr.to(test_cfg["device"])
            hr = hr.to(test_cfg["device"])
            
            # 模型推理
            pred = model(lr)
            
            # filename 是包含一个元素的列表，需要提取
            filename_str = filename[0] if isinstance(filename, (list, tuple)) and len(filename) > 0 else str(filename)
            base_name = filename_str.replace(".nii.gz", "").replace(".nii", "")
            
            # affine 是张量，需要转换为numpy并确保形状正确
            if isinstance(affine, torch.Tensor):
                affine_np = affine.squeeze().numpy()  # 移除批次维度
            else:
                affine_np = affine
            
            # 确保affine是4x4矩阵
            if affine_np.shape != (4, 4):
                print(f"警告: affine形状为 {affine_np.shape}，调整为 (4,4)")
                if affine_np.size >= 16:
                    # 取前16个元素重塑为4x4
                    affine_np = affine_np.flatten()[:16].reshape(4, 4)
                else:
                    # 如果元素不足，使用单位矩阵
                    print("使用单位矩阵作为affine")
                    affine_np = np.eye(4) 
            
            # 应用后处理
            pred_np = pred.squeeze().cpu().numpy()
            hr_np = hr.squeeze().cpu().numpy()
            
            pred_np_processed, hr_np_processed = apply_background_zeroing(
                pred_np, hr_np, test_cfg["background_threshold"]
            )
            
            # 计算后处理指标
            pred_processed = torch.from_numpy(pred_np_processed).unsqueeze(0).float()
            hr_processed = torch.from_numpy(hr_np_processed).unsqueeze(0).float()
            metrics = calculate_metrics(pred_processed, hr_processed, test_cfg["background_threshold"])
            
            # 保存指标
            all_metrics[base_name] = metrics
            for key in overall_metrics:
                overall_metrics[key].append(metrics[key])
            
            # 保存后处理重建结果为NIfTI文件
            pred_nifti_processed = nib.Nifti1Image(pred_np_processed, affine=affine_np)
            nib.save(pred_nifti_processed, 
                    os.path.join(test_cfg["result_dir"], "reconstructed", 
                                f"{base_name}_reconstructed.nii.gz"))
    
    # 计算总体统计
    def calculate_overall_stats(metrics_list):
        """计算总体统计"""
        stats = {}
        for metric_name in metrics_list:
            values = metrics_list[metric_name]
            mean_val = np.mean(values)
            std_val = np.std(values)
            stats[metric_name] = {
                'mean': float(mean_val),
                'std': float(std_val),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        return stats
    
    overall_stats = calculate_overall_stats(overall_metrics)
    
    # 输出结果汇总
    print("\n" + "="*60)
    print("测试结果汇总 (已应用后处理):")
    print("="*60)
    
    print(f"后处理阈值: {test_cfg['background_threshold']}")
    print(f"测试图像数量: {len(test_dataset)}")
    print()
    
    for metric_name in overall_stats:
        stats = overall_stats[metric_name]
        print(f"{metric_name.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f} "
              f"(范围: {stats['min']:.4f}-{stats['max']:.4f})")
    
    # 保存详细测试结果
    results = {
        'test_config': test_cfg,
        'overall_metrics': overall_stats,
        'per_image_metrics': all_metrics,
        'test_summary': {
            'total_images': int(len(test_dataset)),
            'model_used': test_cfg["model_path"],
            'test_timestamp': str(np.datetime64('now')),
            'background_threshold': test_cfg["background_threshold"]
        }
    }
    
    # 保存JSON格式结果
    with open(os.path.join(test_cfg["result_dir"], "metrics", "detailed_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    # 保存文本格式结果
    with open(os.path.join(test_cfg["result_dir"], "metrics", "test_summary.txt"), 'w') as f:
        f.write("3D MRI超分辨率重建测试结果 (已应用后处理)\n")
        f.write("="*60 + "\n")
        f.write(f"测试时间: {results['test_summary']['test_timestamp']}\n")
        f.write(f"测试图像数量: {results['test_summary']['total_images']}\n")
        f.write(f"使用模型: {results['test_summary']['model_used']}\n")
        f.write(f"后处理阈值: {results['test_summary']['background_threshold']}\n")
        f.write("\n")
        
        f.write("评估指标:\n")
        f.write("-"*40 + "\n")
        for metric, stats in overall_stats.items():
            f.write(f"{metric.upper()}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                   f"(范围: {stats['min']:.4f}-{stats['max']:.4f})\n")
    
    # 一键追加到 Excel
    append_to_excel(overall_stats, test_cfg)
    
    print(f"\n测试完成! 结果保存在: {test_cfg['result_dir']}")
    print(f"- 重建文件: {test_cfg['result_dir']}/reconstructed/")
    print(f"- 评估指标: {test_cfg['result_dir']}/metrics/")
    print(f"- Excel记录: {test_cfg['result_dir']}/metrics/test_history.xlsx")

if __name__ == "__main__":
    # 保存测试配置
    with open(os.path.join(test_cfg["result_dir"], "test_config.json"), 'w') as f:
        json.dump(test_cfg, f, indent=2)
    
    # 执行测试
    test()