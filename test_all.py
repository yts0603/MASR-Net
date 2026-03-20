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
import shutil
from datetime import datetime

# 添加训练代码所在目录到Python路径
sys.path.append("/data/yangtianshu/reconstruction")
from model_MASR import downsample2x, MASR_Net 

model_name = "MASR-Net"

# ---------------- 测试配置 ----------------
test_cfg = {
    "test_data_dir": "/data/yangtianshu/reconstruction/infant-data/test_normalized/",
    "model_dir": "/data/yangtianshu/reconstruction/train_result/MASR/1.28_epo50/MASR_Net_model50/epoch_checkpoints/",
    "result_dir": "/data/yangtianshu/reconstruction/test-result/MASR/1.28_epo50all/",
    "batch_size": 1,
    "device": "cuda:2" if torch.cuda.is_available() else "cpu",
    "background_threshold": 0.0005,  # 背景归零阈值
    "num_epochs_to_test": None,  # None表示测试所有，否则指定数量
    "start_epoch": 0,  # 从哪个epoch开始测试
    "end_epoch": None,  # 到哪个epoch结束测试
}

# 创建结果目录
os.makedirs(test_cfg["result_dir"], exist_ok=True)
os.makedirs(os.path.join(test_cfg["result_dir"], "best_model"), exist_ok=True)
os.makedirs(os.path.join(test_cfg["result_dir"], "metrics"), exist_ok=True)

# ---------------- 测试数据集 ----------------
class TestDataset(Dataset):
    def __init__(self, test_dir):
        self.files = sorted(glob.glob(os.path.join(test_dir, "*.nii*")))
        print(f"找到 {len(self.files)} 个测试文件")
    
    def __getitem__(self, idx):
        hr_img = nib.load(self.files[idx])
        hr_np = hr_img.get_fdata(dtype=np.float32)
        affine = hr_img.affine
        
        # 转换为PyTorch张量
        hr = torch.from_numpy(hr_np).unsqueeze(0).float()
        
        # 生成低分辨率输入（用于模型推理）
        lr = downsample2x(hr.unsqueeze(0)).squeeze(0)
        
        return lr, hr, os.path.basename(self.files[idx]), affine
    
    def __len__(self): 
        return len(self.files)

def apply_background_zeroing(pred_np, target_np, threshold):
    """背景归零后处理"""
    background_mask = target_np < threshold
    pred_processed = pred_np.copy()
    target_processed = target_np.copy()
    pred_processed[background_mask] = 0.0
    target_processed[background_mask] = 0.0
    return pred_processed, target_processed

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

# ---------------- 查找并排序epoch模型 ----------------
def find_epoch_models(model_dir):
    """查找所有epoch的模型文件并排序"""
    model_files = glob.glob(os.path.join(model_dir, "epoch_*.pth"))
    
    epoch_models = []
    for model_file in model_files:
        filename = os.path.basename(model_file)
        
        # 解析epoch信息: epoch_000_loss_0.123456.pth
        try:
            parts = filename.split("_")
            if len(parts) >= 4 and parts[0] == "epoch" and parts[2] == "loss":
                epoch_num = int(parts[1])
                loss_val = float(parts[3].replace(".pth", ""))
                
                epoch_models.append({
                    "path": model_file,
                    "epoch": epoch_num,
                    "loss": loss_val,
                    "filename": filename
                })
        except (ValueError, IndexError):
            continue
    
    # 按epoch排序
    epoch_models.sort(key=lambda x: x["epoch"])
    return epoch_models

# ---------------- 主测试函数 ----------------
def test_all_epochs():
    """批量测试所有epoch模型"""
    print("="*60)
    print("开始批量测试所有epoch模型")
    print("="*60)
    print(f"设备: {test_cfg['device']}")
    print(f"模型目录: {test_cfg['model_dir']}")
    print(f"测试数据: {test_cfg['test_data_dir']}")
    print(f"结果保存: {test_cfg['result_dir']}")
    print(f"背景阈值: {test_cfg['background_threshold']}")
    
    # 查找所有epoch模型
    all_models = find_epoch_models(test_cfg["model_dir"])
    if not all_models:
        print(f"错误: 在 {test_cfg['model_dir']} 中没有找到epoch模型文件!")
        return
    
    print(f"找到 {len(all_models)} 个epoch模型")
    
    # 筛选epoch范围
    if test_cfg["start_epoch"] > 0:
        all_models = [m for m in all_models if m["epoch"] >= test_cfg["start_epoch"]]
    if test_cfg["end_epoch"] is not None:
        all_models = [m for m in all_models if m["epoch"] <= test_cfg["end_epoch"]]
    if test_cfg["num_epochs_to_test"] is not None:
        all_models = all_models[:test_cfg["num_epochs_to_test"]]
    
    if not all_models:
        print("错误: 筛选后没有可测试的模型!")
        return
    
    print(f"将测试 {len(all_models)} 个模型 (epoch {all_models[0]['epoch']} 到 {all_models[-1]['epoch']})")
    
    # 加载测试数据集
    test_dataset = TestDataset(test_cfg["test_data_dir"])
    test_loader = DataLoader(test_dataset, batch_size=test_cfg["batch_size"], 
                           shuffle=False, num_workers=2)
    
    # 初始化模型
    model = MASR_Net().to(test_cfg["device"])
    
    # 存储所有epoch的结果
    all_results = []
    
    # 批量测试所有epoch模型
    for model_info in tqdm(all_models, desc="测试进度"):
        model_path = model_info["path"]
        epoch_num = model_info["epoch"]
        train_loss = model_info["loss"]
        
        # 加载模型权重
        try:
            checkpoint = torch.load(model_path, map_location=test_cfg["device"])
            model.load_state_dict(checkpoint["model"])
        except Exception as e:
            print(f"加载模型失败 {model_path}: {e}")
            continue
        
        # 设置为评估模式
        model.eval()
        
        # 测试当前模型
        epoch_metrics = {'psnr': [], 'ssim': [], 'mse': []}
        
        with torch.no_grad():
            for batch_idx, (lr, hr, filename, affine) in enumerate(test_loader):
                lr = lr.to(test_cfg["device"])
                hr = hr.to(test_cfg["device"])
                
                # 模型推理
                pred = model(lr)
                
                # 计算指标
                metrics = calculate_metrics(pred, hr, test_cfg["background_threshold"])
                
                # 保存指标
                for key in epoch_metrics:
                    epoch_metrics[key].append(metrics[key])
        
        # 计算统计量
        stats = {}
        for metric_name, values in epoch_metrics.items():
            stats[metric_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        # 保存结果
        result = {
            "epoch": epoch_num,
            "train_loss": train_loss,
            "model_path": model_path,
            "metrics": stats,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        all_results.append(result)
        
        # 输出当前结果
        print(f"Epoch {epoch_num:03d}: PSNR={stats['psnr']['mean']:.4f}, "
              f"SSIM={stats['ssim']['mean']:.4f}, "
              f"MSE={stats['mse']['mean']:.6f}")
    
    if not all_results:
        print("错误: 没有成功测试任何模型!")
        return
    
    print(f"\n成功测试了 {len(all_results)} 个模型")
    
    # 找到PSNR最好的epoch
    best_epoch = max(all_results, key=lambda x: x["metrics"]["psnr"]["mean"])
    print(f"\n最佳PSNR epoch: {best_epoch['epoch']}")
    print(f"训练损失: {best_epoch['train_loss']:.6f}")
    print(f"测试PSNR: {best_epoch['metrics']['psnr']['mean']:.4f}")
    print(f"测试SSIM: {best_epoch['metrics']['ssim']['mean']:.4f}")
    print(f"测试MSE: {best_epoch['metrics']['mse']['mean']:.6f}")
    
    # 保存最佳模型
    best_model_dest = os.path.join(
        test_cfg["result_dir"], 
        "best_model",
        f"best_epoch_{best_epoch['epoch']:03d}_psnr_{best_epoch['metrics']['psnr']['mean']:.4f}.pth"
    )
    shutil.copy2(best_epoch["model_path"], best_model_dest)
    print(f"最佳模型已保存到: {best_model_dest}")
    
    # 保存最佳模型信息
    best_info = {
        "best_epoch": best_epoch["epoch"],
        "best_psnr": best_epoch["metrics"]["psnr"]["mean"],
        "best_ssim": best_epoch["metrics"]["ssim"]["mean"],
        "best_mse": best_epoch["metrics"]["mse"]["mean"],
        "train_loss": best_epoch["train_loss"],
        "model_path": best_model_dest,
        "original_path": best_epoch["model_path"],
        "found_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(test_cfg["result_dir"], "best_model_info.json"), "w") as f:
        json.dump(best_info, f, indent=2)
    
    # 保存所有结果到CSV
    data = []
    for result in all_results:
        metrics = result["metrics"]
        data.append({
            "epoch": result["epoch"],
            "train_loss": result["train_loss"],
            "psnr_mean": metrics["psnr"]["mean"],
            "psnr_std": metrics["psnr"]["std"],
            "ssim_mean": metrics["ssim"]["mean"],
            "ssim_std": metrics["ssim"]["std"],
            "mse_mean": metrics["mse"]["mean"],
            "mse_std": metrics["mse"]["std"],
            "model_path": result["model_path"]
        })
    
    # 创建DataFrame并保存
    df = pd.DataFrame(data)
    
    # 保存到CSV
    csv_path = os.path.join(test_cfg["result_dir"], "metrics", "all_epochs_results.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"所有epoch结果已保存到: {csv_path}")
    
    # 保存到Excel
    excel_path = os.path.join(test_cfg["result_dir"], "metrics", "all_epochs_results.xlsx")
    df.to_excel(excel_path, index=False, engine='openpyxl')
    print(f"所有epoch结果已保存到: {excel_path}")
    
    # 创建可视化总结
    summary = {
        "test_config": test_cfg,
        "total_epochs_tested": len(all_results),
        "best_epoch": best_epoch["epoch"],
        "best_psnr": best_epoch["metrics"]["psnr"]["mean"],
        "best_ssim": best_epoch["metrics"]["ssim"]["mean"],
        "best_mse": best_epoch["metrics"]["mse"]["mean"],
        "psnr_range": f"{df['psnr_mean'].min():.4f} - {df['psnr_mean'].max():.4f}",
        "ssim_range": f"{df['ssim_mean'].min():.4f} - {df['ssim_mean'].max():.4f}",
        "mse_range": f"{df['mse_mean'].min():.6f} - {df['mse_mean'].max():.6f}",
        "test_completed": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 保存汇总信息
    summary_path = os.path.join(test_cfg["result_dir"], "test_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"测试汇总已保存到: {summary_path}")
    
    # 输出最终总结
    print("\n" + "="*60)
    print("测试完成总结")
    print("="*60)
    print(f"测试了 {len(all_results)} 个epoch模型")
    print(f"PSNR范围: {summary['psnr_range']}")
    print(f"SSIM范围: {summary['ssim_range']}")
    print(f"MSE范围: {summary['mse_range']}")
    print(f"最佳epoch: {best_epoch['epoch']} (PSNR: {best_epoch['metrics']['psnr']['mean']:.4f})")
    print(f"结果保存目录: {test_cfg['result_dir']}")

# ---------------- 主程序 ----------------
if __name__ == "__main__":
    # 保存测试配置
    with open(os.path.join(test_cfg["result_dir"], "test_config.json"), "w") as f:
        json.dump(test_cfg, f, indent=2)
    
    # 执行批量测试
    start_time = time.time()
    test_all_epochs()
    end_time = time.time()
    
    print(f"\n总测试时间: {(end_time - start_time)/60:.2f} 分钟")