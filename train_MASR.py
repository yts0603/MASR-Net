#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, glob, json, time
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm  

from model_MASR import downsample2x,MASR_Net

# ---------------- 配置 ----------------
result_base = "/data/yangtianshu/reconstruction/train_result/MASR/1.28_epo50/epo70/"
model_name = "MASR_Net"
cfg = {
    "hr_dir":     "/data/yangtianshu/reconstruction/infant-data/train-ex",
    "ckpt_dir":   os.path.join(result_base, f"{model_name}_model70"),     
    "log_dir":    os.path.join(result_base, f"{model_name}_logs70"),      
    "result_dir": result_base,                             
    "batch_size": 1,
    "lr":         1e-3,
    "epochs":     70,
    "device":     "cuda:2" if torch.cuda.is_available() else "cpu"
}

# ---------------- 创建所有必要的目录 ----------------

os.makedirs(cfg["log_dir"], exist_ok=True)
os.makedirs(cfg["ckpt_dir"], exist_ok=True)
os.makedirs(cfg["result_dir"], exist_ok=True)

#创建每轮模型保存目录
epoch_ckpt_dir = os.path.join(cfg["ckpt_dir"], "epoch_checkpoints")
os.makedirs(epoch_ckpt_dir, exist_ok=True)
# 创建训练信息目录
info_dir = os.path.join(cfg["result_dir"], "training_info")
os.makedirs(info_dir, exist_ok=True)

writer = SummaryWriter(cfg["log_dir"])
json.dump(cfg, open(f"{info_dir}/config.json", "w"), indent=2)

# ---------------- 训练日志文件 ----------------
training_log_file = os.path.join(info_dir, "training_log.txt")
with open(training_log_file, 'w') as f:
    f.write("Training Started at: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    f.write("="*50 + "\n")
    f.write(f"Configuration: {json.dumps(cfg, indent=2)}\n")
    f.write("="*50 + "\n\n")



# ---------------- 数据集：同时保存128×128×64输入 ----------------
class InfantDataset(Dataset):
    def __init__(self, hr_dir):
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.nii*")))
    
        self.lr_save_dir = "/data/yangtianshu/reconstruction/infant-data/train-input"
        os.makedirs(self.lr_save_dir, exist_ok=True)

    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        hr_np = nib.load(hr_path).get_fdata(dtype=np.float32)
        hr_np = (hr_np - hr_np.min()) / (hr_np.max() - hr_np.min() + 1e-8)

        hr = torch.from_numpy(hr_np).unsqueeze(0).float()  # (1,256,256,128)
        lr = downsample2x(hr.unsqueeze(0)).squeeze(0)      # (1,128,128,64)

        
        lr_np = lr.squeeze(0).cpu().numpy()                # (128,128,64)
        base_name = os.path.basename(hr_path).replace(".nii.gz", "_input.nii.gz")
        lr_save_path = os.path.join(self.lr_save_dir, base_name)

        if not os.path.exists(lr_save_path):               # 避免重复写
            lr_nii = nib.Nifti1Image(lr_np, affine=nib.load(hr_path).affine)
            nib.save(lr_nii, lr_save_path)

        return lr, hr

    def __len__(self):
        return len(self.hr_files)

# ---------------- 损失函数 ----------------
class CompositeLoss(nn.Module):
    def __init__(self, l1=0.7, ssim=0.2, grad=0.1):
        super().__init__()
        self.l1_w, self.ssim_w, self.grad_w = l1, ssim, grad
        self.l1 = nn.L1Loss()
    
    def gradient_loss(self, pred, tgt):
        # 保持原样
        grad_pred = torch.gradient(pred, dim=[2, 3, 4])
        grad_tgt = torch.gradient(tgt, dim=[2, 3, 4])
        
        grad_loss = 0
        for i in range(3):
            grad_loss += F.l1_loss(grad_pred[i], grad_tgt[i])
        
        return grad_loss / 3
    
    def ssim_loss(self, pred, tgt):
        """简化的SSIM，更稳定"""
        # 确保数值范围合理
        pred = torch.clamp(pred, 0, 1)
        tgt = torch.clamp(tgt, 0, 1)
    
        win = 7 
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        mu_p = F.avg_pool3d(pred, win, 1, win//2)
        mu_t = F.avg_pool3d(tgt, win, 1, win//2)
        
        mu_p_sq = mu_p * mu_p
        mu_t_sq = mu_t * mu_t
        mu_p_mu_t = mu_p * mu_t
        
        sigma_p_sq = F.avg_pool3d(pred*pred, win, 1, win//2) - mu_p_sq
        sigma_t_sq = F.avg_pool3d(tgt*tgt, win, 1, win//2) - mu_t_sq
        sigma_p_t = F.avg_pool3d(pred*tgt, win, 1, win//2) - mu_p_mu_t
        
        # 防止负值
        sigma_p_sq = torch.clamp(sigma_p_sq, min=0)
        sigma_t_sq = torch.clamp(sigma_t_sq, min=0)
        
        numerator = (2*mu_p_mu_t + C1) * (2*sigma_p_t + C2)
        denominator = (mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2) + 1e-8
        
        ssim_value = numerator / denominator
        
        return 1 - ssim_value.mean()
    
    def forward(self, pred, tgt):
        l1 = self.l1(pred, tgt)
        
        # 计算所有损失
        ssim = self.ssim_loss(pred, tgt) if self.ssim_w > 0 else 0
        grad = self.gradient_loss(pred, tgt) if self.grad_w > 0 else 0
        
        # 总损失
        total_loss = self.l1_w * l1
        if self.ssim_w > 0:
            total_loss += self.ssim_w * ssim
        if self.grad_w > 0:
            total_loss += self.grad_w * grad
        
        # 返回详细的损失字典
        loss_dict = {
            'l1': l1.item(),
            'total': total_loss.item()
        }
        
        if self.ssim_w > 0:
            loss_dict['ssim'] = ssim.item() if isinstance(ssim, torch.Tensor) else ssim
        
        if self.grad_w > 0:
            loss_dict['grad'] = grad.item() if isinstance(grad, torch.Tensor) else grad
        
        return total_loss, loss_dict

# ---------------- 训练总结函数 ----------------
def save_training_summary(best_loss, best_epoch, training_time, model, dataset, sched, epoch_losses):
    total_params = sum(p.numel() for p in model.parameters())
    
    summary = {
        "training_completed": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_loss": best_loss,
        "best_epoch": best_epoch,
        "total_parameters": total_params,
        "training_time_seconds": training_time,
        "training_time_minutes": training_time / 60,
        "final_learning_rate": sched.get_last_lr()[0],
        "model_architecture": "MambaAtt_UNet",
        "dataset_size": len(dataset),
        "parameters_millions": total_params / 1e6,
        "epoch_losses": epoch_losses  # 保存所有epoch的损失
    }
    
    # 保存JSON格式
    summary_file = os.path.join(info_dir, "training_summary30.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 保存文本格式
    txt_summary = os.path.join(info_dir, "training_summary30.txt")
    with open(txt_summary, 'w') as f:
        f.write("TRAINING SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Training completed at: {summary['training_completed']}\n")
        f.write(f"Best loss: {best_loss:.6f} (achieved at epoch {best_epoch})\n")
        f.write(f"Training time: {training_time/60:.2f} minutes\n")
        f.write(f"Final learning rate: {sched.get_last_lr()[0]:.2e}\n")
        f.write(f"Dataset size: {len(dataset)} samples\n")
        f.write(f"Model: UNet3D with Self-Attention at bottleneck\n")
        f.write(f"\nEpoch Losses:\n") 
    
    # 保存模型结构
    model_arch_file = os.path.join(info_dir, "model_architecture.txt")
    with open(model_arch_file, 'w') as f:
        f.write("MODEL ARCHITECTURE\n")
        f.write("="*50 + "\n")
        f.write(str(model))
        f.write("\n\nLAYER DETAILS:\n")
        f.write("="*50 + "\n")
        for name, param in model.named_parameters():
            f.write(f"{name}: {param.shape}\n")

    # ========== 新增：保存损失曲线数据 ==========
    loss_curve_file = os.path.join(info_dir, "loss_curve.json")
    with open(loss_curve_file, 'w') as f:
        json.dump({
            "epochs": list(range(len(epoch_losses))),
            "losses": epoch_losses,
            "best_epoch": best_epoch,
            "best_loss": best_loss
        }, f, indent=2)
        
# ---------------- 训练函数 ----------------
def train():
    dataset = InfantDataset(cfg["hr_dir"])
    loader  = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                         num_workers=4, pin_memory=True)
    model   = MASR_Net().to(cfg["device"])
    
    # 保存模型结构
    model_arch_file = os.path.join(info_dir, "model_architecture.txt")
    with open(model_arch_file, 'w') as f:
        f.write("MODEL ARCHITECTURE\n")
        f.write("="*50 + "\n")
        f.write(str(model))
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"参数量: {total_params/1e6:.2f}M")
    
    # 记录到日志文件
    with open(training_log_file, 'a') as f:
        f.write(f"Model parameters: {total_params/1e6:.2f}M\n")
        f.write(f"Training started with {len(dataset)} samples\n\n")
    
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    criterion = CompositeLoss(l1=0.7, ssim=0.2, grad=0.1)

    # ========== 初始化最佳记录和损失记录 ==========
    best = float('inf')
    best_epoch = 0
    epoch_losses = []  # 记录每个epoch的损失
    
    
    for epoch in range(cfg["epochs"]):
        model.train()
        running_loss = 0.
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d}")
        
        for lr, hr in pbar:
            lr, hr = lr.to(cfg["device"]), hr.to(cfg["device"])
            opt.zero_grad()
            pred = model(lr)
            loss, loss_dict = criterion(pred, hr)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running_loss += loss_dict['total'] * lr.size(0)
            pbar.set_postfix({"Loss": f"{loss_dict['total']:.6f}"})

        epoch_loss = running_loss / len(dataset)
        sched.step()

        epoch_losses.append(float(epoch_loss))
        
        # TensorBoard记录
        writer.add_scalar("Loss/Total", epoch_loss, epoch)
        writer.add_scalar("LR", sched.get_last_lr()[0], epoch)
        
        print(f"Epoch {epoch:03d}  Total={epoch_loss:.6f}  LR={sched.get_last_lr()[0]:.2e}")

        # ========== 保存每一轮的模型 ==========
        
        epoch_ckpt_path = os.path.join(epoch_ckpt_dir, f"epoch_{epoch:03d}_loss_{epoch_loss:.6f}.pth")
        os.makedirs(os.path.dirname(epoch_ckpt_path), exist_ok=True)
        torch.save({
            "epoch": epoch, 
            "model": model.state_dict(), 
            "opt": opt.state_dict(),
            "loss": epoch_loss,
            "lr": sched.get_last_lr()[0]
        }, epoch_ckpt_path)
        
        # 记录到日志文件
        os.makedirs(os.path.dirname(training_log_file), exist_ok=True)
        with open(training_log_file, 'a') as f:
            f.write(f"Epoch {epoch:03d} - Loss: {epoch_loss:.6f} - Model saved: {os.path.basename(epoch_ckpt_path)}\n")
       

        # ========== 保存最佳模型并标注 ==========
        if epoch_loss < best:
            best = epoch_loss
            best_epoch = epoch
            
            best_ckpt_path = os.path.join(cfg["ckpt_dir"], "best.pth")
            os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
            
            torch.save({
                "epoch": epoch, 
                "model": model.state_dict(), 
                "opt": opt.state_dict(),
                "loss": best,
                "lr": sched.get_last_lr()[0],
                "is_best": True,
                "best_epoch": best_epoch
            }, best_ckpt_path)

    
    # 返回最佳epoch和损失记录 
    return model, best, best_epoch, dataset, sched, epoch_losses

# ---------------- 主程序 ----------------
if __name__ == "__main__":
    start_time = time.time()
    
    # 记录训练开始
    with open(training_log_file, 'a') as f:
        f.write(f"\nTraining process started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 训练模型
    model, best_loss, best_epoch, dataset, sched, epoch_losses = train()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # 保存训练总结
    save_training_summary(best_loss, best_epoch, training_time, model, dataset, sched, epoch_losses)
    
    # 记录训练完成
    with open(training_log_file, 'a') as f:
        f.write(f"\nTraining completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total training time: {training_time/60:.2f} minutes\n")
        f.write(f"Best loss achieved: {best_loss:.6f} at epoch {best_epoch}\n")
        f.write(f"Best loss achieved: {best_loss:.6f}\n")
        f.write("="*50 + "\n")
    
    print(f"\n训练完成! 总耗时: {training_time/60:.2f} 分钟")
    print(f"最佳损失: {best_loss:.6f} (在第 {best_epoch} 轮达到)")
    print(f"模型保存在: {cfg['ckpt_dir']}")
    print(f"TensorBoard日志保存在: {cfg['log_dir']}")
    print(f"训练信息保存在: {info_dir}")
    print(f"使用以下命令查看训练曲线: tensorboard --logdir={cfg['log_dir']}")
    
    writer.close()