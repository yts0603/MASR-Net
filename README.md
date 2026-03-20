# MASR-Net: Synergistic Mamba and Self-Attention for High-Fidelity Infant Brain MRI Super-Resolution

Official repository for our paper submitted to The Visual Computer.
## Overview
![Model Architecture](Figures/Figure_2.tif)
*Figure 1: Overall architecture of MASR-Net.*

## Environment Setup
The project is developed under the following environment:
### Requirements
- Python 3.9
- PyTorch 2.4.1
- CUDA 12.1 (for GPU acceleration)

For installation of the project dependencies, please run:
```bash
pip install -r Requirements.txt
```
## Preprocessing
Preprocessing scripts are located in the pre/ folder. Run them in order to prepare your dataset.

## Training
Before training, edit the cfg dictionary in train_MASR.py to set your data paths and hyperparameters. Then run:
```bash
python train_MASR.py
```
## Test
Edit the test_cfg dictionary in test_all.py to specify the test data and model checkpoint. Then run:
```bash
python test_all.py
```
## Data Availability
The neonatal brain MRI dataset used in this study is not publicly available due to ethical and privacy regulations.
For reproducibility, you may use public alternatives and follow the preprocessing pipeline above.

## Results
Our MASR-Net achieves state-of-the-art performance on our neonatal brain MRI dataset (0-2 years old):
Peak Signal-to-Noise Ratio (PSNR): 34.136 dB
Structural Similarity Index (SSIM): 0.978
Mean Squared Error (MSE): 0.4×10?3
