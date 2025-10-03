#  Real-ESRGAN Super Resolution on DIV2K Dataset

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<your-username>/<your-repo-name>/blob/main/real_esrgan_div2k.ipynb)

This project demonstrates a simplified version of the **Real-ESRGAN** model for **image super-resolution** using the **DIV2K dataset**. It includes model training, evaluation, and visualization — all done using PyTorch and runnable in Google Colab.

---

##  Features

-  Automatic downloading and extraction of the DIV2K dataset
-  Custom PyTorch `Dataset` and `Dataloader`
-  A simplified **Real-ESRGAN** model using RRDB blocks
-  Training loop with L1 Loss and Adam optimizer
-  Evaluation metrics: **PSNR** and **SSIM**
-  Visual comparison of Low-Res, Super-Resolved, and High-Res images

---

##  Getting Started

This project is designed to run on **Google Colab**.

### Step 1: Open in Colab
Click the badge above or use this link to run it:  
📎 https://colab.research.google.com/github/srushtidayanand/Real_Esrgan_Div2K/blob/main/real_esrgan_div2k.ipynb

### Step 2: Run All Cells
- The dataset will be downloaded automatically.
- Training will begin immediately.
- After training, the model is saved and evaluated.
- Sample outputs are visualized at the end.

---

##  Model Architecture

This version of Real-ESRGAN includes:
- Residual Dense Blocks (RDBs)
- Stacked RRDB blocks
- PixelShuffle upsampling for x4 scaling
- Final convolutional output layer

---

##  Evaluation Metrics

We use:
- **PSNR (Peak Signal-to-Noise Ratio)** to measure image fidelity.
- **SSIM (Structural Similarity Index)** to measure perceptual quality.

- PSNR: 22.75 dB
SSIM: 0.6859

Both metrics are calculated over 10 validation images from the DIV2K dataset.

---


![Screenshot 2025-04-14 011213](https://github.com/user-attachments/assets/e77e65a0-9226-4227-8217-2f9d50ac72e0)
