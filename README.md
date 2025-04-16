# SwinIR Super Resolution on Flickr2K Dataset  

This project demonstrates super-resolution using the **SwinIR** model trained on the **Flickr2K** dataset. It covers dataset preparation, training, evaluation, and visualization, implemented in **PyTorch** and optimized for use in **Google Colab**.

---

## Features

- Training SwinIR model from scratch on the Flickr2K dataset  
- High-resolution (HR) and low-resolution (LR x4) image pairing  
- Custom PyTorch dataset and loader  
- Swin Transformer blocks for deep feature extraction  
- Patch-wise training strategy  
- Evaluation using PSNR and SSIM  
- Visual comparison of LR, SR, and HR images  

---
Getting Started

This project is designed to run smoothly on Google Colab.

Step 1: Open in Colab
Use this link to open the notebook directly in Google Colab:
https://colab.research.google.com/github/yourusername/yourrepo/blob/main/SwinIR_Flickr2K.ipynb

Step 2: Run All Cells

Make sure the Flickr2K dataset is available in your Google Drive.
hr_dir = "/content/drive/MyDrive/Flickr2K/Flickr2K/Flickr2K_HR"
lr_dir = "/content/drive/MyDrive/Flickr2K/Flickr2K/Flickr2K_LR_bicubic/X4"

---
Training will start automatically.

Evaluation metrics (PSNR and SSIM) will be displayed.

Output samples will be shown at the end of training.

---
Model Architecture

This implementation is based on the SwinIR architecture, which uses Swin Transformers for image restoration tasks. The structure includes:

Patch embedding and linear positional encoding

Multiple Residual Swin Transformer Blocks (RSTB)

Upsampling using PixelShuffle layers for 4x scaling

Final convolutional output layer to reconstruct the high-resolution image

---
Evaluation Metrics

Two standard image quality metrics are used to evaluate the model:

PSNR (Peak Signal-to-Noise Ratio): Measures image fidelity

SSIM (Structural Similarity Index Measure): Measures perceptual similarity

Example Results (on 10 validation images from the Flickr2K dataset):
PSNR: approximately 27.3 dB
SSIM: approximately 0.79

---

Sample Output

Low-Resolution Input -> Super-Resolved Output -> Ground Truth High-Resolution

You can view sample visual outputs in the notebook once training is completed.

---
![image](https://github.com/user-attachments/assets/997d5c7b-2567-4eaf-84de-d3613b47e58c)

---
Dataset Structure

Make sure your Flickr2K dataset is organized as follows:

Flickr2K/
├── Flickr2K_HR/ (High-resolution ground truth images)
├── Flickr2K_LR_bicubic/
└── X4/ (Low-resolution images, bicubic downsampled by factor of 4)

---
Saving and Loading

The model is automatically saved after training completes.

The saved model can later be reloaded for inference or further fine-tuning.

---
Author
Created by Gopika R Feel free to reach out for questions or collaboration ideas!











