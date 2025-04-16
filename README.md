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

## Getting Started

This project is designed to run in **Google Colab**.

### Step 1: Open in Colab  
Click the badge at the top, or use this link:  
`https://colab.research.google.com/github/yourusername/yourrepo/blob/main/SwinIR_Flickr2K.ipynb`

### Step 2: Run All Cells  

- The dataset should already be available in your Google Drive.
- Modify paths if needed:

```python
hr_dir = "/content/drive/MyDrive/Flickr2K/Flickr2K/Flickr2K_HR"
lr_dir = "/content/drive/MyDrive/Flickr2K/Flickr2K/Flickr2K_LR_bicubic/X4"
Training will begin and logs will show PSNR/SSIM.
Sample outputs are visualized after training.

Model Architecture
This version uses the SwinIR architecture based on Swin Transformers, including:

Patch embedding and linear positional encoding

Residual Swin Transformer blocks (RSTB)

PixelShuffle-based upsampling for ×4 scale

Final convolutional layer for image output

Evaluation Metrics
We use the following metrics to evaluate model performance:

PSNR (Peak Signal-to-Noise Ratio): Measures image reconstruction quality

SSIM (Structural Similarity Index): Evaluates perceptual similarity to ground truth

Example Results on a 10-image validation subset of Flickr2K:

PSNR: ~27.3 dB

SSIM: ~0.79

Sample Output
![image](https://github.com/user-attachments/assets/e4c9e72c-85c4-4a3c-85a7-9bbab18f9f89)

Dataset Structure
The project uses the Flickr2K dataset with the following structure:
Flickr2K/
├── Flickr2K_HR/                      # High-res ground truth images
├── Flickr2K_LR_bicubic/
│   └── X4/                           # Low-res images (bicubic downsampled, x4)

Saving and Loading
The model is saved automatically after training.

You can load the .pth file for inference or fine-tuning later.

Author
Created by Gopika R. Feel free to reach out for questions or collaboration ideas!





