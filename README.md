# Machine_Learning-SWIN-IR
SwinIR Super-Resolution on Flickr2K
This repository contains code for training a simplified SwinIR (Swin Transformer for Image Restoration) model for single-image super-resolution using the Flickr2K dataset.

📁 Project Structure
Dataset: Flickr2K (HR images only)

Model: Simplified SwinIR architecture using PyTorch

Training: 1 epoch using L1 Loss and Adam optimizer

Evaluation: PSNR and SSIM metrics

📦 Dependencies
Make sure you have the following libraries installed:

bash
Copy
Edit
pip install torch torchvision opencv-python matplotlib tqdm scikit-image
📂 Dataset
Download and extract the Flickr2K dataset into the following path:

swift
Copy
Edit
/content/Flickr2K/Flickr2K/Flickr2K_HR
📜 How to Use
1. Dataset Loader
The dataset class loads high-resolution (HR) images, extracts patches, and generates corresponding low-resolution (LR) images using bicubic downsampling.

2. Model Architecture
The SwinIR model is implemented with:

Convolutional layers instead of Swin Transformer blocks for simplicity

PixelShuffle for upsampling

Residual learning with mean normalization

3. Training
Set parameters like batch_size, hr_size, scale, and epochs in the script. To train:

python
Copy
Edit
python train.py
The model will be saved to:

bash
Copy
Edit
/content/swinir_flickr2k.pth
4. Evaluation
The script includes:

PSNR and SSIM metrics using skimage

Visualization of LR input, super-resolved output, and HR ground truth using matplotlib

5. Key Functions
evaluate_metrics(model, dataset): Computes average PSNR and SSIM over a subset of samples.

show_sr_result(model, dataset, idx): Displays sample LR, SR, and HR images.

⚙ Notes
Only .png images are considered from the HR dataset.

HR patches are randomly cropped with a default size of 128x128.

Ensure correct method names (_init, __len, __getitem_) in the dataset class for proper functionality.

🔧 Fixes To Apply
Make sure you correct the following typos in your original code:

python
Copy
Edit
def _init(self, ...):  # Not _init
def _len(self):        # Not _len
def _getitem(self, ...):  # Not _getitem
