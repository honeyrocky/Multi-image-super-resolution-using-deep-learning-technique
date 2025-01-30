# Multi-image-super-resolution-using-deep-learning-technique

## 📌 Overview
This project implements **Multi-Image Super-Resolution (MISR)** using deep learning techniques. MISR leverages multiple low-resolution images of the same scene to reconstruct a higher-quality image, improving details and reducing noise. The model utilizes convolutional neural networks (CNNs) and advanced image processing techniques to enhance resolution effectively.

## 🚀 Features
- Uses multiple low-resolution images to reconstruct a high-resolution output.
- Deep learning-based approach for enhanced image quality.
- Handles noise and distortions efficiently.
- Suitable for satellite imagery, medical imaging, and computer vision applications.

## 🏗️ Model Architecture
The project implements a **CNN-based super-resolution model** with the following components:
- **Feature Extraction:** Extracts useful details from multiple images.
- **Alignment Module:** Aligns images to handle motion and differences.
- **Fusion Network:** Merges multiple image features for enhanced resolution.
- **Reconstruction Network:** Generates the final high-resolution output.

## 📂 Dataset
The model is trained on publicly available datasets, such as:
- **DIV2K** (High-quality image dataset for super-resolution)
- **Flickr1024** (Stereo image dataset)
- Custom datasets (if applicable)

## 🛠️ Installation
To set up and run the project, follow these steps:

### 1️⃣ Clone the repository
```sh
git clone https://github.com/your-username/Multi-Image-Super-Resolution.git
cd Multi-Image-Super-Resolution


Follow steps to run the project: 

pip install -r requirements.txt
python train.py  # To train the model
python test.py   # To test on sample images

Results
The model achieves:

Higher PSNR (Peak Signal-to-Noise Ratio)
Better perceptual quality compared to traditional methods
Robust performance on real-world images


