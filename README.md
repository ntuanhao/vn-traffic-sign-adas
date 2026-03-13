# 🇻🇳 Vietnamese Traffic Sign Recognition System based on Lane Detection

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-GUI-green)](#)

> **Graduation Thesis 2026** - Ho Chi Minh City University of Natural Resources and Environment (HCMUNRE)
> 
> **Author:** Nguyen Tuan Hao

## 📌 Overview
This project introduces a comprehensive Deep Learning pipeline to solve a specific and practical traffic problem in Vietnam: **Identifying traffic signs that apply ONLY to the ego-lane (the lane the vehicle is currently driving in)**. 

Traditional traffic sign recognition systems often detect all signs in the frame, leading to false alerts from signs meant for other lanes. This system effectively combines Semantic Segmentation, Object Detection, Heuristic Spatial Logic, and Image Classification to filter out irrelevant signs and accurately classify the valid ones.

## 🚀 Key Features
* **End-to-End Pipeline:** Integrates 3 deep learning models running sequentially to ensure high precision.
* **Spatial Logic Filtering:** Calculates the intersection and relative position between detected signs and the segmented ego-lane boundaries to reject signs from opposite or parallel lanes.
* **Small Object Detection:** Utilizes **SAHI** (Slicing Aided Hyper Inference) combined with YOLO to detect tiny traffic signs from a distance.
* **Desktop GUI:** A highly interactive user interface built with **PyQt5**, supporting:
    * Image and Video stream processing.
    * Real-time **"Demo Mode"** showing step-by-step pipeline execution (Segmentation -> Detection -> Logic -> Classification).

## 🧠 System Architecture

The system operates through a strict 4-step pipeline:

1. **Lane Segmentation (DeepLabV3+ / U-Net++):** Extracts the drivable area, separating the ego-lane, opposite lanes, and sidewalks.
2. **Traffic Sign Detection (YOLOv8 + SAHI):** Detects all potential traffic signs in the frame (raw bounding boxes). SAHI is applied to handle very small objects that standard YOLO might miss.
3. **Spatial Logic Filter (Algorithm):** Projects the bounding boxes onto the segmentation mask. Rejects boxes that fall outside the mathematical boundaries of the ego-lane (e.g., flags them as "SAI LÀN TRÁI", "SAI LÀN PHẢI").
4. **Classification (EfficientNetV2-S):** Crops the valid bounding boxes and classifies them into 14 specific Vietnamese traffic sign categories.

---

## 🛠️ Technologies & Libraries Used
* **Deep Learning Framework:** PyTorch
* **Segmentation:** Segmentation Models PyTorch (SMP), DeepLabV3 (ResNet50/MobileNetV3 backbone).
* **Detection:** Ultralytics YOLO, SAHI (Slicing Aided Hyper Inference).
* **Classification:** Torchvision EfficientNetV2-S.
* **Computer Vision & Augmentation:** OpenCV, Albumentations.
* **Application GUI:** PyQt5.

## ⚙️ Installation & Usage

### Prerequisites
* Python 3.8+
* CUDA Toolkit (Highly recommended for GPU acceleration)

### Setup
1. Clone the repository:

  git clone https://github.com/ntuanhao/vn-traffic-sign-adas.git
 
  cd vn-traffic-sign-adas

2. Install dependencies:

    pip install -r requirements.txt
   
   Ensure the structure looks like this:

       models/
       ├── gp4_deeplab_best.pth               # Lane Segmentation
       ├── best_sahi_v10.pt                   # Sign Detection
       └── vietnam_traffic_sign_robust_model.pth # Sign Classification

### Run the Application
Start the PyQt5 Dashboard:

    python app_demo.py

## 📊 Model Training Highlights
* **Custom Dataset:** Trained on custom datasets specifically tailored for Vietnamese traffic environments and road conditions.
* **Loss Optimization:** Combined `Dice Loss` and `Focal Loss` for segmentation to handle class imbalance and improve boundary sharpness.
* **Data Enhancement:** Applied `CLAHE` and sharpening filters during preprocessing to improve performance in low-light and hazy conditions.
* **Overfitting Prevention:** Utilized `Early Stopping`, `CosineAnnealingLR` / `ReduceLROnPlateau`, and aggressive Albumentations pipelines.

---

## 👨‍💻 Author
**Nguyen Tuan Hao**
* **Role:** Software Engineer / AI Researcher
* **Email:** tuanhao050403@gmail.com
* **LinkedIn:** www.linkedin.com/in/tuấn-hào-a34b9b218.
* **University:** Ho Chi Minh City University of Natural Resources and Environment (HCMUNRE)

---
*If you find this project interesting or helpful, please consider giving it a ⭐!*
