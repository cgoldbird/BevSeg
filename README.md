# PolarNet Implementation and Integration in MMDetection3D

This repository contains the implementation and integration of the **PolarNet** model within the **MMDetection3D** framework. PolarNet is an efficient model for large-scale 3D point cloud segmentation, particularly suited for autonomous driving scenarios.

---

## Features
- Seamless integration of PolarNet with MMDetection3D.
- Support for data preprocessing, model training, evaluation, and inference.
- Compatible with large-scale outdoor point cloud datasets SemanticKITTI.

---

## Requirements
- Install MMDetection3D and its dependencies following the [official documentation](https://github.com/open-mmlab/mmdetection3d).
- Prepare SemanticKITTI DataSets follow MMDetection3D.
---

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo/PolarNet-MMDetection3D.git
   cd PolarNet-MMDetection3D
2. train
   ```bash
   python train.py configs/bevseg_semantickitti_polar_stlr.py
