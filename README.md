# 3DLAND
Organ-aware 3D lesion segmentation dataset and pipeline for abdominal CT analysis (ACM Multimedia 2025)

# 3DLAND: 3D Lesion Abdominal Anomaly Localization Dataset

This repository contains the code and dataset instructions for **3DLAND**, the first large-scale, organ-aware 3D lesion segmentation benchmark for abdominal CT scans, introduced at *ACM Multimedia 2025*.

## 🌐 Overview

- 6,000+ contrast-enhanced CT studies
- 3D lesion masks aligned with 7 abdominal organs
- Prompt-based annotation and propagation pipeline
- Applications: anomaly detection, lesion retrieval, organ-aware analysis

## 🧠 Pipeline

The lesion segmentation pipeline includes:
1. Organ segmentation via MONAI
2. Lesion-to-organ assignment
3. 2D mask generation using SAM prompts
4. 3D mask propagation using MedSAM2

## 📦 Dataset

The dataset (metadata + mask annotations) is hosted at:  
👉 [Download via Zenodo](https://zenodo.org/...) *(or your link)*
This project uses the DeepLesion dataset, publicly available from the NIH Clinical Center.
Please note that this dataset is not released under Apache 2.0. Users must ensure compliance with NIH terms of use: https://nihcc.app.box.com/v/DeepLesion

## 🚀 Getting Started

```bash
git clone https://github.com/yourusername/3DLAND.git
cd 3DLAND
pip install -r requirements.txt



📜 License

Code in this repository is licensed under Apache License 2.0.

