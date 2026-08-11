# Image Segmentation using Segment Anything Model 2 (SAM2)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Meta SAM2](https://img.shields.io/badge/Meta-SAM2-0081FB?style=for-the-badge)](https://github.com/facebookresearch/segment-anything-2)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

Zero-shot automated image segmentation pipeline integrating custom Roboflow dataset annotations with **Meta's Segment Anything Model 2 (SAM2)** foundation vision model.

---

## 🌟 Features
- 🎯 **Prompt-Guided Masking:** Supports point prompts `[[x, y]]` and bounding box prompts for high-precision boundary extraction.
- 📐 **Evaluation Metrics:** Automatic computation of Intersection over Union (**IoU**) and Dice similarity coefficient.
- 🖼️ **Roboflow Dataset Support:** Seamlessly integrates with custom Roboflow computer vision annotation datasets.

---

## 📁 Repository Structure
```text
SAM2-Image-Segmentation/
├── sam2_pipeline/           # Core Segmentation Module
│   ├── __init__.py
│   ├── model.py             # SAM2 Engine Wrapper
│   └── eval.py              # IoU & Dice Metrics
├── demo.py                  # Executable Demo Script
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚡ Quick Start
```bash
git clone https://github.com/shivangisrivastava013/SAM2-Image-Segmentation.git
cd SAM2-Image-Segmentation

pip install -r requirements.txt
python demo.py
```

---

## 👤 Author
**Shivangi Srivastava**  
MS in Artificial Intelligence @ NJIT  
[LinkedIn Profile](https://www.linkedin.com/in/shivangisrivastava013/) | [Portfolio](https://shivangisrivastava013.github.io/shivangi-portfolio/)
