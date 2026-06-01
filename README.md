# DeepFake Detection on Digital Face Image Using Fine-Tuned DenseNet-121

**Md Shohan Biswas**

**Presented at BIM 2025 Conference**

[![GitHub](https://img.shields.io/badge/GitHub-5H0HAN%2FDeepFakeDensenet_BIM_2025-blue)](https://github.com/5H0HAN/DeepFakeDensenet_BIM_2025)

---

## Paper

The conference paper **"Optimizing Deepfake Detection through Exploring the Efficacy of Fine-Tuned DenseNet121 in Media Integrity"** is included in the repository:

- [`Optimizing Deepfake Detection through Exploring  the Efficacy of Fine-Tuned DenseNet121 in Media  Inte.pdf`](./Optimizing%20Deepfake%20Detection%20through%20Exploring%20%20the%20Efficacy%20of%20Fine-Tuned%20DenseNet121%20in%20Media%20%20Inte%20(1).pdf)

---

## Overview

This project develops a binary classifier to distinguish **real** from **AI-generated fake** face images using deep learning. Multiple CNN architectures are benchmarked, with transfer learning and Sharpness-Aware Minimization (SAM) optimization.

### Key Contributions

- **DenseNet-121** fine-tuning with custom classification head
- **Sharpness-Aware Minimization (SAM)** for improved generalization
- **Explainable AI (Grad-CAM)** for model interpretability
- **Ensemble learning** combining multiple architectures (95.67% accuracy, 0.9982 AUC)
- **Comprehensive data augmentation** pipeline

---

## Dataset

**Deepfake-vs-Real-v2** — 32,121 face images

| Split  | Images |
|--------|--------|
| Train  | 25,696 |
| Val    | 3,212  |
| Test   | 3,213  |
| **Total** | **32,121** |

Classes: `fake` (0), `real` (1)

---

## Architectures Evaluated

| Model | Accuracy | AUC | F1 | Precision | Recall |
|-------|----------|-----|----|-----------|--------|
| **EfficientNet-B0** | **94.93%** | 0.9962 | 0.9492 | 0.9528 | 0.9493 |
| ResNet-50 | 92.81% | 0.9902 | 0.9279 | 0.9337 | 0.9281 |
| MobileNetV3 | 92.59% | 0.9969 | 0.9256 | 0.9347 | 0.9259 |
| DenseNet-121 | 91.85% | 0.9946 | 0.9180 | 0.9280 | 0.9185 |
| **Ensemble (Final)** | **95.67%** | **0.9982** | 0.9567 | 0.9595 | 0.9567 |

Total parameters: **13,272,016**

### Model Architecture (DenseNet-121)

```
Input (3×256×256)
    ↓
DenseNet-121 Backbone (pretrained on ImageNet)
    ↓
ReLU → AdaptiveAvgPool2d(1,1) → Flatten
    ↓
Dropout(0.4) → Linear(1024 → 2)
    ↓
Output: [fake, real]
```

---

## Data Augmentation

- Resize to 256×256
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Elastic transform (p=0.2)
- Random perspective (p=0.5)
- Color jitter (brightness, contrast, saturation, hue)
- Random grayscale (p=0.1)
- Gaussian blur
- Random sharpness adjustment (p=0.2)
- Random erasing (p=0.1)
- ImageNet normalization

---

## Optimization

| Parameter | Value |
|-----------|-------|
| Base Optimizer | AdamW |
| SAM ρ | 0.05 |
| SAM Adaptive | True |
| Scheduler | ReduceLROnPlateau (factor=0.2, patience=3) |
| Min LR | 1×10⁻⁷ |
| Loss | Cross-Entropy |
| Batch Size | 32 |
| Epochs | 30 |
| Dropout | 0.4 |

---

## Explainability (Grad-CAM)

Gradient-weighted Class Activation Mapping highlights regions influencing the model's decision, providing interpretability for deepfake detection. Visualizations are generated in `Only_Densenet_Finetuning_and_xAI/`.

---

## Repository Structure

```
DeepFakeImg/
├── README.md                          # This file
├── .gitignore
├── requirements.txt
├── Optimizing Deepfake Detection ...  # Conference paper (PDF)
│
├── DeepFakeDensenet/                  # ★ Main project (BIM 2025)
│   ├── FromScratchNN_DF_V3.ipynb      #   Full training pipeline (SAM, ensemble, xAI)
│   ├── sam.py                         #   SAM optimizer implementation
│   ├── class_names.json               #   Class mapping
│   ├── dataset/                       #   Dataset splits
│   ├── checkpoints/                   #   Model checkpoints
│   ├── experiments/                   #   Training logs & metrics
│   ├── ensemble_results/              #   Final ensemble model outputs
│   ├── Only_Densenet_Finetuning_and_xAI/   # Fine-tuning + Grad-CAM
│   └── Full_Model_Densenet/           #   Complete model variant
│
├── MyLearning/                        # Learning & development
│   ├── FromScratchNN_DF_V3.ipynb      #   Main experiment notebook
│   ├── Deepfake_Classifyer_NN.ipynb   #   Initial classifier
│   ├── Deepfake_Classifyer_NN_V2.ipynb#   V2 improvements
│   ├── FromScratchNN_DF_V3.ipynb      #   Scratch-to-ensemble experiments
│   ├── experiments/                   #   Per-model experiment reports
│   │   ├── efficientnetb0_builder_20250716_173937/
│   │   ├── densenet121_builder_20250717_184816/
│   │   ├── resnet50_builder_20250717_025348/
│   │   └── mobilenetv3_builder_20250718_235331/
│   ├── ensemble_results/              #   Ensemble metrics & plots
│   ├── artifacts/                     #   Training artifacts
│   ├── For Aftab/                     #   Shared with collaborator
│   └── Deep_Fake (7).pdf, (9).pdf    #   Reference papers
│

│
└── sam/                               # SAM optimizer (upstream)
    ├── sam.py                         #   Core SAM implementation
    ├── example/                       #   Example training with CIFAR
    └── .github/                       #   CI config
```

---

## Setup

```bash
# Clone
git clone https://github.com/5H0HAN/DeepFakeDensenet_BIM_2025.git
cd DeepFakeDensenet_BIM_2025

# Conda environment
conda env create -f MyLearning/For\ Aftab/env.yml
conda activate deepfake

# Or using pip
pip install -r requirements.txt
```

### Requirements

- Python 3.10+
- PyTorch 2.x
- torchvision
- timm (PyTorch Image Models)
- matplotlib, seaborn
- scikit-learn
- numpy, pandas
- tqdm
- Pillow
- jupyter

---

## Running

Open the main notebook:

```bash
jupyter notebook DeepFakeDensenet/FromScratchNN_DF_V3.ipynb
```

Or for the fine-tuning + xAI version:

```bash
jupyter notebook DeepFakeDensenet/Only_Densenet_Finetuning_and_xAI/FromScratchNN_DF_V3.ipynb
```

---

## Results

### Confusion Matrix (Ensemble)

|          | Predicted Fake | Predicted Real |
|----------|---------------|----------------|
| Real     | 95.56%        | 4.44%          |
| Fake     | 0.47%         | 99.53%         |

### ROC Curves & Metrics

Experiment reports and visualizations (ROC curves, confusion matrices, PR curves, training history) are in each model's `experiments/` directory.

---

## Citation

```bibtex
@inproceedings{shohan2025deepfakedensenet,
  title     = {Optimizing Deepfake Detection through Exploring the Efficacy of Fine-Tuned DenseNet121 in Media Integrity},
  author    = {Shohan Biswas, Md. and others},
  booktitle = {Proceedings of BIM 2025},
  year      = {2025}
}
```

---

## Contact

**Md. Shohan Biswas** — [GitHub](https://github.com/5H0HAN)
