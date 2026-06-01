# Deepfake Image Classifier — PyTorch Notebook

A reproducible notebook pipeline for deepfake **image** classification using PyTorch.
The notebook `Deepfake_Classifyer_NN_original.ipynb` implements dataset loaders, transforms, model builders, a training loop, and evaluation with ROC/PR curves and optional TensorBoard logging.

---

## Key Features

- Dataset wrappers using `torchvision.datasets.ImageFolder` with custom `ImageDataset` abstraction.
- Augmentations built with `torchvision.transforms` (resize, flip, rotation, perspective/elastic, blur, color jitter, random sharpness) and ImageNet normalization.
- Multiple backbones with replaceable heads:
  - EfficientNet-B0 (custom head)
- ResNet-50 (custom head)
- DenseNet-121 (custom head)
- MobileNetV3-Large (custom head)
- Training loop with optimizer, loss, scheduler, and TensorBoard logging.
- Evaluation utilities: accuracy, F1, precision, recall, ROC-AUC, PR-AUC, confusion matrix.
- Plot exporters for curves and history.
- Checkpoint saving for best and last model states.

---

## Repository Layout

This project is notebook-centric. Place your dataset in a three-way split and open the notebook to run cells in order.

```
project/
├─ data/
│  ├─ train/  # class subfolders: e.g., real/, fake/
│  ├─ val/
│  └─ test/
├─ Deepfake_Classifyer_NN_original.ipynb
└─ artifacts/                 # created at runtime (optional; recommended)
   ├─ <run_id>/
   │  ├─ config.json
   │  ├─ best_model.pth
   │  ├─ last_model.pth
   │  ├─ model_architecture.txt
   │  ├─ train_history.json
   │  ├─ metrics_val.json
   │  ├─ metrics_test.json
   │  ├─ preds_val.csv
   │  ├─ preds_test.csv
   │  ├─ roc_val.png / pr_val.png / reliability_val.png
   │  ├─ confusion_val.png
   │  └─ tensorboard/       # if enabled
```

> Example dataset roots detected in the notebook:  
- Set your own `train/val/test` paths in the notebook.

---

## Setup

### Environment
- Python ≥ 3.10 recommended
- PyTorch + torchvision (CUDA optional)
- `timm`, `scikit-learn`, `matplotlib`, `tqdm`, `tensorboard` (optional)

```bash
# create and activate env (example with pip)
python -m venv .venv
. .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # choose your CUDA
pip install timm scikit-learn matplotlib tqdm tensorboard
```

### Data
Organize images into class subfolders compatible with `ImageFolder`:
```
data/
  train/
    real/
    fake/
  val/
    real/
    fake/
  test/
    real/
    fake/
```

Update the dataset paths near the top of the notebook where directories are assigned.

---

## How to Run

1. Open `Deepfake_Classifyer_NN_original.ipynb` in Jupyter or VS Code.
2. Execute cells in order:
   - Imports
   - Dataset class and validation
   - Transforms and Dataloaders
   - Choose a model builder and instantiate the backbone
   - Configure optimizer, loss, and scheduler
   - Run the training loop
   - Evaluate on validation and test
   - Plot ROC/PR curves and confusion matrix
3. Check `artifacts/<run_id>/` for saved models, metrics, and figures.

---

## Configuration

The notebook defines a `Config` container to collect paths, hyperparameters, and bookkeeping for a run. Typical knobs:

- `num_epochs`: e.g., 30
- `batch_size`: e.g., 32
- `lr`: optimizer LR, e.g., 1e-3
- `optimizer`: Adam or SGD
- `scheduler`: ReduceLROnPlateau (present)
- `criterion`: `nn.CrossEntropyLoss()`
- `device`: automatic CUDA/CPU selection

Augmentations follow ImageNet normalization with optional geometric and photometric transforms. Adjust in the “Dataset Transformation and Augmentation” cell.

---

## Training

- The training cell prints per-epoch train/val loss and metrics.
- Best model checkpoint is saved when validation improves.
- Learning rate scheduling reduces LR on plateaus.
- TensorBoard (if enabled) logs scalars and can be launched via:
  ```bash
  tensorboard --logdir artifacts
  ```

---

## Evaluation

- Compute accuracy, F1, precision, recall.
- Plot ROC and Precision–Recall curves and report AUCs.
- Plot a confusion matrix to inspect error modes.
- Load the best checkpoint and run evaluation on `val/` or `test/`:

```python
# Example (adjust paths and class names accordingly)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = Densenet121_Builder(num_classes=2, dropout_rate=0.4)  # or another backbone
ckpt   = torch.load("artifacts/<run_id>/best_model.pth", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.to(device).eval()
# Iterate a DataLoader to compute metrics and export curves
```

---

## Reproducibility

- Fix random seeds for Python, NumPy, and PyTorch.
- Set deterministic CuDNN where feasible:
  ```python
  torch.manual_seed(42); torch.cuda.manual_seed_all(42)
  import numpy as np, random
  np.random.seed(42); random.seed(42)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  ```
- Export `config.json`, package versions, and model `state_dict` with each run.

---

## Expected Artifacts

- `best_model.pth`, `last_model.pth`
- `config.json`, `versions.txt`
- Curves: `roc_*.png`, `pr_*.png`
- `confusion_*.png`, `reliability_*.png` (if calibration added)
- `metrics_*.json`, `preds_*.csv`
- `train_history.json`
- TensorBoard logs (optional)

---

## Extending the Notebook

- **Calibration**: add ECE and reliability diagrams.
- **Cross‑dataset**: add loaders for FaceForensics++, Celeb‑DF(v2), DFDC frames; compute zero‑shot ROC‑AUC and TPR@FPR.
- **Robustness**: JPEG, blur, noise sweeps; plot TPR@FPR=1% vs. distortion level.
- **Frequency branch**: add DCT/SRM stream and late fusion with spatial backbone.
- **XAI**: Grad‑CAM / LIME for wins and failures.
- **Deployment**: distill to MobileNetV3/EfficientNet‑Lite; report latency and params.

---

## Troubleshooting

- **Missing keys when loading checkpoints**: Confirm the backbone class name matches the saved model, and that the classification head shape equals `num_classes`.
- **Transforms not applied**: Ensure the same `transform` object is passed to `ImageFolder` or your `ImageDataset` wrapper.
- **Class mapping**: `ImageFolder` maps classes alphabetically; verify `dataset.classes` aligns with your expectations.
- **TensorBoard not found**: `pip install tensorboard` and restart your kernel.

---

## Citation

If you use this notebook in academic work, cite your chosen backbone paper and any datasets employed. Add your own bib entries for experiments and datasets.
