# Melanoma Classification via Ensemble Transfer Learning with Tree-structured Parzen Estimator

Undergraduate Thesis — Informatics, Universitas Muhammadiyah Malang

> *Peningkatan Performa Klasifikasi Melanoma Berbasis Ensemble Transfer Learning Menggunakan Bayesian Optimization dengan Tree-structured Parzen Estimator*

---

## Overview

This repository contains the full experimental code for automated binary classification of melanoma skin lesions (Benign vs. Malignant). Four pre-trained architectures — DenseNet-121, InceptionV3, Xception, and a Vision Transformer (ViT-Small/16) — are each fine-tuned independently, with their hyperparameters searched automatically via **Bayesian Optimisation using Tree-structured Parzen Estimator (TPE)** through [Optuna](https://optuna.org/). Their probability outputs are then fused through a weighted ensemble whose weights are found by a secondary TPE search.

The work replicates and extends the ensemble framework of **Sarıateş & Özbay (2025)** (*Diagnostics*, 15, 1928). Key differences from the reference paper:

- Hyperparameter search is fully automated (10 HPs per model, 15–30 trials each) rather than fixed manual configurations.
- Optuna's multivariate TPE sampler models correlations between hyperparameters, with per-model fANOVA importance analysis.
- A progressive layer-unfreezing schedule is applied during final training (4 phases across 35 epochs), with per-phase learning rate scaling to reduce catastrophic forgetting.
- Ensemble weights are optimised on the validation set via a dual objective: `0.6 × Accuracy + 0.4 × AUC` over 100 trials.
- Generalisation is evaluated on a fully independent external test set (DS2, Mendeley) never seen during training.

---

## Results

### Scenario I — Individual models (DS1 test set, TTA ×10)

| Model | Accuracy | Precision | Recall | F1-Score | AUC | Paper Acc | Δ |
|---|---|---|---|---|---|---|---|
| DenseNet-121 | 95.20% | 95.02% | 95.40% | 95.21% | 0.9874 | 94.50% | +0.70% |
| InceptionV3 | 95.35% | 96.51% | 94.10% | 95.29% | 0.9894 | 91.20% | +4.15% |
| Xception | 93.80% | 94.97% | 92.50% | 93.72% | 0.9850 | 93.80% | +0.00% |
| ViT-Pretrained | 95.10% | 96.98% | 93.10% | 95.00% | 0.9898 | 88.25% | +6.85% |

### Scenario II — Weighted ensemble (DS1 test set)

| | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| Ensemble (this work) | **96.20%** | **96.86%** | **95.50%** | **96.17%** | **0.9942** |
| Sarıateş & Özbay (2025) | 95.25% | 94.20% | 96.22% | 95.20% | — |
| Δ | +0.95% | +2.66% | −0.72% | +0.97% | — |

Optimal ensemble weights found by TPE (100 trials):

| Model | Weight | Share |
|---|---|---|
| DenseNet-121 | 0.2469 | 24.7% |
| InceptionV3 | 0.1794 | 17.9% |
| Xception | 0.1213 | 12.1% |
| ViT-Pretrained | 0.4523 | 45.2% |

Best combined score (0.6 × Acc + 0.4 × AUC) on the validation set: **0.9565**

ViT receives the largest weight (≈ 45%), consistent with its highest individual AUC among the four models.

### Scenario III — Cross-dataset generalisation (DS2 / Mendeley, 1,000 images)

| Model | DS1 Acc | DS2 Acc | Δ | DS1 Recall | DS2 Recall |
|---|---|---|---|---|---|
| DenseNet-121 | 95.20% | 97.00% | +1.80% | 95.40% | 97.60% |
| InceptionV3 | 95.35% | 97.60% | +2.25% | 94.10% | 97.00% |
| Xception | 93.80% | 95.30% | +1.50% | 92.50% | 94.20% |
| ViT-Pretrained | 95.10% | 97.10% | +2.00% | 93.10% | 96.60% |
| Ensemble | 96.20% | 97.40% | +1.20% | 95.50% | 97.00% |

No model degraded when evaluated on DS2. All four architectures, trained exclusively on DS1, maintained or improved across all metrics on the independent Mendeley dataset.

### Confusion Matrix Summary (DS1 / DS2)

| Model | DS1 TP | DS1 FP | DS1 FN | DS1 TN | DS2 TP | DS2 FP | DS2 FN | DS2 TN |
|---|---|---|---|---|---|---|---|---|
| DenseNet-121 | 954 | 50 | 46 | 950 | 488 | 18 | 12 | 482 |
| InceptionV3 | 941 | 34 | 59 | 966 | 485 | 9 | 15 | 491 |
| Xception | 925 | 49 | 75 | 951 | 471 | 18 | 29 | 482 |
| ViT-Pretrained | 931 | 29 | 69 | 971 | 483 | 12 | 17 | 488 |

---

## Optimal Hyperparameters

### CNN Models

| Hyperparameter | DenseNet-121 | InceptionV3 | Xception |
|---|---|---|---|
| Learning rate | 9.03 × 10⁻³ | 3.33 × 10⁻³ | 4.33 × 10⁻⁴ |
| Optimizer | Adamax | AdamW | Adam |
| Unfreeze init | last_2_blocks | last_2_blocks | last_block |
| Batch size | 64 | 64 | 32 |
| Dropout | 0.258 | 0.220 | 0.430 |
| Focal loss | No | No | No |
| LR scheduler | OneCycle | Cosine | OneCycle |
| Label smoothing | — | — | 0.055 |
| Weight decay | — | — | 2.94 × 10⁻⁶ |
| Best val acc (TPE) | — | — | 0.9091 |

### ViT-Small/16

| Hyperparameter | Value |
|---|---|
| Learning rate | 3.67 × 10⁻⁴ |
| Optimizer | AdamW |
| Batch size | 32 |
| Dropout | 0.115 |
| Layer decay | 0.801 |
| Unfreeze init | last_block |
| Focal loss | Yes (γ = 1.24) |
| Label smoothing | 0.081 |
| Warmup epochs | 4 |
| Best val acc (TPE) | 0.9310 |

### fANOVA hyperparameter importance

| Rank | DenseNet-121 | Score | InceptionV3 | Score | Xception | Score | ViT | Score |
|---|---|---|---|---|---|---|---|---|
| 1 | `lr` | 0.6227 | `lr` | 0.2812 | `dropout` | 0.2712 | `unfreeze_init` | 0.4508 |
| 2 | `use_focal` | 0.1081 | `weight_decay` | 0.1714 | `focal_gamma` | 0.2273 | `lr` | 0.1776 |
| 3 | `weight_decay` | 0.0991 | `focal_gamma` | 0.1666 | `label_smoothing` | 0.2188 | `layer_decay` | 0.1661 |

Learning rate dominates in DenseNet-121 (62% of variance). For Xception, the top three hyperparameters are more evenly distributed across dropout, focal gamma, and label smoothing. For ViT, the choice of which layer group to unfreeze first is the single most important factor.

---

## Repository Structure

```
.
├── data/
│   ├── Melanoma Cancer Image Dataset/
│   │   ├── test/
│   │   └── train/
│   └── Melanoma Skin Cancer Dataset of 10000 Images/
│       ├── test/
│       └── train/
├── main/
│   ├── densenet121/
│   │   ├── full_pipeline_results/
│   │   └── S1_DenseNet121.ipynb
│   ├── ensemble_learning/
│   │   ├── ensemble_results/
│   │   │   ├── confusion_matrix_all_models.png
│   │   │   ├── ensemble_weights.json
│   │   │   ├── final_summary.csv
│   │   │   ├── scenario2_confmat.png
│   │   │   ├── scenario2_results.json
│   │   │   ├── scenario2_vs_paper.png
│   │   │   ├── scenario2_weights_roc.png
│   │   │   ├── scenario3_comparison.png
│   │   │   ├── scenario3_confmat.png
│   │   │   └── scenario3_results.json
│   │   └── S2_S3_Ensemble_CrossDataset.ipynb
│   ├── inceptionv3/
│   │   ├── full_pipeline_results/
│   │   │   ├── confusion_matrix_inceptionv3.png
│   │   │   ├── ds1_labels.npy
│   │   │   ├── ds2_labels.npy
│   │   │   ├── probs_ds1_inception_v3.npy
│   │   │   ├── probs_ds2_inception_v3.npy
│   │   │   ├── tpe_analysis_inception_v3.png
│   │   │   ├── tpe_inception_v3.json
│   │   │   ├── val_labels.npy
│   │   │   └── val_probs_inception_v3.npy
│   │   └── S1_InceptionV3.ipynb
│   ├── vit/
│   │   ├── full_pipeline_results/
│   │   │   ├── confusion_matrix_vit.png
│   │   │   ├── ds1_labels.npy
│   │   │   ├── ds2_labels.npy
│   │   │   ├── probs_ds1_vit_scratch.npy
│   │   │   ├── probs_ds2_vit_scratch.npy
│   │   │   ├── tpe_analysis_vit_scratch.png
│   │   │   ├── tpe_vit_pretrained.json
│   │   │   ├── val_labels.npy
│   │   │   └── val_probs_vit_scratch.npy
│   │   └── S1_ViT.ipynb
│   └── xception/
│       ├── full_pipeline_results/
│       │   ├── confusion_matrix_xception.png
│       │   ├── ds1_labels.npy
│       │   ├── ds2_labels.npy
│       │   ├── probs_ds1_xception.npy
│       │   ├── probs_ds2_xception.npy
│       │   ├── tpe_analysis_xception.png
│       │   ├── tpe_xception.json
│       │   ├── val_labels.npy
│       │   ├── val_probs_xception.npy
│       │   └── xception_best.pth
│       ├── s1-xception-nb1-tpe.ipynb
│       └── s1-xception-nb2-training.ipynb
├── paper/
│   ├── melanoma-classification-cbir-replication.ipynb
│   └── Transfer Learning-Based Ensemble of CNNs and Vision Transformers for Accurate Mela....pdf
└── README.md
```

Each model lives in its own subdirectory under `main/`. Xception is split across two notebooks because TPE search and final training were run in separate Kaggle sessions — `tpe_xception.json` is saved by NB1 and loaded as an input dataset in NB2. The ensemble notebook and all final output files reside under `main/ensemble_learning/`.

---

## Datasets

| | DS1 | DS2 |
|---|---|---|
| Name | Melanoma Cancer Image Dataset | Melanoma Skin Cancer 10K |
| Source | [Kaggle / Bhavesh Mittal](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset) | [Kaggle / Hasnainjaved](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) |
| Train images | 11,879 | 9,605 |
| Test images | 2,000 | 1,000 |
| Resolution | 224 × 224 px | 300 × 300 px |
| Role | Primary training & evaluation | External generalisation test only |

Both datasets use two classes: **Benign** and **Malignant**.

---

## Methodology

### TPE search space

#### CNN models (DenseNet-121, InceptionV3, Xception)

| # | Parameter | Type | Range |
|---|---|---|---|
| 1 | `lr` | log-uniform | model-specific |
| 2 | `batch_size` | categorical | {32, 64} |
| 3 | `dropout` | uniform | model-specific |
| 4 | `weight_decay` | log-uniform | model-specific |
| 5 | `unfreeze_init` | categorical | {last_block, last_2_blocks} |
| 6 | `optimizer_name` | categorical | {Adam, AdamW, Adamax} |
| 7 | `use_focal` | categorical | {True, False} |
| 8 | `focal_gamma` | uniform | [0.5, 2.5] |
| 9 | `label_smoothing` | uniform | model-specific |
| 10 | `scheduler_type` | categorical | {cosine, onecycle} |

Sampler: `multivariate=True`, `n_startup_trials=10`. Trials: 30 per CNN model, 12 epochs each with early stopping (patience = 3).

#### ViT-Small/16

Same 10-parameter structure with ViT-specific substitutions: `layer_decay` replaces `scheduler_type`; `warmup_epochs` replaces `optimizer_name`; optimizer is fixed to AdamW. 15 trials, 8 epochs each.

### Progressive unfreeze schedule (final training, 35 epochs)

| Epoch range | Strategy | Approximate unfrozen params |
|---|---|---|
| 1 – 4 | Frozen backbone | Head only |
| 5 – 9 | last_block | ~15% |
| 10 – 17 | last_2_blocks | ~30% |
| 18 – 35 | Full fine-tune | All (`lr × 0.02`) |

During full fine-tuning, the classification head uses 5× the backbone learning rate. For ViT, Layer-wise Learning Rate Decay (LLRD) is applied instead.

### Ensemble weight optimisation

Weights are searched over 100 TPE trials. The objective is evaluated on the validation set only — DS1 test and DS2 data are never used during search.

```
Objective = 0.6 × Accuracy + 0.4 × AUC    (maximise)

w_i ~ Uniform[0.05, 1.0],  normalised so Σ w_i = 1
```

---

## Requirements

| Package | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.5.1+cu118 / cu121 |
| timm | 1.0.22 – 1.0.26 |
| Optuna | 4.8.0 |
| scikit-learn | ≥ 1.3 |

Training was run on an NVIDIA Tesla P100-PCIE-16GB (Kaggle). The ensemble notebook was run locally on an NVIDIA GeForce RTX 2050.

```bash
pip install torch torchvision timm optuna scikit-learn matplotlib seaborn pandas numpy
```

---

## Running the experiments

**Step 1.** Run the Scenario I notebooks on Kaggle (or any CUDA machine). Each saves its outputs to a `full_pipeline_results/` folder before the session ends.

```
S1_DenseNet121.ipynb
S1_InceptionV3.ipynb
s1-xception-nb1-tpe.ipynb   →  saves tpe_xception.json
s1-xception-nb2-training.ipynb   (loads tpe_xception.json as input dataset)
S1_ViT.ipynb
```

**Step 2.** Download each model's `full_pipeline_results/` folder and place it under the corresponding subdirectory (`main/densenet121/`, `main/inceptionv3/`, `main/vit/`, `main/xception/`). Set `BASE_DIR` in Cell 6 of the ensemble notebook:

```python
BASE_DIR = Path(r'D:\Tree-Structured Parzen Estimator\main')  # Windows
# BASE_DIR = Path('/home/user/project/main')                  # Linux / macOS
```

**Step 3.** Run `main/ensemble_learning/S2_S3_Ensemble_CrossDataset.ipynb` locally. Results are saved to `main/ensemble_learning/ensemble_results/`. No GPU is required for this step.

---

## Reference

Sarıateş, M.; Özbay, E. Transfer Learning-Based Ensemble of CNNs and Vision Transformers for Accurate Melanoma Diagnosis and Image Retrieval. *Diagnostics* **2025**, *15*, 1928. https://doi.org/10.3390/diagnostics15151928

---

## Author

Rofiq Samanhudi (202210370311260)  
Program Studi Informatika, Universitas Muhammadiyah Malang  
Supervisor: Dr. Ir. Agus Eko Minarno, S.Kom., M.Kom., IPM.