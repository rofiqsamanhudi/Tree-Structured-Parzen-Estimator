# Melanoma Classification via Ensemble Transfer Learning with Bayesian Optimisation (TPE)

> **Undergraduate Thesis Project — Informatics, Universitas Muhammadiyah Malang**
>
> *Peningkatan Performa Klasifikasi Melanoma Berbasis Ensemble Transfer Learning Menggunakan Bayesian Optimization dengan Tree-structured Parzen Estimator*

---

## Overview

This repository contains the full experimental pipeline for automated melanoma classification using an ensemble of four pre-trained deep learning architectures. Hyperparameters for each individual model are optimised automatically via **Bayesian Optimisation with Tree-structured Parzen Estimator (TPE)** using [Optuna](https://optuna.org/), replacing the static configurations used in the reference paper. The ensemble fusion weights are subsequently found through a second TPE search, maximising a dual objective on the held-out validation set.

The work replicates and extends the ensemble framework proposed by **Sarıateş & Özbay (2025)** (*Diagnostics*, 15, 1928), with the following principal contributions:

- Per-model automatic hyperparameter search (10 HP per model, 15–30 trials)
- Multivariate TPE sampling that models correlations between hyperparameters (fANOVA importance analysis)
- Progressive layer-unfreezing schedule with adaptive learning-rate scaling to prevent catastrophic forgetting
- Dual-objective ensemble weight optimisation (0.6 × Accuracy + 0.4 × AUC)
- Cross-dataset generalisation evaluation on an independent external test set (DS2 / Mendeley)

---

## Results Summary

### Scenario I — Individual Model Performance (DS1 Test Set, TTA ×10)

| Model | Accuracy | Precision | Recall | F1-Score | AUC | Paper Acc | Δ vs Paper |
|---|---|---|---|---|---|---|---|
| DenseNet-121 | **95.20%** | 95.02% | 95.40% | 95.21% | 0.9874 | 94.50% | **+0.70%** |
| InceptionV3 | **95.35%** | 96.51% | 94.10% | 95.29% | 0.9894 | 91.20% | **+4.15%** |
| Xception | 93.00% | 94.70% | 91.10% | 92.86% | 0.9818 | 93.80% | −0.80% |
| ViT-Pretrained | **95.10%** | 96.98% | 93.10% | 95.00% | 0.9898 | 88.25% | **+6.85%** |

### Scenario II — Weighted Ensemble (DS1 Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| **Ensemble (ours)** | **96.05%** | **95.77%** | **94.90%** | **96.04%** | **0.9940** |
| Paper reference | 95.25% | 94.20% | 96.22% | 95.20% | — |
| **Δ vs Paper** | **+0.80%** | +1.57% | −1.32% | +0.84% | — |

**Optimal ensemble weights (TPE):**

| Model | Weight | Share |
|---|---|---|
| DenseNet-121 | 0.1736 | 17.4% |
| InceptionV3 | 0.1886 | 18.9% |
| Xception | 0.1483 | 14.8% |
| ViT-Pretrained | 0.4894 | **48.9%** |

### Scenario III — Cross-Dataset Generalisation (DS2 / Mendeley, 1,000 images)

| Model | DS1 Acc | DS2 Acc | Delta | DS1 Recall | DS2 Recall |
|---|---|---|---|---|---|
| DenseNet-121 | 95.20% | **97.00%** | +1.80% | 95.40% | 97.60% |
| InceptionV3 | 95.35% | **97.60%** | +2.25% | 94.10% | 97.00% |
| Xception | 93.00% | 93.80% | +0.80% | 91.10% | 92.40% |
| ViT-Pretrained | 95.10% | **97.10%** | +2.00% | 93.10% | 96.20% |
| **Ensemble** | **96.05%** | **97.30%** | +1.25% | 94.90% | 96.80% |

> No performance degradation was observed on DS2. All models trained exclusively on DS1 maintained or improved accuracy on the independent Mendeley dataset, confirming the generalisation capability of the TPE-optimised pipeline.

---

## Optimal Hyperparameters Found by TPE

| Hyperparameter | DenseNet-121 | InceptionV3 | Xception |
|---|---|---|---|
| Learning rate | 9.03e-3 | 3.33e-3 | 3.80e-3 |
| Optimizer | Adamax | AdamW | Adamax |
| Unfreeze init | last\_2\_blocks | last\_2\_blocks | last\_block |
| Batch size | 64 | 64 | 32 |
| Dropout | 0.258 | 0.220 | 0.380 |
| use\_focal | False | False | False |
| Scheduler | onecycle | cosine | cosine |

### Top Hyperparameter Importance (fANOVA)

| Rank | DenseNet-121 | Score | InceptionV3 | Score | Xception | Score |
|---|---|---|---|---|---|---|
| 1 | lr | 0.6227 | lr | 0.2812 | lr | 0.3853 |
| 2 | use\_focal | 0.1081 | weight\_decay | 0.1714 | focal\_gamma | 0.3530 |
| 3 | weight\_decay | 0.0991 | focal\_gamma | 0.1666 | label\_smoothing | 0.0975 |

---

## Repository Structure

```
.
├── S1_DenseNet121.ipynb              # Scenario I — DenseNet-121 TPE + final training
├── S1_InceptionV3.ipynb              # Scenario I — InceptionV3 TPE + final training
├── S1_Xception.ipynb                 # Scenario I — Xception TPE + final training
├── S1_ViT.ipynb                      # Scenario I — ViT-Pretrained TPE + final training
├── S2_S3_Ensemble_CrossDataset.ipynb # Scenario II (ensemble) + Scenario III (DS2)
└── README.md

main/                                 # Local results directory (not committed)
├── densenet121/full_pipeline_results/
│   ├── probs_ds1_densenet121.npy
│   ├── probs_ds2_densenet121.npy
│   ├── val_probs_densenet121.npy
│   ├── val_labels.npy
│   ├── ds1_labels.npy
│   ├── ds2_labels.npy
│   ├── tpe_densenet121.json
│   └── tpe_analysis_densenet121.png
├── inceptionv3/full_pipeline_results/
├── xception/full_pipeline_results/
├── vit/full_pipeline_results/
└── ensemble_results/
    ├── ensemble_weights.json
    ├── final_summary.csv
    ├── scenario2_vs_paper.png
    ├── scenario2_weights_roc.png
    ├── scenario2_confmat.png
    ├── scenario3_comparison.png
    ├── scenario3_confmat.png
    ├── scenario2_results.json
    └── scenario3_results.json
```

---

## Datasets

| Dataset | Source | Train | Test | Resolution | Usage |
|---|---|---|---|---|---|
| **DS1** — Melanoma Cancer Image Dataset | [Kaggle / Bhavesh Mittal](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset) | 11,879 | 2,000 | 224×224 px | Training, validation, primary evaluation |
| **DS2** — Melanoma Skin Cancer 10K | [Kaggle / Hasnainjaved](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images) | 9,605 | 1,000 | 300×300 px | External test set (unseen during training) |

Both datasets contain two binary classes: **Benign** and **Malignant**.

---

## Methodology

### TPE Search Space (per model)

Each individual model is optimised over 10 hyperparameters using Optuna's multivariate TPE sampler (`n_startup_trials=10`, `multivariate=True`):

| # | Hyperparameter | Type | CNN Range | ViT Range |
|---|---|---|---|---|
| 1 | `lr` | log-uniform | model-specific | [5e-6, 5e-4] |
| 2 | `batch_size` | categorical | {32, 64} | {16, 32} |
| 3 | `dropout` | uniform | model-specific | [0.05, 0.25] |
| 4 | `weight_decay` | log-uniform | model-specific | [1e-4, 1e-2] |
| 5 | `unfreeze_init` | categorical | {last\_block, last\_2\_blocks} | same |
| 6 | `optimizer_name` | categorical | {adam, adamw, adamax} | AdamW (fixed) |
| 7 | `use_focal` | categorical | {True, False} | {True, False} |
| 8 | `focal_gamma` | uniform | [0.5, 2.5] | [0.5, 1.5] |
| 9 | `label_smoothing` | uniform | model-specific | [0.05, 0.15] |
| 10 | `scheduler_type` / `layer_decay` / `warmup_epochs` | — | {cosine, onecycle} | ViT-specific |

### Progressive Unfreeze Schedule (Final Training)

| Epoch range | Strategy | CNN params unfrozen |
|---|---|---|
| 1 – 4 | `frozen` | Head only |
| 5 – 9 | `last_block` | Top 15% |
| 10 – 17 | `last_2_blocks` | Top 30% |
| 18 → end | `full` | All (lr × 0.02) |

LR is scaled per phase to prevent catastrophic forgetting: `full` phase uses `lr × 0.02` with differential LR (head 5× backbone).

### Ensemble Weight Optimisation (Scenario II)

Weights are searched over 100 TPE trials on the **validation set only** (no test set leakage):

```
Objective = 0.6 × Accuracy + 0.4 × AUC   (maximise)
```

Each weight `w_i ~ Uniform[0.05, 1.0]`, normalised so `Σw_i = 1`.

---

## Environment

| Component | Version |
|---|---|
| Python | 3.10+ |
| PyTorch | 2.5.1+cu121 |
| timm | 1.0.22 |
| Optuna | 4.8.0 |
| scikit-learn | ≥ 1.3 |
| GPU (training) | NVIDIA P100 (Kaggle) |
| GPU (local) | NVIDIA GeForce RTX 2050 |

### Installation

```bash
pip install torch torchvision timm optuna scikit-learn matplotlib seaborn pandas numpy
```

---

## Execution Order

```
# Step 1 — Train each model independently on Kaggle GPU
S1_DenseNet121.ipynb      →  saves probs + tpe params to /full_pipeline_results/
S1_InceptionV3.ipynb      →  saves probs + tpe params
S1_Xception.ipynb         →  saves probs + tpe params
S1_ViT.ipynb              →  saves probs + tpe params

# Step 2 — Run ensemble + cross-dataset evaluation locally
S2_S3_Ensemble_CrossDataset.ipynb   (loads .npy files, no GPU required)
```

Set `BASE_DIR` in Cell 3 of `S2_S3_Ensemble_CrossDataset.ipynb` to match your local path:

```python
BASE_DIR = Path(r'D:\Tree-Structured Parzen Estimator\main')  # Windows
# BASE_DIR = Path('/home/user/project/main')                  # Linux / macOS
```

---

## Reference

> Sarıateş, M.; Özbay, E. *Transfer Learning-Based Ensemble of CNNs and Vision Transformers for Accurate Melanoma Diagnosis and Image Retrieval.* **Diagnostics** 2025, 15, 1928. https://doi.org/10.3390/diagnostics15151928

---

## Author

**Rofiq Samanhudi** — 202210370311260
Program Studi Informatika, Universitas Muhammadiyah Malang
Supervisor: Dr. Ir. Agus Eko Minarno, S.Kom., M.Kom., IPM.
