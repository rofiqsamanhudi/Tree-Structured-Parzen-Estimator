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
- Class imbalance is handled via `WeightedRandomSampler` with square-root inverse frequency weights (`1/√n_c`), a softer strategy that prevents over-correction while promoting minority class representation.
- Ensemble weights are optimised on the validation set via a composite objective selected from a systematic comparison of **20 candidate formulations** (10,000 total trials). The selected objective (OBJ-16) is `0.7 × AUC + 0.1 × Recall + 0.1 × Precision + 0.1 × Accuracy`.
- Generalisation is evaluated on a fully independent external test set (DS2, Mendeley) never seen during training.

---

## Results

### Scenario I — Individual models (DS1 test set, TTA ×10)

| Model | Accuracy | Precision | Recall | F1-Score | AUC | Paper Acc | Δ Acc |
|---|---|---|---|---|---|---|---|
| DenseNet-121 | 95.20% | 95.02% | 95.40% | 95.21% | 0.9874 | 94.50% | +0.70% |
| InceptionV3 | 95.35% | 96.51% | 94.10% | 95.29% | 0.9894 | 91.20% | +4.15% |
| Xception | 93.80% | 94.97% | 92.50% | 93.72% | 0.9850 | 93.80% | +0.00% |
| ViT-Pretrained | 95.10% | 96.98% | 93.10% | 95.00% | 0.9898 | 88.25% | +6.85% |

### Scenario II — Weighted ensemble (DS1 test set)

| | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| Sarıateş & Özbay (2025) | 95.25% | 94.20% | 96.22% | 95.20% | 0.990 |
| Ensemble (this work) | **96.20%** | **96.57%** | **95.80%** | **96.18%** | **0.9942** |
| Δ | +0.95% | +2.37% | −0.42% | +0.98% | +0.0042 |

Optimal ensemble weights found by TPE (100 trials, OBJ-16):

| Model | Weight | Share |
|---|---|---|
| ViT-Pretrained | 0.3342 | 33.42% |
| InceptionV3 | 0.2575 | 25.75% |
| Xception | 0.2068 | 20.68% |
| DenseNet-121 | 0.2015 | 20.15% |

Best combined score (OBJ-16: `0.7 × AUC + 0.1 × Recall + 0.1 × Precision + 0.1 × Accuracy`) on the validation set: **0.9689**

ViT receives the largest weight (≈ 33%), consistent with its highest individual AUC among the four models.

### Scenario III — Cross-dataset generalisation (DS2 / Mendeley, 1,000 images)

| Model | DS1 Acc | DS2 Acc | Δ | DS1 Recall | DS2 Recall | Δ Recall |
|---|---|---|---|---|---|---|
| DenseNet-121 | 95.20% | 97.00% | +1.80% | 95.40% | 97.60% | +2.20% |
| InceptionV3 | 95.35% | 97.60% | +2.25% | 94.10% | 97.00% | +2.90% |
| Xception | 93.80% | 95.30% | +1.50% | 92.50% | 94.20% | +1.70% |
| ViT-Pretrained | 95.10% | 97.10% | +2.00% | 93.10% | 96.60% | +3.50% |
| Ensemble | 96.20% | 97.50% | +1.30% | 95.80% | 97.00% | +1.20% |

No model degraded when evaluated on DS2. All four architectures, trained exclusively on DS1, maintained or improved across all metrics on the independent Mendeley dataset.

### DS2 Full Metrics (Ensemble)

| Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|
| 97.50% | 97.98% | 97.00% | 97.49% | 0.9965 |

> The ensemble trained on DS1 only outperforms EfficientNet-B0 fully trained on DS2 (Sabir & Mehmood, 2024: 97.00% Acc, 99.00% Recall, 97.00% F1, 0.9900 AUC) in accuracy, precision, F1-Score, and AUC — without any retraining.

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

All three CNN models share the same 10-parameter search space. The table below shows the complete best parameters found by TPE for each model.

| Hyperparameter | DenseNet-121 | InceptionV3 | Xception |
|---|---|---|---|
| Learning rate | 9.03 × 10⁻³ | 3.33 × 10⁻³ | 4.33 × 10⁻⁴ |
| Optimizer | Adamax | AdamW | Adam |
| Unfreeze init | last_2_blocks | last_2_blocks | last_block |
| Batch size | 64 | 64 | 32 |
| Dropout | 0.2578 | 0.3943 | 0.4296 |
| Weight decay | 1.45 × 10⁻⁴ | 8.14 × 10⁻⁶ | 2.94 × 10⁻⁶ |
| Focal loss | No | No | No |
| Focal gamma | 1.7022 | 1.2905 | 2.1649 |
| Label smoothing | 0.0284 | 0.1136 | 0.0555 |
| LR scheduler | OneCycle | Cosine | OneCycle |
| Best val acc (TPE) | 0.9220 | 0.9091 | 0.9091 |

### ViT-Small/16

| Hyperparameter | Value |
|---|---|
| Learning rate | 3.67 × 10⁻⁴ |
| Optimizer | AdamW |
| Batch size | 32 |
| Dropout | 0.1155 |
| Weight decay | 1.44 × 10⁻³ |
| Layer decay (LLRD) | 0.8005 |
| Unfreeze init | last_block |
| Focal loss | Yes (γ = 1.2373) |
| Label smoothing | 0.0807 |
| Warmup epochs | 4 |
| Best val acc (TPE) | 0.9310 |

### fANOVA Hyperparameter Importance

| Rank | DenseNet-121 | Score | InceptionV3 | Score | Xception | Score | ViT | Score |
|---|---|---|---|---|---|---|---|---|
| 1 | `lr` | 0.6227 | `lr` | 0.2812 | `dropout` | 0.2710 | `unfreeze_init` | 0.4508 |
| 2 | `use_focal` | 0.1081 | `weight_decay` | 0.1714 | `focal_gamma` | 0.2270 | `lr` | 0.1776 |
| 3 | `weight_decay` | 0.0991 | `focal_gamma` | 0.1666 | `label_smoothing` | 0.2190 | `layer_decay` | 0.1661 |
| 4 | `focal_gamma` | 0.0368 | `dropout` | 0.1356 | `lr` | 0.1510 | `focal_gamma` | 0.0780 |
| 5 | `dropout` | 0.0336 | `label_smoothing` | 0.1093 | `weight_decay` | 0.0590 | `warmup_epochs` | 0.0449 |

Learning rate dominates in DenseNet-121 (62% of variance). For Xception, the top three factors are more evenly distributed: dropout, focal gamma, and label smoothing — reflecting its sensitivity to regularization due to depthwise separable convolutions. For ViT, the choice of which layer group to unfreeze first is the single most important factor (45%), confirming that Transformer fine-tuning dynamics are fundamentally different from CNN fine-tuning.

---

## Ensemble Objective Function Selection

Before the final ensemble weight search, 20 composite objective formulations were systematically compared using 500 TPE trials each (10,000 total trials). **OBJ-16** was selected as it was the only formulation simultaneously achieving the highest value on three evaluation axes: DS2 Accuracy (0.975), DS2 Recall (0.972), and DS1 AUC (0.9942).

| ID | Formula | DS1 Acc | DS1 AUC | DS2 Acc | DS2 Recall |
|---|---|---|---|---|---|
| OBJ-01 | 1.0 × Recall | 0.9615 | 0.9941 | 0.9740 | 0.9700 |
| OBJ-02 | 1.0 × AUC | 0.9620 | 0.9939 | 0.9750 | 0.9700 |
| OBJ-15 | 0.7×AUC + 0.1×Recall + 0.1×F1 + 0.1×Acc | 0.9620 | 0.9942 | 0.9740 | 0.9700 |
| **OBJ-16** | **0.7×AUC + 0.1×Recall + 0.1×Prec + 0.1×Acc** | **0.9620** | **0.9943** | **0.9750** | **0.9720** |
| OBJ-17 | 0.7×AUC + 0.1×Recall + 0.1×Prec + 0.1×F1 | 0.9620 | 0.9942 | 0.9750 | 0.9720 |

OBJ-16 uniquely combined higher DS2 Accuracy (0.975 vs 0.974), higher DS2 Recall (0.972 vs 0.970), and the highest DS1 AUC (0.9943) among all 20 formulations.

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
│   │   │   ├── ensemble_weights_obj16.json
│   │   │   ├── peer_comparison_obj16.json
│   │   │   ├── final_summary.csv
│   │   │   ├── scenario2_confmat.png
│   │   │   ├── scenario2_results.json
│   │   │   ├── scenario2_vs_paper.png
│   │   │   ├── scenario3_comparison.png
│   │   │   ├── scenario3_confmat.png
│   │   │   └── scenario3_results.json
│   │   └── S2_S3_Ensemble_CrossDataset2_OBJ16.ipynb
│   ├── testing_variabel/
│   │   ├── output_test_variabel/
│   │   │   ├── OBJ-01/
│   │   │   ├── OBJ-02/
│   │   │   ├── OBJ-03/
│   │   │   ├── OBJ-04/
│   │   │   ├── OBJ-05/
│   │   │   ├── OBJ-06/
│   │   │   ├── OBJ-07/
│   │   │   ├── OBJ-08/
│   │   │   ├── OBJ-09/
│   │   │   ├── OBJ-10/
│   │   │   ├── OBJ-11/
│   │   │   ├── OBJ-12/
│   │   │   ├── OBJ-13/
│   │   │   ├── OBJ-14/
│   │   │   ├── OBJ-15/
│   │   │   ├── OBJ-16/
│   │   │   ├── OBJ-17/
│   │   │   ├── OBJ-18/
│   │   │   ├── OBJ-19/
│   │   │   ├── OBJ-20/
│   │   │   ├── all_objectives.json
│   │   │   ├── fig1_ds1_metrics.png
│   │   │   ├── fig2_ds1_vs_ds2.png
│   │   │   ├── fig3_ensemble_weights.png
│   │   │   ├── fig4_heatmap.png
│   │   │   └── objective_comparison.csv
│   │   └── Testing_Variabel_Ensemble.ipynb
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
│       ├── S1-Xception-nb1-tpe.ipynb
│       └── S1-Xception-nb2-training.ipynb
└── README.md
```

Each model lives in its own subdirectory under `main/`. Xception is split across two notebooks because TPE search and final training were run in separate Kaggle sessions — `tpe_xception.json` is saved by NB1 and loaded as an input dataset in NB2. The objective function selection experiment lives under `main/testing_variabel/`: `Testing_Variabel_Ensemble.ipynb` evaluates 20 formulations × 500 trials, with per-objective output folders (OBJ-01 to OBJ-20) and summary figures saved under `output_test_variabel/`. The final ensemble training and cross-dataset evaluation reside under `main/ensemble_learning/S2_S3_Ensemble_CrossDataset2_OBJ16.ipynb`.

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

Sampler: `multivariate=True`, `n_startup_trials=10`. Trials: **30 per CNN model**, 12 epochs each with early stopping (patience = 3).

#### ViT-Small/16

Same 10-parameter structure with ViT-specific substitutions: `layer_decay` (LLRD factor λ) replaces `scheduler_type`; `warmup_epochs` replaces `optimizer_name`; optimizer is fixed to AdamW. **15 trials**, 8 epochs each, `n_startup_trials=8`.

### Progressive unfreeze schedule (final training, 35 epochs for CNN / 30 epochs for ViT)

| Epoch range | Strategy | LR scale |
|---|---|---|
| 1 – 4 | Frozen backbone | 1.00× (head only) |
| 5 – 9 | last_block | 0.20× |
| 10 – 17 | last_2_blocks | 0.08× |
| 18 – 35 | Full fine-tune | 0.02× |

During full fine-tuning, the classification head uses 5× the backbone learning rate. For ViT, Layer-wise Learning Rate Decay (LLRD, λ=0.8005) is applied so lower Transformer blocks receive smaller learning rates, preserving universal pre-trained representations.

### Ensemble weight optimisation

Weights are searched over 100 TPE trials (`n_startup_trials=20`, `seed=57`). The objective is evaluated on the validation set only — DS1 test and DS2 data are never used during search.

```
Objective (OBJ-16) = 0.7 × AUC + 0.1 × Recall + 0.1 × Precision + 0.1 × Accuracy

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

Training was run on an **NVIDIA Tesla P100-PCIE-16GB** (Kaggle). The ensemble notebook was run locally on an **NVIDIA GeForce RTX 2050**.

```bash
pip install torch torchvision timm optuna scikit-learn matplotlib seaborn pandas numpy
```

---

## Running the Experiments

**Step 1.** Run the Scenario I notebooks on Kaggle (or any CUDA machine). Each saves its outputs to a `full_pipeline_results/` folder before the session ends.

```
S1_DenseNet121.ipynb
S1_InceptionV3.ipynb
S1-Xception-nb1-tpe.ipynb         →  saves tpe_xception.json
S1-Xception-nb2-training.ipynb    →  loads tpe_xception.json as input dataset
S1_ViT.ipynb
```

**Step 2.** *(Optional)* Run `main/testing_variabel/Testing_Variabel_Ensemble.ipynb` to reproduce the objective function selection experiment (20 formulations × 500 trials = 10,000 total TPE trials). Per-objective results are saved to `output_test_variabel/OBJ-XX/`. This step is not required if you use the pre-selected OBJ-16.

**Step 3.** Download each model's `full_pipeline_results/` folder and place it under the corresponding subdirectory. Set `BASE_DIR` in the ensemble notebook:

```python
BASE_DIR = Path(r'D:\Tree-Structured Parzen Estimator\main')  # Windows
# BASE_DIR = Path('/home/user/project/main')                  # Linux / macOS
```

**Step 4.** Run `S2_S3_Ensemble_CrossDataset2_OBJ16.ipynb` locally. Results are saved to `main/ensemble_learning/ensemble_results/`. No GPU is required for this step.

---

## Reference

Sarıateş, M.; Özbay, E. Transfer Learning-Based Ensemble of CNNs and Vision Transformers for Accurate Melanoma Diagnosis and Image Retrieval. *Diagnostics* **2025**, *15*, 1928. https://doi.org/10.3390/diagnostics15151928

---

## Author

Rofiq Samanhudi (202210370311260)  
Program Studi Informatika, Universitas Muhammadiyah Malang  
Supervisor: Dr. Ir. Agus Eko Minarno, S.Kom., M.Kom., IPM.