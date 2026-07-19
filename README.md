# Melanoma Classification Model

A deep learning model for classifying dermoscopic skin images into two categories: **benign** and **malignant**. The model is built using an ensemble transfer learning approach combining four CNN architectures and a Vision Transformer.

## Folder Contents

| File | Description |
|---|---|
| `densenet121_best.pth` | DenseNet-121 model |
| `inception_v3_best.pth` | InceptionV3 model |
| `xception_best.pth` | Xception model |
| `vit_best.pth` | ViT-Small/16 (Vision Transformer) model |
| `ensemble_best.pth` | Combination of all four models above using ensemble weights |
| `model_def.py` | Architecture code for loading and running the models |

Each of the five models can be used individually (per architecture) or as a unified ensemble, depending on the use case.

## Requirements

- Python 3.9 or later
- **torch** — core framework for loading and running the models
- **torchvision** — image transformations for preprocessing
- **timm** — required specifically for the ViT-Small/16 and Xception architectures, which are loaded through this library
- **Pillow** — reading and processing input image files

### Installation (CPU)

```bash
pip install torch torchvision timm pillow
```

### Installation (GPU, CUDA 12.1)

To run on GPU, install `torch`/`torchvision` from the official PyTorch index matching your system's CUDA version (check your CUDA version with `nvidia-smi`):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm pillow
```

For other CUDA versions, adjust `cu121` in the URL accordingly (e.g., `cu118`, `cu124`). See the full list at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

### Versions Used During Development

To reproduce the research results exactly, the following library versions were used during development (trained with CUDA 12.1):

```
torch==2.5.1+cu121
torchvision==0.20.1+cu121
timm==1.0.22
Pillow==12.0.0
```

> **Note:** The `+cu121` suffix indicates the GPU build of PyTorch used during training. If you install via the CUDA 12.1 command above, `pip` will resolve to this exact build automatically. For CPU-only installs, use the plain versions instead (`torch==2.5.1`, `torchvision==0.20.1`), since the `+cu121` build is not available through the default PyPI index.

## Usage

Example of running a prediction using the ensemble (all four models at once):

```python
from model_def import MelanomaEnsemble

model = MelanomaEnsemble("ensemble_best.pth", device="cpu")
result = model.predict("skin_image.jpg")

print(result)
```

Output includes predictions from each individual model as well as the final ensemble result:

```python
{
  "ensemble": {
    "prediction": "benign",
    "confidence": 0.9516,
    "probabilities": {"benign": 0.9516, "malignant": 0.0484}
  },
  "models": {
    "densenet121": {"prediction": "benign", "confidence": 1.0, ...},
    "inception_v3": {"prediction": "benign", "confidence": 1.0, ...},
    "xception": {"prediction": "benign", "confidence": 0.85, ...},
    "vit_scratch": {"prediction": "benign", "confidence": 0.96, ...}
  }
}
```

## Model Specifications

| Model | Input Size | Ensemble Weight |
|---|---|---|
| DenseNet-121 | 224x224 | 0.4706 |
| InceptionV3 | 299x299 | 0.0893 |
| Xception | 299x299 | 0.2850 |
| ViT-Small/16 | 224x224 | 0.1551 |

Image normalization follows the standard ImageNet convention (mean: `[0.485, 0.456, 0.406]`, std: `[0.229, 0.224, 0.225]`), handled automatically within `model_def.py`.

Class label order: `["benign", "malignant"]`.

## Dataset Sources

The model was trained on the following public dermoscopic datasets:
- [Melanoma Cancer Dataset (Bhavesh Mittal)](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset)
- [Melanoma Skin Cancer Dataset of 10000 Images (Hasnain Javed)](https://data.mendeley.com/datasets/ggh6g39ps2/2)

## Authorship

This model was developed as part of an undergraduate thesis research project:

**Rofiq Samanhudi**
Student ID 202210370311260
Department of Informatics, Universitas Muhammadiyah Malang

Thesis Advisor: Ir. Agus Eko Minarno S.Kom., M.Kom. IPM.

Research title (Eng): Enhanced Melanoma Classification Performance Based on Ensemble Transfer Learning Using Bayesian Optimization with Tree-Structured Parzen Estimator

Research title (Ind): Peningkatan Performa Klasifikasi Melanoma Berbasis Ensemble Transfer Learning Menggunakan Bayesian Optimization dengan Tree-structured Parzen Estimator