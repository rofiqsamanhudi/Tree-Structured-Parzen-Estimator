# Model Klasifikasi Melanoma

Model deep learning untuk klasifikasi citra dermoskopi kulit menjadi dua kelas: **benign** (jinak) dan **malignant** (ganas). Model dibangun menggunakan pendekatan ensemble transfer learning dari empat arsitektur CNN dan Vision Transformer.

## Isi Folder

| File | Keterangan |
|---|---|
| `densenet121_best.pth` | Model DenseNet-121 |
| `inception_v3_best.pth` | Model InceptionV3 |
| `xception_best.pth` | Model Xception |
| `vit_best.pth` | Model ViT-Small/16 (Vision Transformer) |
| `ensemble_best.pth` | Gabungan keempat model di atas dengan bobot ensemble |
| `model_def.py` | Kode arsitektur untuk memuat dan menjalankan model |

Kelima model bisa dipakai secara terpisah (per arsitektur) maupun sebagai satu kesatuan ensemble, tergantung kebutuhan.

## Library yang Dibutuhkan

```
torch>=2.0
timm>=1.0
Pillow
torchvision
```

Instalasi:

```bash
pip install torch timm pillow torchvision
```

## Cara Pemakaian

Contoh menjalankan prediksi menggunakan ensemble (4 model sekaligus):

```python
from model_def import MelanomaEnsemble

model = MelanomaEnsemble("ensemble_best.pth", device="cpu")
result = model.predict("gambar_kulit.jpg")

print(result)
```

Output berupa hasil prediksi dari masing-masing model dan hasil akhir ensemble:

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

## Spesifikasi Model

| Model | Ukuran Input | Bobot Ensemble |
|---|---|---|
| DenseNet-121 | 224x224 | 0.4706 |
| InceptionV3 | 299x299 | 0.0893 |
| Xception | 299x299 | 0.2850 |
| ViT-Small/16 | 224x224 | 0.1551 |

Normalisasi gambar menggunakan standar ImageNet (mean: `[0.485, 0.456, 0.406]`, std: `[0.229, 0.224, 0.225]`), sudah otomatis ditangani di dalam `model_def.py`.

Urutan label kelas: `["benign", "malignant"]`.

## Sumber Dataset

Model dilatih menggunakan dataset dermoskopi publik:
- [Melanoma Cancer Dataset (Bhavesh Mittal)](https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset)
- [Melanoma Skin Cancer Dataset of 10000 Images (Hasnain Javed)](https://data.mendeley.com/datasets/ggh6g39ps2/2)

## Authorship

Model ini dikembangkan sebagai bagian dari penelitian skripsi:

**Rofiq Samanhudi**
NIM 202210370311260
Program Studi Informatika, Universitas Muhammadiyah Malang

Dosen Pembimbing: Ir. Agus Eko Minarno S.Kom., M.Kom. IPM.

Judul penelitian: Peningkatan Performa Klasifikasi Melanoma Berbasis Ensemble Transfer Learning Menggunakan Bayesian Optimization dengan Tree-structured Parzen Estimator