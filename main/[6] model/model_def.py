"""
model_def.py
============
Definisi arsitektur untuk model klasifikasi melanoma (ensemble 4 model).
Dipakai untuk load bobot dari file .pth dan melakukan inferensi.

Cara pakai cepat:
    from model_def import MelanomaEnsemble

    ensemble = MelanomaEnsemble("ensemble_best.pth", device="cpu")
    result = ensemble.predict("path/ke/gambar.jpg")
    print(result)
    # {'label': 'malignant', 'probability': 0.87, 'per_model': {...}}

Requirements:
    pip install torch timm pillow
"""

import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms

# =========================================================
# 1. Konstanta
# =========================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Ukuran input per sub-model (WAJIB sesuai training, jangan diubah)
IMG_SIZE = {
    "densenet121": 224,
    "inception_v3": 299,
    "xception": 299,
    "vit_scratch": 224,
}

# Urutan kelas: index 0 dan 1 HARUS dicek ulang terhadap
# `full_ds.classes` dari ImageFolder training kamu (ImageFolder mengurutkan
# nama folder secara alfabetis). Ganti sesuai kelas asli kamu.
CLASS_NAMES = ["benign", "malignant"]  # <-- SESUAIKAN jika urutan aslinya beda

NUM_CLASSES = 2

# Nama tampilan tiap model, untuk label card di dashboard
MODEL_LABELS = {
    "densenet121": "DenseNet-121",
    "inception_v3": "InceptionV3",
    "xception": "Xception",
    "vit_scratch": "ViT-Pretrained",
}


# =========================================================
# 2. Arsitektur CNN (DenseNet121 / InceptionV3 / Xception)
# =========================================================

class CNNWithHead(nn.Module):
    """CNN backbone (timm) + custom head: Linear -> BN -> GELU -> Dropout -> Linear."""

    def __init__(self, backbone, n_features, dropout, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


def build_cnn(arch_name: str, dropout: float = 0.3, pretrained: bool = False):
    """Bangun CNN (densenet121 / inception_v3 / xception) dengan head custom."""
    backbone = timm.create_model(
        arch_name, pretrained=pretrained, drop_rate=dropout, num_classes=0
    )
    n_features = backbone.num_features
    return CNNWithHead(backbone, n_features, dropout, NUM_CLASSES)


# =========================================================
# 3. Arsitektur ViT
# =========================================================

def build_vit(dropout: float = 0.1, pretrained: bool = False):
    """Bangun ViT-Small/16 dengan head custom 2-lapis."""
    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=pretrained,
        drop_rate=dropout,
        num_classes=NUM_CLASSES,
    )
    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(256, NUM_CLASSES),
    )
    return model


# =========================================================
# 4. Preprocessing
# =========================================================

def get_eval_transform(img_size: int):
    """Transform untuk inferensi (tanpa augmentasi): resize + normalize."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


# =========================================================
# 5. Wrapper Ensemble — siap pakai untuk backend website
# =========================================================

class MelanomaEnsemble:
    """
    Load ensemble_best.pth dan lakukan prediksi end-to-end.
    Cocok dipakai langsung di backend (FastAPI/Flask) untuk dashboard.
    """

    ARCH_BUILDERS = {
        "densenet121": lambda: build_cnn("densenet121"),
        "inception_v3": lambda: build_cnn("inception_v3"),
        "xception": lambda: build_cnn("xception"),
        "vit_scratch": build_vit,
    }

    def __init__(self, ensemble_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(ensemble_path, map_location=self.device, weights_only=False)

        self.weights = ckpt["weights"]  # dict: {model_key: bobot}
        self.model_keys = ckpt["model_keys"]

        self.models = {}
        for key in self.model_keys:
            model = self.ARCH_BUILDERS[key]()
            state_dict = self._extract_submodel_state_dict(ckpt["state_dict"], key)
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device).eval()
            self.models[key] = model

        self.transforms = {key: get_eval_transform(IMG_SIZE[key]) for key in self.model_keys}

    @staticmethod
    def _extract_submodel_state_dict(full_state_dict, model_key):
        """
        ensemble_best.pth menyimpan state_dict gabungan dengan prefix
        'models.<model_key>.' untuk tiap sub-model. Fungsi ini memisahkannya
        kembali per sub-model.
        """
        prefix = f"models.{model_key}."
        sub = {
            k[len(prefix):]: v
            for k, v in full_state_dict.items()
            if k.startswith(prefix)
        }
        if not sub:
            raise KeyError(f"Tidak ada key dengan prefix '{prefix}' di state_dict.")
        return sub

    @torch.no_grad()
    def predict(self, image_path: str):
        """
        Jalankan prediksi dari 4 sub-model + 1 hasil ensemble.
        Formatnya dibuat supaya langsung bisa dipetakan ke 5 card di dashboard
        (4 model individual + 1 ensemble sebagai hasil akhir/rekomendasi).
        """
        img = Image.open(image_path).convert("RGB")

        models_result = {}
        weighted_sum = torch.zeros(NUM_CLASSES)

        for key, model in self.models.items():
            x = self.transforms[key](img).unsqueeze(0).to(self.device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

            pred_idx = int(torch.argmax(probs))
            models_result[key] = {
                "model_label": MODEL_LABELS[key],
                "prediction": CLASS_NAMES[pred_idx],
                "confidence": round(float(probs[pred_idx]), 4),
                "probabilities": {
                    CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(NUM_CLASSES)
                },
                "ensemble_weight": self.weights[key],
            }
            weighted_sum += self.weights[key] * probs

        ens_idx = int(torch.argmax(weighted_sum))
        ensemble_result = {
            "model_label": "Ensemble (Weighted Average)",
            "prediction": CLASS_NAMES[ens_idx],
            "confidence": round(float(weighted_sum[ens_idx]), 4),
            "probabilities": {
                CLASS_NAMES[i]: round(float(weighted_sum[i]), 4) for i in range(NUM_CLASSES)
            },
        }

        return {
            "ensemble": ensemble_result,   # hasil akhir/rekomendasi utama
            "models": models_result,       # dict 4 sub-model, key: densenet121/inception_v3/xception/vit_scratch
        }


if __name__ == "__main__":
    import sys

    ensemble_path = sys.argv[1] if len(sys.argv) > 1 else "ensemble_best.pth"
    image_path = sys.argv[2] if len(sys.argv) > 2 else None

    ens = MelanomaEnsemble(ensemble_path, device="cpu")
    print("Model berhasil dimuat. Sub-model:", list(ens.models.keys()))
    print("Bobot ensemble:", ens.weights)

    if image_path:
        result = ens.predict(image_path)
        print("\n--- Hasil per model ---")
        for key, r in result["models"].items():
            print(f"  {r['model_label']:<18} -> {r['prediction']:<10} (confidence: {r['confidence']})")
        print("\n--- Hasil ensemble (final) ---")
        e = result["ensemble"]
        print(f"  {e['model_label']:<18} -> {e['prediction']:<10} (confidence: {e['confidence']})")
