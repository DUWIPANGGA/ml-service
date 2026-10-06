"""Konfigurasi service dari environment variable (FR-10.6, FR-10.9, FR-10.10)."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Muat .env bila ada (opsional — tanpa dependensi tambahan).
_ENV_FILE = BASE_DIR / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _key, _, _value = _line.partition("=")
        os.environ.setdefault(_key.strip(), _value.strip())

# Model dasar di Hugging Face — arsitektur & processor diambil dari sini.
BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "google/vit-base-patch16-224")

# Checkpoint fine-tuned (11 kelas penyakit ikan).
# Bisa berupa folder save_pretrained (config.json + model.safetensors)
# atau satu file state_dict (.safetensors / .pt / .bin).
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(BASE_DIR / "models")))

# "auto" | "cpu" | "cuda" | "cuda:0" | "mps"
DEVICE = os.getenv("DEVICE", "auto")

MODEL_VERSION = os.getenv("MODEL_VERSION", "vit-v1.0.0")

LABEL_MAP_PATH = Path(os.getenv("LABEL_MAP_PATH", str(BASE_DIR / "label_map.json")))

# Threshold confidence (FR-02.2.3): di bawah ini -> low_confidence_warning.
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "60"))

# Hard limit inferensi (NFR-01.2) -> 504 INFERENCE_TIMEOUT.
PREDICT_TIMEOUT_SECONDS = float(os.getenv("PREDICT_TIMEOUT_SECONDS", "30"))

# Batas upload (API.md §2.7): 10 MB, maks 5 gambar per batch.
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
MAX_BATCH_IMAGES = int(os.getenv("MAX_BATCH_IMAGES", "5"))

# Batas deteksi kualitas gambar (FR-10.12).
# Variance-of-Laplacian: di bawah ini -> blur. Nilai ~100 wajar untuk foto tajam.
BLUR_MIN_VARIANCE = float(os.getenv("BLUR_MIN_VARIANCE", "60"))
# Brightness rata-rata (0-255): di luar rentang -> terlalu gelap / terlalu terang.
DARK_MAX_BRIGHTNESS = float(os.getenv("DARK_MAX_BRIGHTNESS", "40"))
BRIGHT_MIN_BRIGHTNESS = float(os.getenv("BRIGHT_MIN_BRIGHTNESS", "235"))

# Format file yang diterima (FR-02.1.1).
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

# GradCAM (FR-10.8) — diaktifkan per-request via ?gradcam=true.
GRADCAM_ENABLED = os.getenv("GRADCAM_ENABLED", "true").lower() in {"1", "true", "yes"}
