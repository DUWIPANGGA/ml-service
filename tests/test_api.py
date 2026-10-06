"""Test API ML Service (tanpa butuh checkpoint — classifier di-mock)."""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import config
from app.labels import load_labels
from app.main import app
from app.model import classifier


# ────────────────────────────────────────────────────────────────── fixtures


class _DummyModel:
    """Pengganti model Torch saat test — cukup untuk .parameters() & .eval()."""

    def parameters(self):
        return []

    def eval(self):
        return self


@pytest.fixture()
def fake_classifier(monkeypatch):
    """Classifier palsu: model 'ter-load', probabilitas deterministik."""
    labels = load_labels()
    # No-op lifespan: jangan coba load checkpoint asli saat test.
    monkeypatch.setattr(classifier, "load", lambda: None, raising=False)
    monkeypatch.setattr(classifier, "labels", labels)
    monkeypatch.setattr(classifier, "model", _DummyModel(), raising=False)
    monkeypatch.setattr(classifier, "processor", None, raising=False)
    monkeypatch.setattr(classifier, "device", "cpu", raising=False)
    monkeypatch.setattr(classifier, "error", None, raising=False)
    monkeypatch.setattr(classifier, "model_format", "safetensors", raising=False)
    monkeypatch.setattr(classifier, "source_path", "models/vit-fish-11", raising=False)
    monkeypatch.setattr(classifier, "loaded_at", "2026-10-06T00:00:00+00:00", raising=False)

    # Kelas index 1 (aeromoniasis) menang dengan confidence 87.3%
    probs = np.zeros(len(labels), dtype=np.float32)
    probs[1] = 0.873
    probs[4] = 0.046
    probs[5] = 0.081
    probs /= probs.sum()
    monkeypatch.setattr(classifier, "predict_probabilities", lambda image: probs)
    return classifier


@pytest.fixture()
def client(fake_classifier):
    with TestClient(app) as c:
        yield c


def _png_bytes(width: int = 224, height: int = 224, color=(120, 90, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _noisy_jpeg_bytes(width: int = 224, height: int = 224) -> bytes:
    """Gambar berisik -> variance-of-Laplacian tinggi (lolos deteksi blur)."""
    rng = np.random.default_rng(42)
    arr = rng.integers(60, 200, size=(height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG")
    return buf.getvalue()


def _dark_jpeg_bytes(width: int = 224, height: int = 224) -> bytes:
    arr = np.full((height, width, 3), 8, dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG")
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────────── /health

def test_health_without_model(monkeypatch):
    monkeypatch.setattr(classifier, "load", lambda: None, raising=False)
    monkeypatch.setattr(classifier, "model", None, raising=False)
    monkeypatch.setattr(classifier, "error", "MODEL_PATH tidak ditemukan", raising=False)
    with TestClient(app) as c:
        res = c.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "degraded"
    assert "model_version" in body
    assert "uptime_seconds" in body


def test_health_with_model(client):
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["model_format"] == "safetensors"


# ──────────────────────────────────────────────────────────────── /model/info

def test_model_info_matches_metadata_contract(client):
    res = client.get("/model/info")
    assert res.status_code == 200
    body = res.json()

    # Metadata model sesuai spesifikasi yang diberikan
    assert body["model_name"] == "google/vit-base-patch16-224"
    assert body["architecture"] == "Vision Transformer (ViT)"
    assert body["task"] == "Image Classification"
    assert body["pretrained_on"].startswith("ImageNet-21k")
    assert body["fine_tuned_on"].startswith("ImageNet-1k")
    assert "Dosovitskiy" in body["paper"]

    # Kontrak API.md §10.6
    assert body["model_version"] == config.MODEL_VERSION
    assert body["input_spec"]["height"] == 224
    assert body["input_spec"]["width"] == 224
    assert body["input_spec"]["channels"] == 3
    assert len(body["labels"]) == 11
    assert body["loaded_at"]


def test_model_info_without_model_returns_503(monkeypatch):
    monkeypatch.setattr(classifier, "load", lambda: None, raising=False)
    monkeypatch.setattr(classifier, "model", None, raising=False)
    monkeypatch.setattr(classifier, "error", "MODEL_PATH tidak ditemukan", raising=False)
    with TestClient(app) as c:
        res = c.get("/model/info")
    assert res.status_code == 503
    assert res.json()["detail"]["code"] == "MODEL_NOT_LOADED"


# ────────────────────────────────────────────────────────────────── /diseases

def test_diseases_returns_11_labels(client):
    res = client.get("/diseases")
    assert res.status_code == 200
    body = res.json()
    assert body["model_version"] == config.MODEL_VERSION
    assert len(body["labels"]) == 11
    for row in body["labels"]:
        assert set(row) >= {"label", "name", "slug", "severity", "disease_id"}
    # Label asli model (lihat models/config.json id2label)
    labels = [r["label"] for r in body["labels"]]
    assert "aeromoniasis" in labels
    assert "healthy" in labels
    assert "shrimp_white_spot" in labels


# ────────────────────────────────────────────────────────────────── /predict

def test_predict_success(client):
    res = client.post(
        "/predict",
        files={"image": ("ikan.jpg", _noisy_jpeg_bytes(), "image/jpeg")},
        data={"fish_type": "Lele"},
    )
    assert res.status_code == 200
    body = res.json()

    assert body["disease"]["label"] == "aeromoniasis"
    assert body["disease"]["slug"] == "aeromoniasis"
    assert body["disease"]["severity"] == "berat"
    assert body["confidence"] == pytest.approx(87.3, abs=0.1)
    assert len(body["top_predictions"]) == 3
    assert body["top_predictions"][0]["label"] == "aeromoniasis"
    assert body["low_confidence_warning"] is False
    assert body["gradcam_image"] is None
    assert body["processing_time_ms"] >= 0
    assert body["model_version"] == config.MODEL_VERSION


def test_predict_low_confidence_warning(client, fake_classifier, monkeypatch):
    probs = np.full(len(fake_classifier.labels), 1 / 11, dtype=np.float32)
    monkeypatch.setattr(fake_classifier, "predict_probabilities", lambda image: probs)

    res = client.post(
        "/predict",
        files={"image": ("ikan.jpg", _noisy_jpeg_bytes(), "image/jpeg")},
    )
    assert res.status_code == 200
    assert res.json()["low_confidence_warning"] is True


def test_predict_unsupported_media_type(client):
    res = client.post(
        "/predict",
        files={"image": ("catatan.txt", b"bukan gambar", "text/plain")},
    )
    assert res.status_code == 400
    assert res.json()["detail"] == "UNSUPPORTED_MEDIA_TYPE"


def test_predict_corrupt_image(client):
    res = client.post(
        "/predict",
        files={"image": ("rusak.jpg", b"isi tidak valid", "image/jpeg")},
    )
    assert res.status_code == 400
    assert res.json()["detail"] == "UNSUPPORTED_MEDIA_TYPE"


def test_predict_dark_image_quality_low(client):
    res = client.post(
        "/predict",
        files={"image": ("gelap.jpg", _dark_jpeg_bytes(), "image/jpeg")},
    )
    assert res.status_code == 422
    detail = res.json()["detail"]
    assert detail["code"] == "IMAGE_QUALITY_LOW"
    assert "gelap" in detail["message"].lower()
    assert detail["suggestions"]  # FR-10.12: sertakan saran perbaikan


def test_predict_without_model_returns_503(monkeypatch):
    monkeypatch.setattr(classifier, "load", lambda: None, raising=False)
    monkeypatch.setattr(classifier, "model", None, raising=False)
    monkeypatch.setattr(classifier, "error", "gagal memuat", raising=False)
    with TestClient(app) as c:
        res = c.post(
            "/predict",
            files={"image": ("ikan.jpg", _noisy_jpeg_bytes(), "image/jpeg")},
        )
    assert res.status_code == 503
    assert res.json()["detail"]["code"] == "MODEL_NOT_LOADED"


# ──────────────────────────────────────────────────────────── /predict/batch

def test_predict_batch_success(client):
    res = client.post(
        "/predict/batch",
        files=[
            ("images", ("a.jpg", _noisy_jpeg_bytes(), "image/jpeg")),
            ("images", ("b.jpg", _noisy_jpeg_bytes(), "image/jpeg")),
        ],
    )
    assert res.status_code == 200
    body = res.json()
    assert len(body["results"]) == 2
    assert body["aggregated"]["disease"]["label"] == "aeromoniasis"
    assert body["aggregated"]["method"] == "weighted_average"
    assert body["aggregated"]["confidence"] == pytest.approx(87.3, abs=0.5)
    assert body["processing_time_ms"] >= 0


def test_predict_batch_too_many_images(client):
    files = [("images", (f"{i}.jpg", _noisy_jpeg_bytes(), "image/jpeg")) for i in range(6)]
    res = client.post("/predict/batch", files=files)
    assert res.status_code == 422
    assert "1-5" in res.json()["detail"]
