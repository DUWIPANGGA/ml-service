"""ML Service — Deteksi Penyakit Ikan (ViT google/vit-base-patch16-224).

Endpoint mengikuti kontrak API.md §10 / SRS FR-10:
  POST /predict       prediksi 1 gambar (multipart)
  POST /predict/batch prediksi 2-5 gambar, hasil diagregasi
  GET  /diseases      label penyakit yang didukung model
  GET  /health        health check
  GET  /model/info    metadata model aktif
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from . import config
from .gradcam import generate_gradcam_b64
from .labels import load_labels
from .model import ModelNotLoadedError, classifier
from .preprocess import (
    ImageQualityError,
    ImageTooLargeError,
    UnsupportedMediaTypeError,
    decode_rgb,
    ensure_quality,
    validate_upload,
)

logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("ml.main")

START_TIME = time.time()

# Inference dijalankan di threadpool terpisah agar hard limit 30 dtk (NFR-01.2) bisa dipaksa.
_executor: concurrent.futures.ThreadPoolExecutor | None = None


def _get_executor() -> concurrent.futures.ThreadPoolExecutor:
    """Lazy — dibuat saat startup, dibuat ulang bila service di-restart."""
    global _executor
    if _executor is None:
        _executor = concurrent.futures.ThreadPoolExecutor(max_workers=int(os.getenv("PREDICT_WORKERS", "2")))
    return _executor


def _load_model_background() -> None:
    """Muat checkpoint di thread terpisah — lihat catatan di lifespan()."""
    try:
        classifier.load()
        logger.info("Startup OK — model %s siap", config.MODEL_VERSION)
    except Exception as exc:  # noqa: BLE001
        classifier.error = str(exc)
        classifier.state = "failed"
        logger.exception("Model GAGAL dimuat: %s", exc)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Muat model di BACKGROUND; gagal -> service tetap jalan (503, NFR-03.2).

    Sengaja tidak dipanggil sinkron: memuat checkpoint 327 MB butuh 40-60 detik,
    dan bila dilakukan di dalam lifespan, /health ikut macet selama itu — health
    check Docker/Kubernetes menganggap container mati lalu me-restart berulang.
    """
    _get_executor()
    loader = threading.Thread(target=_load_model_background, name="model-loader", daemon=True)
    loader.start()
    yield
    # Biasanya sudah selesai sejak lama; timeout hanya agar shutdown tidak
    # tersangkut bila load masih berjalan saat SIGTERM datang.
    loader.join(timeout=10)
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)
        _executor = None


app = FastAPI(
    title="Ngingu ML Service — Deteksi Penyakit Ikan",
    description="Prediksi penyakit ikan dari gambar menggunakan ViT fine-tuned (11 kelas).",
    version="1.0.0",
    lifespan=lifespan,
)


# ────────────────────────────────────────────────────────────── helper prediksi

def _ensure_loaded() -> None:
    if not classifier.is_loaded:
        if classifier.is_loading:
            message = "Model masih dimuat, coba lagi dalam beberapa detik."
        else:
            message = classifier.error or "Model belum dimuat"
        raise HTTPException(
            status_code=503,
            detail={"code": "MODEL_NOT_LOADED", "message": message},
        )


def _run_with_timeout(image: Image.Image) -> np.ndarray:
    """Jalankan inferensi dengan hard limit (NFR-01.2) -> 504 INFERENCE_TIMEOUT."""
    future = _get_executor().submit(classifier.predict_probabilities, image)
    try:
        return future.result(timeout=config.PREDICT_TIMEOUT_SECONDS)
    except concurrent.futures.TimeoutError as exc:
        future.cancel()
        raise HTTPException(
            status_code=504,
            detail={
                "code": "INFERENCE_TIMEOUT",
                "message": f"Prediksi melewati batas {int(config.PREDICT_TIMEOUT_SECONDS)} detik. Coba lagi.",
            },
        ) from exc


def _read_upload(file: UploadFile) -> Image.Image:
    """Validasi (ekstensi/MIME/ukuran) -> decode -> cek kualitas (FR-10.12)."""
    data = file.file.read(config.MAX_IMAGE_BYTES + 1)
    try:
        validate_upload(file.filename, file.content_type, len(data))
        image = decode_rgb(data)
        ensure_quality(image)
        return image
    except UnsupportedMediaTypeError as exc:
        # Bentuk terstruktur seperti 422/503/504 — backend membaca `detail.code`
        # dan menerjemahkannya ke MlException::UNSUPPORTED_MEDIA_TYPE
        # (MlPredictionNormalizer::exception). String polos membuat kode hilang.
        raise HTTPException(
            status_code=400,
            detail={
                "code": "UNSUPPORTED_MEDIA_TYPE",
                "message": str(exc) or "File harus berupa gambar JPG/PNG/WEBP",
            },
        ) from exc
    except ImageTooLargeError as exc:
        raise HTTPException(
            status_code=413,
            detail={"code": "PAYLOAD_TOO_LARGE", "message": str(exc)},
        ) from exc
    except ImageQualityError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": exc.code, "message": exc.message, "suggestions": exc.suggestions},
        ) from exc


def _top_predictions(probs: np.ndarray, limit: int = 3) -> list[dict]:
    labels = classifier.labels
    order = np.argsort(probs)[::-1][:limit]
    return [
        {
            "label": labels[i].label,
            "name": labels[i].name,
            "confidence": round(float(probs[i]) * 100, 1),
        }
        for i in order
    ]


def _predict_one(image: Image.Image, want_gradcam: bool, started_at: float) -> dict:
    """Prediksi satu gambar -> payload sesuai API.md §10.1."""
    _ensure_loaded()

    probs = _run_with_timeout(image)
    top = _top_predictions(probs)
    best_index = int(np.argmax(probs))
    best = classifier.labels[best_index]
    confidence = round(float(probs[best_index]) * 100, 1)

    gradcam_b64 = None
    if want_gradcam and config.GRADCAM_ENABLED:
        gradcam_b64 = generate_gradcam_b64(
            classifier.model, classifier.processor, image, best_index, classifier.device
        )

    return {
        "disease": {
            "label": best.label,
            "name": best.name,
            "slug": best.slug,
            "severity": best.severity,
        },
        "confidence": confidence,
        "top_predictions": top,
        "low_confidence_warning": confidence < config.LOW_CONFIDENCE_THRESHOLD,
        "gradcam_image": gradcam_b64,
        "processing_time_ms": int((time.time() - started_at) * 1000),
        "model_version": config.MODEL_VERSION,
    }


def _aggregate(results: list[dict], all_probs: list[np.ndarray]) -> dict:
    """Rata-rata tertimbang probabilitas antar foto; foto terbaik jadi referensi (FR-10.5)."""
    stacked = np.stack(all_probs)
    weights = np.array([p.max() for p in stacked])  # makin yakin makin berat
    mean_probs = np.average(stacked, axis=0, weights=weights)
    labels = classifier.labels
    best_index = int(np.argmax(mean_probs))
    best = labels[best_index]

    return {
        "disease": {"label": best.label, "name": best.name, "slug": best.slug, "severity": best.severity},
        "confidence": round(float(mean_probs[best_index]) * 100, 1),
        "method": "weighted_average",
        "best_photo_index": int(np.argmax([r["confidence"] for r in results])),
    }


# ──────────────────────────────────────────────────────────────────── endpoints

@app.get("/health", tags=["ops"])
def health(ready: bool = False) -> dict:
    """FR-10.2 — health check. Selalu 200 supaya liveness tidak membunuh container
    yang sedang memuat model.

    `GET /health?ready=1` -> 200 hanya bila model sudah siap, selain itu 503.
    Pakai ini sebagai readiness probe Docker/Kubernetes.
    """
    if classifier.is_loaded:
        status = "healthy"
    elif classifier.is_loading:
        status = "starting"
    else:
        status = "degraded"

    body = {
        "status": status,
        "model_version": config.MODEL_VERSION,
        "model_format": classifier.model_format,
        "uptime_seconds": int(time.time() - START_TIME),
        "model_loaded": classifier.is_loaded,
    }

    if ready and not classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail={"code": "MODEL_NOT_LOADED", "state": classifier.state, "message": classifier.error or body["status"]},
        )

    return body


@app.get("/model/info", tags=["ops"])
def model_info() -> dict:
    """FR-10.9 / FR-10.10 — metadata model aktif + input spec + label."""
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail={"code": "MODEL_NOT_LOADED", "message": classifier.error or "Model belum dimuat"},
        )
    return {
        "model_name": config.BASE_MODEL_ID,
        "architecture": "Vision Transformer (ViT)",
        "task": "Image Classification",
        "params": sum(p.numel() for p in classifier.model.parameters()),
        "input": "Gambar RGB 224×224",
        "output": f"Skor untuk {len(classifier.labels)} kelas (setelah fine-tuning)",
        "pretrained_on": "ImageNet-21k (14 juta gambar, 21k kelas)",
        "fine_tuned_on": "ImageNet-1k (1.28 juta, 1000 kelas)",
        "paper": "An Image is Worth 16x16 Words (Dosovitskiy et al., 2020)",
        "model_version": config.MODEL_VERSION,
        "model_format": classifier.model_format,
        "source_path": classifier.source_path,
        "input_spec": classifier.input_spec(),
        "labels": [lb.label for lb in classifier.labels],
        "loaded_at": classifier.loaded_at,
    }


@app.get("/diseases", tags=["labels"])
def diseases() -> dict:
    """FR-10.3 — label penyakit yang didukung model aktif."""
    try:
        labels = classifier.labels or load_labels()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"LABEL_MAP_ERROR: {exc}") from exc
    return {
        "model_version": config.MODEL_VERSION,
        "labels": [lb.as_dict() for lb in labels],
    }


@app.post("/predict", tags=["predict"])
def predict(
    image: UploadFile = File(..., description="1 foto ikan (JPG/PNG/WEBP, ≤10 MB)"),
    gradcam: bool = Form(False),  # FR-10.8
    fish_type: str | None = Form(None),  # hint opsional
) -> dict:
    """FR-10.1 — prediksi penyakit dari 1 gambar (multipart/form-data)."""
    started_at = time.time()
    img = _read_upload(image)
    logger.info("predict: file=%s fish_type=%s size=%s", image.filename, fish_type, img.size)
    return _predict_one(img, gradcam, started_at)


@app.post("/predict/batch", tags=["predict"])
def predict_batch(
    images: list[UploadFile] = File(..., description="2-5 foto ikan (JPG/PNG/WEBP, ≤10 MB/file)"),
    gradcam: bool = Form(False),
) -> dict:
    """FR-10.5 — batch inference; hasil dirata-ratakan (weighted_average)."""
    started_at = time.time()
    if not 1 <= len(images) <= config.MAX_BATCH_IMAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Jumlah gambar harus 1-{config.MAX_BATCH_IMAGES} (diterima: {len(images)})",
        )

    results: list[dict] = []
    all_probs: list[np.ndarray] = []
    decoded: list[Image.Image] = []
    for upload in images:
        img = _read_upload(upload)
        decoded.append(img)
        _ensure_loaded()
        probs = _run_with_timeout(img)
        all_probs.append(probs)
        best_index = int(np.argmax(probs))
        best = classifier.labels[best_index]
        confidence = round(float(probs[best_index]) * 100, 1)
        results.append(
            {
                "disease": {
                    "label": best.label, "name": best.name, "slug": best.slug, "severity": best.severity
                },
                "confidence": confidence,
                "top_predictions": _top_predictions(probs),
                "low_confidence_warning": confidence < config.LOW_CONFIDENCE_THRESHOLD,
                # GradCAM dihitung sekali untuk foto terbaik saja (lihat bawah):
                # menghitungnya per foto membuat 5 foto × ~10 s melampaui hard limit 30 s.
                "gradcam_image": None,
                "processing_time_ms": int((time.time() - started_at) * 1000),
                "model_version": config.MODEL_VERSION,
            }
        )

    aggregated = _aggregate(results, all_probs)

    if gradcam and config.GRADCAM_ENABLED and decoded:
        best_i = min(max(int(aggregated.get("best_photo_index", 0)), 0), len(decoded) - 1)
        # Kelas yang dibangkitkan GradCAM = kelas hasil agregasi, bukan kelas
        # per foto — supaya heatmap menjelaskan keputusan akhir.
        agg_label = (aggregated.get("disease") or {}).get("label")
        target = next(
            (i for i, lb in enumerate(classifier.labels) if lb.label == agg_label),
            int(np.argmax(all_probs[best_i])),
        )
        results[best_i]["gradcam_image"] = generate_gradcam_b64(
            classifier.model, classifier.processor, decoded[best_i], target, classifier.device
        )

    return {
        "results": results,
        "aggregated": aggregated,
        "processing_time_ms": int((time.time() - started_at) * 1000),
    }


@app.exception_handler(ModelNotLoadedError)
async def _model_not_loaded_handler(_, exc: ModelNotLoadedError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"detail": {"code": "MODEL_NOT_LOADED", "message": str(exc)}},
    )
