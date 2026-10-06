"""Preprocessing & validasi gambar (FR-10.4, FR-10.10, FR-10.12).

- Validasi format file (JPG/PNG/WEBP) & ukuran (≤10 MB)
- Deteksi kualitas: blur (variance-of-Laplacian) & brightness
- Decode ke RGB — resize/normalize diserahkan ke ViTImageProcessor model
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from . import config


class UnsupportedMediaTypeError(ValueError):
    """Bukan JPG/PNG/WEBP -> 400 UNSUPPORTED_MEDIA_TYPE."""


class ImageTooLargeError(ValueError):
    """File > batas ukuran -> 413 PAYLOAD_TOO_LARGE."""


class ImageQualityError(ValueError):
    """Gambar blur/gelap/terlalu terang -> 422 IMAGE_QUALITY_LOW (FR-10.12)."""

    def __init__(self, code: str, message: str, suggestions: list[str]):
        super().__init__(message)
        self.code = code
        self.message = message
        self.suggestions = suggestions


@dataclass
class QualityReport:
    blur_variance: float
    brightness: float
    issues: list[str] = field(default_factory=list)


def validate_upload(filename: str | None, content_type: str | None, size: int) -> None:
    """Cek ekstensi/MIME/ukuran sebelum decode."""
    ext = (filename or "").rsplit(".", 1)[-1].lower() if filename and "." in filename else ""
    if f".{ext}" not in config.ALLOWED_EXTENSIONS:
        raise UnsupportedMediaTypeError("File harus berupa gambar JPG/PNG/WEBP")
    if content_type and content_type not in config.ALLOWED_CONTENT_TYPES:
        raise UnsupportedMediaTypeError("File harus berupa gambar JPG/PNG/WEBP")
    if size > config.MAX_IMAGE_BYTES:
        raise ImageTooLargeError(f"Ukuran gambar melebihi {config.MAX_IMAGE_BYTES // (1024 * 1024)} MB")


def decode_rgb(data: bytes) -> Image.Image:
    """Decode bytes -> gambar RGB. Gagal decode = bukan gambar valid."""
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise UnsupportedMediaTypeError("File bukan gambar yang valid") from exc

    if image.format and image.format.upper() not in {"JPEG", "PNG", "WEBP"}:
        raise UnsupportedMediaTypeError("File harus berupa gambar JPG/PNG/WEBP")

    # Auto-orient EXIF (foto HP) lalu paksa RGB untuk konsistensi input model.
    return ImageOps.exif_transpose(image).convert("RGB")


def analyze_quality(image: Image.Image) -> QualityReport:
    """Hitung blur (variance-of-Laplacian) & brightness; tandai bila di luar batas (FR-10.12)."""
    gray = np.asarray(image.convert("L"), dtype=np.float64)
    brightness = float(gray.mean())

    # Variance-of-Laplacian via kernel 3x3 (tanpa dependensi OpenCV).
    lap = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )
    blur_variance = float(lap.var())

    issues: list[str] = []
    if blur_variance < config.BLUR_MIN_VARIANCE:
        issues.append("blur")
    if brightness < config.DARK_MAX_BRIGHTNESS:
        issues.append("too_dark")
    elif brightness > config.BRIGHT_MIN_BRIGHTNESS:
        issues.append("too_bright")

    return QualityReport(blur_variance=blur_variance, brightness=brightness, issues=issues)


def ensure_quality(image: Image.Image) -> QualityReport:
    """Lempar ImageQualityError bila gambar tidak layak diprediksi."""
    report = analyze_quality(image)
    if not report.issues:
        return report

    suggestions: list[str] = []
    messages: list[str] = []
    if "blur" in report.issues:
        messages.append("Gambar blur / tidak fokus")
        suggestions += ["Fokuskan kamera pada ikan", "Bersihkan lensa kamera", "Tahan kamera stabil"]
    if "too_dark" in report.issues:
        messages.append("Gambar terlalu gelap")
        suggestions += ["Foto dengan pencahayaan lebih terang", "Hindari bayangan ke objek"]
    if "too_bright" in report.issues:
        messages.append("Gambar terlalu terang / overexposed")
        suggestions += ["Kurangi pencahayaan langsung", "Hindari kilatan flash dekat objek"]

    raise ImageQualityError(
        code="IMAGE_QUALITY_LOW",
        message="; ".join(messages),
        suggestions=suggestions,
    )
