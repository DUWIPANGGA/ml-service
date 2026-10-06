"""Pemuat model ViT untuk deteksi penyakit ikan (FR-10.6, FR-10.9, FR-10.10).

Dua bentuk checkpoint yang didukung:
  1. Folder `save_pretrained` (config.json + model.safetensors)
  2. Satu file state_dict tunggal (.safetensors / .pt / .bin)

Model dimuat sekali saat startup lalu dipakai inference `eval()` tanpa grad.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .labels import Label

logger = logging.getLogger("ml.model")


class ModelNotLoadedError(RuntimeError):
    """Dipakai saat inference diminta tapi model belum/bisa dimuat."""


class DiseaseClassifier:
    """Wrapper ViT fine-tuned: preprocess -> logits -> probabilitas 11 kelas."""

    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = "cpu"
        self.labels: list[Label] = []
        self.source_path: str = ""
        self.model_format: str = "unknown"
        self.loaded_at: str | None = None
        self.error: str | None = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    # ------------------------------------------------------------- muat model
    def load(self) -> None:
        """Muat model + processor. Melempar RuntimeError bila gagal."""
        import torch
        from transformers import ViTForImageClassification, ViTImageProcessor

        self.labels = self._load_labels()
        n_classes = len(self.labels)
        id2label = {lb.index: lb.name for lb in self.labels}
        label2id = {lb.name: lb.index for lb in self.labels}

        path = config.MODEL_PATH
        self.source_path = str(path)

        self.device = self._resolve_device(torch)
        logger.info("Memuat ViT dari %s (kelas=%d, device=%s)", path, n_classes, self.device)

        if path.is_dir() and (path / "config.json").exists():
            # Bentuk 1: folder save_pretrained hasil fine-tuning.
            self.model_format = "safetensors" if (path / "model.safetensors").exists() else "pytorch"

            # Baca id2label ASLI dari disk dulu — dari_pretrained() akan menimpa
            # config dengan id2label dari label_map, jadi guard harus pakai yang asli.
            import json as _json

            original_id2label = {
                int(k): v for k, v in _json.loads((path / "config.json").read_text(encoding="utf-8")).get(
                    "id2label", {}
                ).items()
            }
            self._verify_label_order(self.labels, original_id2label)

            self.model = ViTForImageClassification.from_pretrained(
                str(path),
                num_labels=n_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
            self.processor = ViTImageProcessor.from_pretrained(str(path))
        elif path.is_file():
            # Bentuk 2: file state_dict tunggal di atas arsitektur base.
            self.model_format = path.suffix.lstrip(".").replace("bin", "pytorch")
            self.model = ViTForImageClassification.from_pretrained(
                config.BASE_MODEL_ID,
                num_labels=n_classes,
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            )
            self._load_state_dict(path)
            self.processor = ViTImageProcessor.from_pretrained(config.BASE_MODEL_ID)
        else:
            raise FileNotFoundError(
                f"MODEL_PATH tidak ditemukan: {path}. "
                "Letakkan checkpoint fine-tuned di ml-service/models/vit-fish-11/ "
                "(folder save_pretrained atau file .safetensors/.pt)."
            )

        self.model.to(self.device)
        self.model.eval()

        from datetime import datetime, timezone

        self.loaded_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.error = None
        n_params = sum(p.numel() for p in self.model.parameters())
        logger.info("Model siap: %s parameter, format=%s", f"{n_params:,}", self.model_format)

    def _load_labels(self) -> list[Label]:
        from .labels import load_labels

        return load_labels()

    @staticmethod
    def _verify_label_order(labels: list[Label], id2label: dict) -> None:
        """Guard FR-10.11: urutan label_map.json HARUS sama dengan config model.

        Kalau beda, prediksi akan salah mapping diam-diam — lebih baik gagal saat
        startup dengan pesan jelas.
        """
        mismatches = [
            f"  index {lb.index}: model={id2label.get(lb.index)!r} vs label_map={lb.model_label!r}"
            for lb in labels
            if lb.model_label is not None
            and id2label.get(lb.index) is not None
            and id2label[lb.index] != lb.model_label
        ]
        if mismatches:
            raise ValueError(
                "Urutan label_map.json tidak cocok dengan config.json model — "
                "prediksi akan salah mapping. Selaraskan urutannya:\n" + "\n".join(mismatches)
            )

    def _load_state_dict(self, path: Path) -> None:
        import torch

        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(path))
        else:
            state = torch.load(str(path), map_location="cpu", weights_only=True)

        # Bungkusan umum: beberapa training script menyimpan kunci "state_dict".
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("Missing keys (%d): %s...", len(missing), missing[:5])
        if unexpected:
            logger.warning("Unexpected keys (%d): %s...", len(unexpected), unexpected[:5])

    @staticmethod
    def _resolve_device(torch) -> str:
        requested = config.DEVICE
        if requested != "auto":
            return requested
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ------------------------------------------------------------- inference
    def predict_probabilities(self, image: Image.Image) -> np.ndarray:
        """Kembalikan vektor probabilitas softmax sepanjang jumlah label."""
        if not self.is_loaded:
            raise ModelNotLoadedError(self.error or "Model belum dimuat")

        import torch

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        with self._lock:  # inferensi thread-safe di threadpool FastAPI
            with torch.no_grad():
                logits = self.model(pixel_values=pixel_values).logits
                probs = torch.softmax(logits, dim=-1)[0]

        return probs.cpu().numpy()

    def input_spec(self) -> dict:
        if self.processor is None:
            return {
                "height": 224,
                "width": 224,
                "channels": 3,
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            }

        # transformers >=5 memakai SizeDict (mirip dict, tapi bukan dict) —
        # jadi jangan cek isinstance(..., dict) saja.
        size = getattr(self.processor, "size", None)
        if hasattr(size, "keys"):
            size = dict(size)
        if isinstance(size, int):
            size = {"height": size, "width": size}
        if not isinstance(size, dict):
            size = {}

        height = int(size.get("height", size.get("shortest_edge", 224)))
        width = int(size.get("width", size.get("longest_edge", height)))
        mean = list(getattr(self.processor, "image_mean", [0.485, 0.456, 0.406]))
        std = list(getattr(self.processor, "image_std", [0.229, 0.224, 0.225]))
        return {
            "height": height,
            "width": width,
            "channels": 3,
            "normalization": {"mean": mean, "std": std},
        }


# Singleton — dimuat saat startup oleh lifespan di main.py.
classifier = DiseaseClassifier()
