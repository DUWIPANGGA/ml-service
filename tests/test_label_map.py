"""Guard urutan label_map.json ↔ config.json model (FR-10.11)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.labels import load_labels
from app.model import DiseaseClassifier

MODEL_CONFIG = Path(__file__).resolve().parent.parent / "models" / "config.json"


def _write_label_map(tmp_path: Path, labels: list[dict]) -> Path:
    path = tmp_path / "label_map.json"
    path.write_text(json.dumps({"model_version": "test", "labels": labels}), encoding="utf-8")
    return path


def _base_labels() -> list[dict]:
    """Ambil label_map.json asli lalu biarkan kita ubah urutannya."""
    raw = json.loads((Path(__file__).resolve().parent.parent / "label_map.json").read_text(encoding="utf-8"))
    return [dict(row) for row in raw["labels"]]


def test_label_map_matches_model_config():
    """label_map.json resmi HARUS selaras dengan config.json model."""
    labels = load_labels()
    model_id2label = {int(k): v for k, v in json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))["id2label"].items()}

    assert len(labels) == len(model_id2label) == 11
    for lb in labels:
        assert lb.model_label == model_id2label[lb.index], (
            f"index {lb.index}: label_map={lb.model_label!r} vs model={model_id2label[lb.index]!r}"
        )


def test_verify_passes_when_order_matches():
    labels = load_labels()
    id2label = {lb.index: lb.model_label for lb in labels}
    DiseaseClassifier._verify_label_order(labels, id2label)  # tidak boleh melempar


def test_verify_ignores_rows_without_model_label():
    """Baris tanpa model_label di-skip (kompatibel label_map lama)."""
    from app.labels import Label

    labels = [Label(index=0, label="x", name="X", slug="x", severity="ringan", disease_id=None, model_label=None)]
    DiseaseClassifier._verify_label_order(labels, {0: "Whatever"})


def test_verify_rejects_wrong_order(tmp_path):
    """Tukar dua label → harus gagal dengan pesan yang menyebut index."""
    rows = _base_labels()
    rows[0], rows[1] = rows[1], rows[0]
    labels = load_labels(_write_label_map(tmp_path, rows))

    model_id2label = {int(k): v for k, v in json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))["id2label"].items()}

    with pytest.raises(ValueError, match="index 0"):
        DiseaseClassifier._verify_label_order(labels, model_id2label)
