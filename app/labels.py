"""Label map: 11 kelas penyakit ikan -> metadata untuk mapping output model (FR-10.11)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass(frozen=True)
class Label:
    index: int
    label: str
    name: str
    slug: str
    severity: str
    disease_id: int | None
    # Label mentah di config.json model — dipakai guard urutan (FR-10.11).
    model_label: str | None = None

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "name": self.name,
            "slug": self.slug,
            "severity": self.severity,
            "disease_id": self.disease_id,
        }


def load_labels(path: Path | None = None) -> list[Label]:
    """Muat label_map.json. Urutan array = urutan indeks output model."""
    path = path or config.LABEL_MAP_PATH
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw["labels"] if isinstance(raw, dict) else raw

    labels = [
        Label(
            index=i,
            label=str(row["label"]),
            name=str(row["name"]),
            slug=str(row.get("slug") or row["label"]),
            severity=str(row.get("severity") or "sedang"),
            disease_id=row.get("disease_id"),
            model_label=row.get("model_label"),
        )
        for i, row in enumerate(rows)
    ]
    if not labels:
        raise ValueError(f"label_map kosong: {path}")
    return labels
