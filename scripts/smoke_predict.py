"""Smoke test: load model asli + prediksi 1 gambar (tanpa server)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from app.model import classifier


def main() -> int:
    t0 = time.time()
    classifier.load()
    print(f"[OK] model dimuat dalam {time.time() - t0:.1f}s")
    print(f"     device={classifier.device} format={classifier.model_format}")
    print(f"     kelas={len(classifier.labels)} path={classifier.source_path}")
    print(f"     input_spec={classifier.input_spec()}")

    # Buat gambar uji: separuh berisik (bukan gambar ikan asli — hanya cek pipeline)
    rng = np.random.default_rng(7)
    arr = rng.integers(60, 200, size=(224, 224, 3), dtype=np.uint8)

    # Kalau ada gambar uji pakai itu
    sample = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if sample and sample.exists():
        img = Image.open(sample).convert("RGB")
        print(f"[..] prediksi {sample} ({img.size[0]}x{img.size[1]})")
    else:
        img = Image.fromarray(arr, "RGB")
        print("[..] prediksi gambar acak (bukan ikan asli — hanya cek pipeline)")

    t1 = time.time()
    probs = classifier.predict_probabilities(img)
    ms = (time.time() - t1) * 1000

    order = np.argsort(probs)[::-1][:5]
    print(f"[OK] inferensi {ms:.0f} ms, total prob sum={probs.sum():.4f}")
    for i in order:
        lb = classifier.labels[int(i)]
        print(f"     {float(probs[int(i)]) * 100:6.2f}%  {lb.name}  ({lb.model_label})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
