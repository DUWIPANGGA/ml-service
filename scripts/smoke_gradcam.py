"""Smoke test GradCAM: apakah layer terdeteksi dan PNG keluar?"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from app.gradcam import _find_target_layers, generate_gradcam_b64
from app.model import classifier


def main() -> int:
    classifier.load()
    layers = _find_target_layers(classifier.model)
    print(f"[{'OK' if layers else 'FAIL'}] target_layers={len(layers)} "
          f"-> {[type(x).__name__ for x in layers]}")

    if not layers:
        return 1

    sample = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    img = Image.open(sample).convert("RGB") if sample and sample.exists() else None
    if img is None:
        import numpy as np
        rng = np.random.default_rng(7)
        img = Image.fromarray(rng.integers(60, 200, size=(224, 224, 3), dtype=np.uint8), "RGB")

    probs = classifier.predict_probabilities(img)
    best = int(probs.argmax())
    print(f"[..] kelas pemenang #{best} = {classifier.labels[best].name}")

    t0 = time.time()
    b64 = generate_gradcam_b64(classifier.model, classifier.processor, img, best, classifier.device)
    ms = (time.time() - t0) * 1000

    if not b64:
        print(f"[FAIL] gradcam=None ({ms:.0f} ms)")
        return 1

    print(f"[OK] gradcam {len(b64)} karakter, {ms:.0f} ms, prefix={b64[:30]}...")

    import base64
    out = Path(__file__).resolve().parent / "gradcam_out.png"
    out.write_bytes(base64.b64decode(b64.split(",", 1)[1]))
    print(f"[OK] disimpan: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
