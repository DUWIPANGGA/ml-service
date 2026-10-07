"""Smoke test end-to-end ke server ML — verifikasi deploy.

Dua mode:

    python scripts/smoke_server.py
        Jalankan server sendiri (serve.py), tunggu model siap, uji semua
        endpoint, lalu berhenti. Paling cocok sebagai langkah verifikasi
        sesudah `docker compose up` / `systemctl start ml-service`.

    python scripts/smoke_server.py --url http://127.0.0.1:8009
        Uji server yang SUDAH jalan (mis. di mesin lain / container).

Opsi:
    --timeout 240     detik tunggu model siap (cold start ±100 dtk)
    --port 8009       port untuk mode lokal
    --gradcam         ikut uji GradCAM (tambahan ±10 dtk; default dilewati)

Exit code 0 = semua pemeriksaan lulus.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

# Field kontrak API.md §10.1 — kalau salah satu hilang, integrasi Laravel pecah.
PREDICT_KEYS = {
    "disease",
    "confidence",
    "top_predictions",
    "low_confidence_warning",
    "gradcam_image",
    "processing_time_ms",
    "model_version",
}
DISEASE_KEYS = {"label", "name", "slug", "severity"}


class Checker:
    def __init__(self) -> None:
        self.failed: list[str] = []

    def check(self, ok: bool, label: str, extra: str = "") -> bool:
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {label}" + (f" — {extra}" if extra else ""))
        if not ok:
            self.failed.append(label)
        return ok


def make_image(seed: int = 7) -> bytes:
    """Foto uji sintetis: cukup tajam & terang agar lolos deteksi kualitas (FR-10.12)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    arr = rng.integers(60, 200, size=(224, 224, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, "RGB").save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def payload(res: httpx.Response) -> Any:
    try:
        return res.json()
    except Exception:  # noqa: BLE001
        return None


def run_checks(base: str, timeout: float, with_gradcam: bool) -> Checker:
    c = Checker()
    with httpx.Client(base_url=base, timeout=60.0) as client:
        # ── 1. /health liveness ────────────────────────────────────────────
        print("\n1) /health (liveness)")
        try:
            res = client.get("/health")
        except Exception as exc:  # noqa: BLE001
            c.check(False, "server menjawab", str(exc))
            return c

        c.check(res.status_code == 200, "GET /health -> 200", f"got {res.status_code}")
        body = payload(res) or {}
        for key in ("status", "model_version", "model_format", "uptime_seconds"):
            c.check(key in body, f"field `{key}` ada")

        # ── 2. tunggu model siap ───────────────────────────────────────────
        print(f"\n2) menunggu model siap (maks {timeout:.0f} dtk)")
        deadline = time.time() + timeout
        started_at = time.time()
        status = body.get("status")
        while status != "healthy" and time.time() < deadline:
            time.sleep(3)
            try:
                r = client.get("/health")
                status = (payload(r) or {}).get("status")
                print(f"    status={status}  t={time.time() - started_at:5.1f}s")
            except Exception:  # noqa: BLE001
                print("    (server belum menjawab)")
        ready_ok = c.check(
            status == "healthy", "model berstatus `healthy`", f"akhir: {status}"
        )
        if not ready_ok:
            print("    -> berhenti di sini; perbaiki load model dulu (lihat log).")
            return c

        # ── 3. readiness probe ─────────────────────────────────────────────
        print("\n3) /health?ready=1 (readiness)")
        res = client.get("/health", params={"ready": "1"})
        c.check(res.status_code == 200, "GET /health?ready=1 -> 200", str(res.status_code))

        # ── 4. /diseases — FR-10.11 ────────────────────────────────────────
        print("\n4) /diseases (11 label + disease_id)")
        res = client.get("/diseases")
        c.check(res.status_code == 200, "GET /diseases -> 200", str(res.status_code))
        data = payload(res) or {}
        labels = data.get("labels", []) if isinstance(data, dict) else data
        c.check(len(labels) == 11, "jumlah label = 11", f"got {len(labels)}")
        if labels:
            c.check(DISEASE_KEYS <= set(labels[0]), "field label lengkap", str(sorted(labels[0])))
            ids = [x.get("disease_id") for x in labels]
            c.check(
                ids == list(range(1, 12)),
                "disease_id = 1..11 urut sesuai indeks model",
                str(ids),
            )

        # ── 5. /model/info ─────────────────────────────────────────────────
        print("\n5) /model/info (metadata)")
        res = client.get("/model/info")
        c.check(res.status_code == 200, "GET /model/info -> 200", str(res.status_code))
        info = payload(res) or {}
        c.check("model_name" in info, "field `model_name` ada", str(info.get("model_name")))
        c.check(bool(info.get("input_spec")), "field `input_spec` ada")
        c.check(len(info.get("labels") or []) == 11, "metadata memuat 11 label")

        # ── 6. /predict ────────────────────────────────────────────────────
        print("\n6) POST /predict (API.md §10.1)")
        files = {"image": ("smoke_0.jpg", make_image(7), "image/jpeg")}
        data = {"fish_type": "Lele", "gradcam": "false"}
        res = client.post("/predict", files=files, data=data)
        c.check(res.status_code == 200, "POST /predict -> 200", str(res.status_code))
        pred = payload(res) or {}
        if res.status_code == 200:
            missing = PREDICT_KEYS - set(pred)
            c.check(not missing, "field kontrak lengkap", f"missing={sorted(missing)}")
            conf = pred.get("confidence")
            c.check(
                isinstance(conf, (int, float)) and 0 <= conf <= 100,
                "confidence rentang 0..100",
                str(conf),
            )
            dis = pred.get("disease") or {}
            c.check(DISEASE_KEYS <= set(dis), "field disease lengkap", str(sorted(dis)))
            tops = pred.get("top_predictions") or []
            c.check(len(tops) >= 1, "top_predictions terisi", f"{len(tops)} entri")
            c.check(
                pred.get("gradcam_image") is None,
                "gradcam_image null bila diminta false",
            )
            print(
                f"      -> {dis.get('name')} ({conf}%) "
                f"severity={dis.get('severity')} slug={dis.get('slug')}"
            )
        else:
            print(f"      -> {json.dumps(pred, ensure_ascii=False)[:300]}")

        # ── 7. /predict/batch ──────────────────────────────────────────────
        # Kontrak API.md §10.2:
        #   { "results": [<predict>, ...], "aggregated": {disease, confidence, method} }
        print("\n7) POST /predict/batch (2 gambar)")
        files = [
            ("images", ("smoke_0.jpg", make_image(7), "image/jpeg")),
            ("images", ("smoke_1.jpg", make_image(11), "image/jpeg")),
        ]
        res = client.post("/predict/batch", files=files, data={"gradcam": "false"})
        c.check(res.status_code == 200, "POST /predict/batch -> 200", str(res.status_code))
        batch = payload(res) or {}
        if res.status_code == 200:
            results = batch.get("results") or []
            agg = batch.get("aggregated") or {}
            c.check(len(results) == 2, "results berisi 2 entri", f"got {len(results)}")
            if results:
                missing = PREDICT_KEYS - set(results[0])
                c.check(not missing, "bentuk tiap result = bentuk /predict",
                        f"missing={sorted(missing)}")
            c.check(agg.get("method") == "weighted_average",
                    "aggregated.method == weighted_average", str(agg.get("method")))
            c.check(DISEASE_KEYS <= set(agg.get("disease") or {}),
                    "aggregated.disease lengkap",
                    str(sorted(agg.get("disease") or {})))
            print(
                f"      -> agregat {((agg.get('disease') or {}).get('name'))} "
                f"({agg.get('confidence')}%)"
            )
        else:
            print(f"      -> {json.dumps(batch, ensure_ascii=False)[:300]}")

        # ── 8. validasi media type ─────────────────────────────────────────
        print("\n8) validasi upload")
        res = client.post(
            "/predict",
            files={"image": ("malware.txt", b"bukan gambar", "text/plain")},
        )
        c.check(
            res.status_code == 400
            and (payload(res) or {}).get("detail", {}).get("code") == "UNSUPPORTED_MEDIA_TYPE",
            "file non-gambar -> 400 UNSUPPORTED_MEDIA_TYPE",
            str(res.status_code),
        )

        # ── 9. GradCAM (opsional) ──────────────────────────────────────────
        if with_gradcam:
            print("\n9) POST /predict dengan gradcam=true (±10 dtk)")
            res = client.post(
                "/predict",
                files={"image": ("smoke_gc.jpg", make_image(7), "image/jpeg")},
                data={"gradcam": "true"},
                timeout=90.0,
            )
            c.check(res.status_code == 200, "prediksi + gradcam -> 200", str(res.status_code))
            gc = (payload(res) or {}).get("gradcam_image")
            c.check(gc is None or isinstance(gc, str), "gradcam_image string atau null", type(gc).__name__)
            if isinstance(gc, str):
                c.check(gc.startswith("data:image") or gc.startswith("http"), "format gambar masuk akal", gc[:60])

    return c


def spawn_server(port: int) -> subprocess.Popen:
    env = dict(os.environ)
    env.update({"HOST": "127.0.0.1", "PORT": str(port)})
    log = io.open(ROOT / "smoke_server.log", "w", encoding="utf-8")
    print(f"  start: {sys.executable} serve.py (log -> smoke_server.log)")
    return subprocess.Popen(
        [sys.executable, "serve.py"],
        cwd=str(ROOT),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test end-to-end ML service")
    ap.add_argument("--url", default="", help="uji server yang sudah jalan")
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--timeout", type=float, default=240.0, help="detik tunggu model siap")
    ap.add_argument("--gradcam", action="store_true", help="ikut uji GradCAM")
    args = ap.parse_args()

    base = args.url or f"http://127.0.0.1:{args.port}"
    print(f"Target: {base}")

    proc: subprocess.Popen | None = None
    if not args.url:
        proc = spawn_server(args.port)
        # Beri waktu bind socket; model sendirinya dipuat di background.
        time.sleep(3)

    try:
        checker = run_checks(base, args.timeout, args.gradcam)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    print()
    if checker.failed:
        print(f"GAGAL — {len(checker.failed)} pemeriksaan:")
        for f in checker.failed:
            print(f"  - {f}")
        return 1
    print("SEMUA PEMERIKSAAN LULUS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
