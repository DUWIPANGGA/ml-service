"""GradCAM untuk ViT (FR-10.8) — opsional; gagal -> null, prediksi tetap jalan.

Catatan implementasi (penting):
    Head klasifikasi ViT HF hanya membaca token CLS, sehingga **gradien terhadap
    token patch = 0 secara persis**. Kalau peta (patch x gradien) dipakai seperti
    ConvNet, hasilnya mask semua-nol -> overlay kosong dan app menampilkan foto
    asli sebagai "heatmap". Karena itu di sini bobot diambil dari gradien token
    CLS, lalu dikalikan dengan aktivasi token patch (CAM khas ViT).
"""

from __future__ import annotations

import base64
import io
import logging
import math
import threading

import numpy as np
from PIL import Image

logger = logging.getLogger("ml.gradcam")

# GradCAM memakai hook pada layer bersama + model.zero_grad(): serialkan antar
# request supaya dua request tidak saling menimpa aktivasi/gradien.
_lock = threading.Lock()


def _find_target_layers(model, count: int = 1):
    """Ambil layer encoder transformer terakhir sebagai sumber GradCAM.

    Struktur ViT berbeda antar versi transformers:
      4.x  → model.vit.encoder.layer   (ModuleList, ada pembungkus encoder)
      5.x  → model.vit.layers          (ModuleList langsung di ViTModel)
    Fallback terakhir: ModuleList terakhir pada DFS traversal model.
    """
    import torch.nn as nn

    owners = (getattr(getattr(model, "vit", None), "encoder", None), getattr(model, "vit", None), model)

    for owner in owners:
        if owner is None:
            continue
        for attr in ("layer", "layers"):
            layers = getattr(owner, attr, None)
            if isinstance(layers, nn.ModuleList) and len(layers) >= count:
                return list(layers)[-count:]

    module_lists = [m for m in model.modules() if isinstance(m, nn.ModuleList)]

    if module_lists and len(module_lists[-1]) >= count:
        return list(module_lists[-1])[-count:]

    logger.warning("Struktur ViT tidak dikenali untuk GradCAM")
    return []


def _capture_forward(layer):
    """Hook: simpan aktivasi layer (B, N, C) + pasang hook gradien pada tensornya."""
    captured: dict = {}

    def _hook(_module, _inputs, output):
        if getattr(output, "ndim", 0) != 3:  # bukan bentuk token sequence
            return
        captured["activation"] = output
        output.register_hook(lambda grad: captured.__setitem__("gradient", grad))

    return layer.register_forward_hook(_hook), captured


def _cls_patch_cam(image: Image.Image, activation, gradient):
    """Bangun peta CAM dari (gradien CLS × aktivasi patch) -> PIL 'L' seukuran foto."""
    act = activation.detach()[0].float().cpu()  # (N, C), N = 1 + s*s
    grad = gradient.detach()[0].float().cpu()  # (N, C)

    if act.ndim != 2 or act.shape != grad.shape:
        logger.warning("GradCAM: bentuk aktivasi/gradien tak dikenali %s / %s", tuple(act.shape), tuple(grad.shape))
        return None

    patches = act[1:]  # token patch tanpa CLS
    side = math.isqrt(patches.shape[0])
    if side == 0 or side * side != patches.shape[0]:
        logger.warning("GradCAM: jumlah token patch (%d) bukan grid persegi", patches.shape[0])
        return None

    # Kontribusi negatif bukan "bukti"; tanpa ReLU, cam - cam.min() akan
    # menggeser seluruh peta jadi positif dan seluruh foto tertimpa merah.
    import torch

    cam = torch.relu((patches * grad[0].unsqueeze(0)).sum(dim=1))  # (s*s,)
    peak = float(cam.max())
    if not math.isfinite(peak) or peak <= 1e-12:
        logger.warning("GradCAM: peta gradien datar (peak=%.3g) — overlay dilewati", peak)
        return None
    cam = cam / peak  # [0,1] — nol tetap nol, daerah tak relevan tetap transparan

    grid = cam.reshape(side, side).numpy()
    mask = Image.fromarray((grid * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    if int(np.asarray(mask).max()) == 0:
        logger.warning("GradCAM: mask hasil resize tetap nol — overlay dilewati")
        return None
    return mask


def generate_gradcam_b64(model, processor, image: Image.Image, class_index: int, device: str) -> str | None:
    """Kembalikan data URI PNG overlay GradCAM, atau None bila gagal/dinonaktifkan."""
    if model is None or processor is None:
        return None
    try:
        import torch
    except ImportError as exc:
        logger.info("torch tidak tersedia (%s) — GradCAM dilewati", exc)
        return None

    target_layers = _find_target_layers(model)
    if not target_layers:
        return None

    try:
        with _lock:
            layer = target_layers[-1]
            handle, captured = _capture_forward(layer)
            was_training = model.training
            try:
                model.eval()
                pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
                with torch.enable_grad():  # tetap hitung walau dipanggil di dalam no_grad
                    model.zero_grad(set_to_none=True)
                    logits = model(pixel_values=pixel_values).logits
                    logits[0, int(class_index)].backward()
            finally:
                handle.remove()
                model.zero_grad(set_to_none=True)
                model.train(was_training)

            activation = captured.get("activation")
            gradient = captured.get("gradient")
            if activation is None or gradient is None:
                logger.warning("GradCAM: gradien tidak tertangkap dari %s", type(layer).__name__)
                return None

            mask_img = _cls_patch_cam(image, activation, gradient)
            if mask_img is None:
                return None

        base = image.convert("RGB").copy()
        overlay = Image.new("RGB", base.size, (255, 0, 0))
        base.paste(overlay, (0, 0), mask=mask_img)

        buf = io.BytesIO()
        base.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:  # noqa: BLE001 — GradCAM tidak boleh menggagalkan prediksi
        logger.warning("GradCAM gagal: %s", exc)
        return None
