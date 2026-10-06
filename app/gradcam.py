"""GradCAM untuk ViT (FR-10.8) — opsional; gagal -> null, prediksi tetap jalan."""

from __future__ import annotations

import base64
import io
import logging
import math

import numpy as np
from PIL import Image

logger = logging.getLogger("ml.gradcam")


def _vit_reshape_transform(tensor):
    """(B, 197, C) -> (B, C, 14, 14): buang CLS token, susun patch jadi grid."""
    if tensor.ndim != 3:
        return tensor
    n_tokens = tensor.shape[1] - 1  # tanpa CLS
    side = math.isqrt(n_tokens)
    if side * side != n_tokens:
        return tensor[:, 1:, :].transpose(1, 2).unsqueeze(-1)
    grid = tensor[:, 1:, :].reshape(tensor.shape[0], side, side, tensor.shape[-1])
    return grid.permute(0, 3, 1, 2)


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


def generate_gradcam_b64(model, processor, image: Image.Image, class_index: int, device: str) -> str | None:
    """Kembalikan data URI PNG overlay GradCAM, atau None bila gagal/dinonaktifkan."""
    if model is None or processor is None:
        return None
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        import torch
    except ImportError as exc:
        logger.info("pytorch-grad-cam tidak tersedia (%s) — GradCAM dilewati", exc)
        return None

    target_layers = _find_target_layers(model)
    if not target_layers:
        return None

    try:
        inputs = processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        # Model ViT HF membawa head klasifikasi; GradCAM cukup pakai logits.
        class _LogitsOnly(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(pixel_values=x).logits

        wrapped = _LogitsOnly(model).to(device)
        with GradCAM(model=wrapped, target_layers=target_layers, reshape_transform=_vit_reshape_transform) as cam:
            grayscale = cam(input_tensor=pixel_values, targets=[ClassifierOutputTarget(class_index)])
        del wrapped

        mask = np.asarray(grayscale[0], dtype=np.float32)
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)

        base = image.convert("RGB").copy()
        overlay = Image.new("RGB", base.size, (255, 0, 0))
        base.paste(overlay, (0, 0), mask=mask_img)

        buf = io.BytesIO()
        base.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:  # noqa: BLE001 — GradCAN tidak boleh menggagalkan prediksi
        logger.warning("GradCAM gagal: %s", exc)
        return None
