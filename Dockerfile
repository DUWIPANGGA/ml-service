# syntax=docker/dockerfile:1
# ─────────────────────────────────────────────────────────────────────────────
# Ngingu ML Service — deteksi penyakit ikan (ViT 11 kelas)
#
#   docker build -t ngingu/ml-service .
#   docker run --rm -p 127.0.0.1:8009:8009 ngingu/ml-service
#
# Catatan: image ini belum pernah di-build di mesin pengembangan (Docker tidak
# tersedia di sana) — lihat README §Docker untuk checklist verifikasi.
# ─────────────────────────────────────────────────────────────────────────────

# Python 3.12: wheel torch/transformers tersedia luas dan teruji.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# curl          -> HEALTHCHECK
# libglib2.0-0  -> opencv (dipakai grad-cam)
# libgl1        -> libGL untuk opencv non-headless
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/ml-service

# torch + torchvision versi CPU dari index resmi PyTorch.
# Wheel PyPI default membawa runtime CUDA (±2 GB lebih besar) dan tidak terpakai
# di server tanpa GPU. Dependen (filelock, sympy, dst.) juga tersedia di index ini.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.4.0" "torchvision>=0.19.0"

# Sisanya. torch/torchvision sudah terpenuhi oleh layer di atas,
# jadi pip tidak akan menimpanya dengan wheel CUDA.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Kode + label map (urutan indeks model — jangan diubah tanpa cek FR-10.11).
COPY app/ ./app/
COPY serve.py label_map.json ./

# Checkpoint fine-tuned 327 MB. models/ memang tidak masuk git, jadi pastikan
# file-nya ada di build context. Alternatif: komentar baris ini dan mount
# volume (lihat docker-compose.yml).
COPY models/ ./models/

# Service ini TIDAK punya autentikasi — HOST=0.0.0.0 hanya aman di belakang
# jaringan internal / reverse proxy. HF_OFFLINE=true aman karena MODEL_PATH
# adalah folder save_pretrained lengkap (tidak perlu unduh dari Hugging Face).
# Pakai bentuk "1 file state_dict"? Set HF_OFFLINE=false.
ENV HOST=0.0.0.0 \
    PORT=8009 \
    WORKERS=1 \
    HF_OFFLINE=true \
    DEVICE=cpu

RUN useradd --system --create-home --uid 10001 mlsvc \
    && chown -R mlsvc:mlsvc /srv/ml-service
USER mlsvc

EXPOSE 8009

# /health selalu 200 (liveness). Model butuh ±100 detik untuk siap, jadi
# start_period diberi margin lebar agar cold start tidak dianggap container
# mati. readiness = /health?ready=1 (503 sampai model benar-benar siap).
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/health?ready=1" >/dev/null || exit 1

CMD ["python", "serve.py"]
