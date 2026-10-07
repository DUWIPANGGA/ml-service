#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Ngingu ML Service — jalankan (Linux / macOS)
#
#   ./run.sh              sekali jalan: buat venv + install + start
#   HOST=0.0.0.0 ./run.sh buka ke jaringan lokal
#
# Ctrl+C untuk berhenti.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PY="${PYTHON:-python3}"

if [ ! -d .venv ]; then
    echo "[1/3] Membuat virtualenv ..."
    "$PY" -m venv .venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

# Penanda agar instalasi hanya sekali — selanjutnya start langsung.
if [ ! -f .venv/.deps-installed ]; then
    echo "[2/3] Memasang dependensi (torch + transformers, bisa 5-10 menit) ..."
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    touch .venv/.deps-installed
else
    echo "[2/3] Dependensi sudah terpasang."
fi

if [ ! -f .env ]; then
    cp .env.example .env
    echo "      .env dibuat dari .env.example"
fi

echo "[3/3] Menjalankan uvicorn (model dimuat di background ±100 detik) ..."
echo "      health  : http://127.0.0.1:${PORT:-8009}/health"
echo "      docs    : http://127.0.0.1:${PORT:-8009}/docs"
echo

exec python serve.py
