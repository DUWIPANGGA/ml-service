"""Entrypoint produksi/dev — `python serve.py`.

Host, port, worker, dan log level semuanya diambil dari `.env` (lihat
`app/config.py`) supaya Docker, systemd, dan laptop developer menjalankan
service yang persis sama. Tidak ada konfigurasi tersembunyi di CLI.

    python serve.py                 # baca .env
    HOST=0.0.0.0 PORT=8009 python serve.py

Menjalankan uvicorn langsung (bukan gunicorn) karena lifespan & model yang di-
muat di background thread sudah menangani kebutuhan kita, dan menambah process
manager hanya menggandakan pemakaian RAM (tiap worker memuat model 327 MB).
"""

from __future__ import annotations

import uvicorn

from app import config


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        workers=config.WORKERS,
        log_level=config.LOG_LEVEL.lower(),
        timeout_keep_alive=config.TIMEOUT_KEEP_ALIVE,
        limit_concurrency=config.LIMIT_CONCURRENCY,
        access_log=True,
    )


if __name__ == "__main__":
    main()
