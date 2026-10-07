# 🐟 ML Service — Deteksi Penyakit Ikan

FastAPI service untuk klasifikasi penyakit ikan dari foto menggunakan
**`google/vit-base-patch16-224`** (Vision Transformer) yang sudah di-fine-tune ke
**11 kelas penyakit ikan**.

> Kontrak: [API.md §10](../API.md) · SRS FR-10 · Dipanggil **server-to-server** oleh
> Laravel (`MlClient`) — bukan langsung dari mobile app.

| | |
|---|---|
| **Endpoint** | 5: `/health`, `/model/info`, `/diseases`, `/predict`, `/predict/batch` |
| **Port** | **8009** (8000 sudah dipakai backend Laravel — `run_dev.ps1`) |
| **Model** | 85.807.115 parameter · safetensors 327 MB · CPU ±1 dtk inferensi |
| **Cold start** | ±100 detik (model dimuat **di background**, `/health` tetap menjawab) |

---

## Menjalankan

### 1. Lokal — Windows

```powershell
cd ml-service
.\run.ps1
```

Sekali jalan, `run.ps1` akan: buat `.venv` → `pip install -r requirements.txt`
(bisa 5–10 menit, torch besar) → salin `.env.example` ke `.env` → start uvicorn.
Start berikutnya langsung jalan.

### 1b. Lokal — Linux / macOS

```bash
cd ml-service
./run.sh
```

> `run.sh` belum punya bit executable? `chmod +x run.sh`

### 2. Setelah start — jangan panik dengan 100 detik

Muat checkpoint 327 MB + import torch/transformers butuh ±100 detik. Selama itu
**`/health` tetap menjawab** (status `starting`), hanya `/predict` yang menjawab
`503 MODEL_NOT_LOADED`. Tanda siap di log:

```
ml.model INFO Model siap: 85,807,115 parameter, format=safetensors
ml.main  INFO Startup OK — model vit-v1.0.0 siap
```

### 3. Verifikasi

```powershell
# 1) liveness — 200 sejak detik pertama
curl http://127.0.0.1:8009/health

# 2) readiness — 503 selama model belum siap, 200 sesudahnya
curl http://127.0.0.1:8009/health?ready=1

# 3) label & metadata
curl http://127.0.0.1:8009/diseases
curl http://127.0.0.1:8009/model/info

# 4) prediksi sungguhan
curl -X POST http://127.0.0.1:8009/predict `
  -F "image=@foto_ikan.jpg" `
  -F "fish_type=Lele" `
  -F "gradcam=false"
```

Swagger UI: `http://127.0.0.1:8009/docs`

### 4. Hubungkan ke Laravel

`Kaliwedi-Backend/.env`:

```env
ML_DRIVER=http
ML_BASE_URL=http://127.0.0.1:8009
```

Default di `config/ngingu.php` juga sudah `:8009`. Selama `ML_DRIVER=fake`,
service Python **tidak wajib hidup** — driver fake memberi prediksi buatan (acak)
di dalam Laravel, tanpa menjalankan model ViT, khusus untuk dev/test.

---

## Deploy

### A. Docker

```bash
cd ml-service
docker compose up -d --build
docker compose logs -f ml-service
curl http://127.0.0.1:8009/health?ready=1
```

Tanpa compose:

```bash
docker build -t ngingu/ml-service .
docker run -d --name ngingu-ml -p 127.0.0.1:8009:8009 --memory=4g ngingu/ml-service
```

Detail isi `Dockerfile`:

- `python:3.12-slim` + dependensi OS untuk opencv (`libglib2.0-0`, `libgl1`).
- `torch`/`torchvision` diinstal dari **`--index-url https://download.pytorch.org/whl/cpu`**
  → wheel CPU saja, hemat ±2 GB dibanding wheel CUDA default yang tidak terpakai
  di server tanpa GPU.
- `ENV HF_OFFLINE=true` → tidak ada panggilan Hugging Face Hub, karena
  `MODEL_PATH` adalah folder `save_pretrained` lengkap. **Pakai `-e HF_OFFLINE=false`**
  bila `MODEL_PATH` Anda berupa 1 file state_dict (arsitektur diunduh dari hub).
- `HEALTHCHECK ... --start-period=180s` dengan `/health?ready=1` — margin lebar
  agar cold start ±100 dtk tidak dianggap container mati.
- Berjalan sebagai user non-root `mlsvc` (uid 10001).
- `docker-compose.yml` memetakan **`127.0.0.1:8009`** saja (service tak punya
  autentikasi). Butuh diakses container lain di jaringan compose? Ganti jadi
  `"8009:8009"` dan arahkan `ML_BASE_URL` ke `http://ml-service:8009`.

> ⚠️ **Belum pernah di-build di mesin pengembangan** (Docker tidak tersedia di
> sana). Jalankan checklist verifikasi di bawah sebelum dipakai produksi.

### B. Bare-metal / VPS (systemd + nginx)

```bash
# 1. checkout + venv
sudo mkdir -p /opt/ngingu && sudo chown $USER /opt/ngingu
git clone <repo> /opt/ngingu/ml-service
cd /opt/ngingu/ml-service
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # lalu isi; HOST tetap 127.0.0.1

# 2. user service
sudo useradd --system --home /opt/ngingu/ml-service --shell /usr/sbin/nologin ngingu
sudo chown -R ngingu:ngingu /opt/ngingu/ml-service

# 3. systemd
sudo cp deploy/ml-service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ml-service
systemctl status ml-service
journalctl -u ml-service -f

# 4. nginx reverse proxy
sudo cp deploy/nginx.conf /etc/nginx/conf.d/ngingu-ml.conf
sudo nginx -t && sudo systemctl reload nginx
```

`deploy/ml-service.service` sudah mengatur `Restart=on-failure`,
`TimeoutStartSec=300`, `MemoryMax=4G`, dan hardening.
`deploy/nginx.conf` sudah mengatur `client_max_body_size 12m` dan
`proxy_read_timeout 60s` (prediksi + GradCAM bisa 10–20 detik).

**Kenapa wajib nginx?** Service ini **tidak punya autentikasi**. systemd unit dan
default `HOST=127.0.0.1` sengaja mengikat loopback. Kalau harus diekspos,
aktifkan basic auth (contoh di komentar `deploy/nginx.conf`).

### Checklist verifikasi deploy

```bash
# 1. service hidup & model siap
curl -fsS http://127.0.0.1:8009/health?ready=1        # harus 200
# 2. 11 label + disease_id 1-11 ikut terbaca
curl -fsS http://127.0.0.1:8009/diseases | head -c 400
# 3. prediksi nyata (gradcam=false agar cepat)
curl -fsS -X POST http://127.0.0.1:8009/predict -F "image=@foto.jpg" -F "gradcam=false"
# 4. backend menunjuk port yang benar
grep ML_BASE_URL Kaliwedi-Backend/.env                # http://127.0.0.1:8009
# 5. CI guard urutan label
php artisan ml:sync-label-map --check                 # exit 0
```

---

## Endpoint

| Method | Path | Keterangan |
|---|---|---|
| GET | `/health` | Health check — selalu 200 (liveness) |
| GET | `/health?ready=1` | Readiness probe — 503 sampai model siap |
| GET | `/model/info` | Metadata model + input spec + label |
| GET | `/diseases` | 11 label penyakit yang didukung (FR-10.11) |
| POST | `/predict` | Prediksi 1 gambar — multipart (`image`, `gradcam`, `fish_type`) |
| POST | `/predict/batch` | Prediksi 2–5 gambar → diagregasi `weighted_average` |

### Contoh response `/predict` (API.md §10.1)

```json
{
  "disease": {
    "label": "aeromoniasis",
    "name": "Infeksi Bakteri Aeromonas (Aeromoniasis)",
    "slug": "aeromoniasis",
    "severity": "berat"
  },
  "confidence": 87.3,
  "top_predictions": [
    { "label": "aeromoniasis",  "name": "Infeksi Bakteri Aeromonas (Aeromoniasis)", "confidence": 87.3 },
    { "label": "bacterial_red_disease", "name": "Penyakit Bakteri Red Disease", "confidence": 8.1 },
    { "label": "healthy",       "name": "Ikan Sehat", "confidence": 4.6 }
  ],
  "low_confidence_warning": false,
  "gradcam_image": null,
  "processing_time_ms": 1243,
  "model_version": "vit-v1.0.0"
}
```

### Contoh response `/health`

```json
{ "status": "healthy", "model_version": "vit-v1.0.0",
  "model_format": "safetensors", "uptime_seconds": 101, "model_loaded": true }
```

`status`:

| Nilai | Arti |
|---|---|
| `healthy` | Model siap, `/predict` berfungsi |
| `starting` | Model sedang dimuat (±100 dtk pertama) |
| `degraded` | Gagal/tidak ada load — cek log, `/predict` → 503 |

### Kode error

| HTTP | `detail.code` | Kondisi |
|---|---|---|
| 400 | `UNSUPPORTED_MEDIA_TYPE` | Bukan JPG/PNG/WEBP atau file rusak |
| 413 | `PAYLOAD_TOO_LARGE` | File > 10 MB |
| 422 | `IMAGE_QUALITY_LOW` | Blur / gelap / terlalu terang (FR-10.12) — ada `suggestions[]` |
| 503 | `MODEL_NOT_LOADED` | Checkpoint belum ada, masih dimuat, atau gagal dimuat |
| 504 | `INFERENCE_TIMEOUT` | Inferensi > `PREDICT_TIMEOUT_SECONDS` (NFR-01.2) |

---

## Checkpoint fine-tuned

Taruh hasil fine-tuning di **`models/`** (default `MODEL_PATH`) — dua bentuk
didukung (FR-10.9):

| Bentuk | Isi | Butuh internet |
|---|---|---|
| Folder `save_pretrained` | `config.json` + `model.safetensors` + `preprocessor_config.json` | ❌ tidak |
| File state_dict tunggal | `model.safetensors` / `.pt` / `.bin` | ✅ ya (unduh arsitektur dari HF) |

Bila belum ada checkpoint, service **tetap jalan** — endpoint prediksi
mengembalikan `503 MODEL_NOT_LOADED` (degraded mode, NFR-03.2).

---

## Label map (11 kelas)

Semua label ada di **[`label_map.json`](./label_map.json)** — urutan array = urutan
indeks output model. Field `disease_id` **sudah terisi 1–11** dan disinkronkan ke
tabel `diseases` Laravel lewat:

```bash
php artisan ml:sync-label-map           # tulis disease_id dari DB ke label_map.json
php artisan ml:sync-label-map --check   # untuk CI: exit 1 bila tidak sinkron
```

Kelas asli model (dari `models/config.json` → `id2label`):

| # | `model_label` (mentah) | `label` | `name` | `severity` | `disease_id` |
|---|---|---|---|---|---|
| 0 | Fish_Bacterial Red disease | `bacterial_red_disease` | Penyakit Bakteri Red Disease | berat | 1 |
| 1 | Fish_Bacterial diseases - Aeromoniasis | `aeromoniasis` | Infeksi Bakteri Aeromonas (Aeromoniasis) | berat | 2 |
| 2 | Fish_Bacterial gill disease | `bacterial_gill_disease` | Penyakit Bakteri Insang | sedang | 3 |
| 3 | Fish_Fungal diseases Saprolegniasis | `saprolegniasis` | Saprolegniasis (Jamur Air) | sedang | 4 |
| 4 | Fish_Healthy Fish | `healthy` | Ikan Sehat | ringan | 5 |
| 5 | Fish_Parasitic diseases | `parasitic_disease` | Penyakit Parasit | sedang | 6 |
| 6 | Fish_Viral diseases White tail disease | `white_tail_disease` | Penyakit Virus White Tail | berat | 7 |
| 7 | Shrimp_Black_Gill | `shrimp_black_gill` | Udang Black Gill | sedang | 8 |
| 8 | Shrimp_Healthy | `shrimp_healthy` | Udang Sehat | ringan | 9 |
| 9 | Shrimp_White_Spot_Syndrome_Virus | `shrimp_white_spot` | Udang White Spot (WSSV) | berat | 10 |
| 10 | Shrimp_White_Spot_Syndrome_Virus_and_Black_Gill | `shrimp_wssv_black_gill` | Udang White Spot + Black Gill | kritis | 11 |

> **Guard urutan (FR-10.11):** field `model_label` dibandingkan dengan `id2label`
> di `models/config.json` saat startup. Kalau urutannya tidak persis sama, service
> **menolak load** dengan pesan yang menunjukkan index beda — karena urutan beda =
> prediksi salah mapping diam-diam. `ProcessScanJob` lalu me-resolve `disease_slug`
> ke tabel `diseases`, jadi `disease_id` harus sinkron.

---

## Konfigurasi

Salin `.env.example` → `.env`, atau set env var langsung (env var nyata
**menang** atas `.env`). Poin utama:

| Env | Default | Fungsi |
|---|---|---|
| `HOST` | `127.0.0.1` | Bind address — ubah ke `0.0.0.0` hanya di belakang proxy |
| `PORT` | `8009` | Port HTTP |
| `WORKERS` | `1` | **Jangan dinaikkan sembarangan** — tiap worker memuat model 327 MB sendiri |
| `MODEL_PATH` | `models` | Checkpoint fine-tuned |
| `MODEL_VERSION` | `vit-v1.0.0` | Versi di response |
| `DEVICE` | `auto` | `cpu` / `cuda` / `mps` |
| `HF_OFFLINE` | `false` (`true` di Docker) | Larang akses Hugging Face Hub |
| `PREDICT_TIMEOUT_SECONDS` | `30` | Hard limit → 504 |
| `LOW_CONFIDENCE_THRESHOLD` | `60` | < ini → warning |
| `BLUR_MIN_VARIANCE` | `60` | Threshold deteksi blur |
| `GRADCAM_ENABLED` | `true` | Saklar GradCAM global (tambah ±10 dtk) |
| `LOG_LEVEL` | `INFO` | Level log |

---

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Test berjalan **tanpa checkpoint** (classifier di-mock) — menguji format response,
validasi gambar, urutan label, health/readiness, dan seluruh endpoint.

---

## Troubleshooting

| Gejala | Penyebab | Solusi |
|---|---|---|
| `/health?ready=1` terus 503 | Model belum/gagal dimuat | `journalctl -u ml-service -f` / `docker compose logs` — cari `Model GAGAL dimuat` |
| `status: degraded` permanen | `MODEL_PATH` salah / checkpoint hilang | Cek `ls models/` — harus ada `config.json` + `model.safetensors` |
| `ValueError: urutan label ...` | `label_map.json` ≠ `models/config.json` | `php artisan ml:sync-label-map` lalu cek `model_label` |
| `504 INFERENCE_TIMEOUT` | GradCAM + CPU lambat | Naikkan `PREDICT_TIMEOUT_SECONDS` **dan** `ML_TIMEOUT` di backend, plus `proxy_read_timeout` nginx |
| Container restart-loop | `start_period` terlalu pendek | Pakai `--start-period=180s` (sudah ada di Dockerfile) |
| `connection refused` dari Laravel | Port salah / service mati | `ML_BASE_URL` harus `:8009`; `curl http://127.0.0.1:8009/health` |
| `137 (OOM killed)` | RAM kurang untuk torch + model | Naikkan limit (±4 GB per proses) atau pakai `WORKERS=1` |
| Import lambat ±90 dtk tiap start | Normal pada Python 3.14 | Pertimbangkan Python 3.12 di produksi (Docker sudah 3.12) |
