# 🐟 ML Service — Deteksi Penyakit Ikan

FastAPI service untuk klasifikasi penyakit ikan dari foto menggunakan
**`google/vit-base-patch16-224`** (Vision Transformer) yang sudah di-fine-tune ke
**11 kelas penyakit ikan**.

> Kontrak: [API.md §10](../API.md) · SRS FR-10 · Dipanggil **server-to-server** oleh
> Laravel (`MlClient`) — bukan langsung dari mobile app.

## Setup

```powershell
cd ml-service
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env     # lalu sesuaikan bila perlu
```

### Letakkan checkpoint fine-tuned

Taruh hasil fine-tuning di **`models/`** (folder ini sudah default `MODEL_PATH`) — dua
bentuk didukung (FR-10.9):

| Bentuk | Isi | Contoh |
|---|---|---|
| Folder `save_pretrained` | `config.json` + `model.safetensors` (+ `preprocessor_config.json`) | dari `model.save_pretrained("models")` |
| File state_dict tunggal | `model.safetensors` / `model.pt` / `model.bin` | checkpoint dari `torch.save(model.state_dict(), ...)` |

Lokasi bisa diganti via env `MODEL_PATH`. Bila belum ada checkpoint, service
**tetap jalan** — endpoint prediksi mengembalikan `503 MODEL_NOT_LOADED`
(degraded mode, NFR-03.2).

**Checklist sebelum jalan:** `label_map.json` ↔ `models/config.json` `id2label` harus
urutannya sama (dicek otomatis saat startup — lihat [Label map](#label-map-11-kelas)).

## Menjalankan

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Swagger UI: `http://127.0.0.1:8000/docs`

## Endpoint

| Method | Path | Keterangan |
|---|---|---|
| POST | `/predict` | Prediksi 1 gambar — multipart (`image`, `gradcam`, `fish_type`) |
| POST | `/predict/batch` | Prediksi 2–5 gambar → diagregasi `weighted_average` |
| GET | `/diseases` | 11 label penyakit yang didukung |
| GET | `/health` | Health check |
| GET | `/model/info` | Metadata model + input spec + label |

### Contoh

```powershell
# Prediksi
curl -X POST http://127.0.0.1:8000/predict `
  -F "image=@foto_ikan.jpg" `
  -F "fish_type=Lele" `
  -F "gradcam=false"

# Health
curl http://127.0.0.1:8000/health
```

**Response `/predict` (API.md §10.1):**

```json
{
  "disease": { "label": "white_spot", "name": "White Spot (Ich)", "slug": "white-spot-ich", "severity": "berat" },
  "confidence": 87.3,
  "top_predictions": [
    { "label": "white_spot", "name": "White Spot (Ich)", "confidence": 87.3 },
    { "label": "trichodina", "name": "Trichodina", "confidence": 8.1 },
    { "label": "healthy", "name": "Ikan Sehat", "confidence": 4.6 }
  ],
  "low_confidence_warning": false,
  "gradcam_image": null,
  "processing_time_ms": 1243,
  "model_version": "vit-v1.0.0"
}
```

### Kode error

| HTTP | `detail.code` | Kondisi |
|---|---|---|
| 400 | `UNSUPPORTED_MEDIA_TYPE` | Bukan JPG/PNG/WEBP atau file rusak |
| 413 | (pesan) | File > 10 MB |
| 422 | `IMAGE_QUALITY_LOW` | Blur / gelap / terlalu terang (FR-10.12) — ada `suggestions[]` |
| 503 | `MODEL_NOT_LOADED` | Checkpoint belum ada / gagal dimuat |
| 504 | `INFERENCE_TIMEOUT` | Inferensi > 30 detik (NFR-01.2) |

## Label map (11 kelas)

Semua label dikelola di **[`label_map.json`](./label_map.json)** — urutan array = urutan
indeks output model. **Sesuaikan `name`/`slug`/`disease_id`** dengan tabel `diseases`
Laravel agar prediksi otomatis ter-resolve (FR-10.11).

Kelas asli model (dari `models/config.json` → `id2label`):

| # | `model_label` (mentah) | `label` | `name` |
|---|---|---|---|
| 0 | Fish_Bacterial Red disease | `bacterial_red_disease` | Penyakit Bakteri Red Disease |
| 1 | Fish_Bacterial diseases - Aeromoniasis | `aeromoniasis` | Infeksi Bakteri Aeromonas |
| 2 | Fish_Bacterial gill disease | `bacterial_gill_disease` | Penyakit Bakteri Insang |
| 3 | Fish_Fungal diseases Saprolegniasis | `saprolegniasis` | Saprolegniasis (Jamur Air) |
| 4 | Fish_Healthy Fish | `healthy` | Ikan Sehat |
| 5 | Fish_Parasitic diseases | `parasitic_disease` | Penyakit Parasit |
| 6 | Fish_Viral diseases White tail disease | `white_tail_disease` | Penyakit Virus White Tail |
| 7 | Shrimp_Black_Gill | `shrimp_black_gill` | Udang Black Gill |
| 8 | Shrimp_Healthy | `shrimp_healthy` | Udang Sehat |
| 9 | Shrimp_White_Spot_Syndrome_Virus | `shrimp_white_spot` | Udang White Spot (WSSV) |
| 10 | Shrimp_White_Spot_Syndrome_Virus_and_Black_Gill | `shrimp_wssv_black_gill` | Udang White Spot + Black Gill |

```json
{ "label": "aeromoniasis", "model_label": "Fish_Bacterial diseases - Aeromoniasis",
  "name": "Infeksi Bakteri Aeromonas (Aeromoniasis)", "slug": "aeromoniasis",
  "severity": "berat", "disease_id": null }
```

> **Guard urutan (FR-10.11):** field `model_label` dibandingkan dengan `id2label` di
> `models/config.json` saat startup. Kalau urutannya tidak persis sama, service
> **menolak load** dengan pesan menunjukkan index yang beda — karena urutan beda =
> prediksi salah mapping diam-diam.
>
> `disease_id` masih `null` — isi dengan id record `diseases` Laravel agar
> `GET /diseases` sinkron (FR-10.11).

## Konfigurasi

Salin `.env.example` → `.env`, atau set env var langsung. Poin utama:

| Env | Default | Fungsi |
|---|---|---|
| `MODEL_PATH` | `models/vit-fish-11` | Checkpoint fine-tuned |
| `MODEL_VERSION` | `vit-v1.0.0` | Versi di response |
| `DEVICE` | `auto` | `cpu` / `cuda` / `mps` |
| `PREDICT_TIMEOUT_SECONDS` | `30` | Hard limit → 504 |
| `LOW_CONFIDENCE_THRESHOLD` | `60` | < ini → warning |
| `BLUR_MIN_VARIANCE` | `60` | Threshold deteksi blur |
| `GRADCAM_ENABLED` | `true` | Saklar GradCAM global |

## Terhubung dengan Laravel

Di `.env` Laravel (Kaliwedi-Backend):

```env
ML_DRIVER=http
ML_BASE_URL=http://127.0.0.1:8000
```

> **Catatan:** `MlClient.php` saat ini mengirim JSON `{images: [s3_key]}`, sedangkan
> kontrak API.md §10.1 adalah **multipart file**. Untuk end-to-end, `MlClient` perlu
> disesuaikan (upload file, bukan string path).

## Test

```powershell
.\.venv\Scripts\python.exe -m pytest -v
```

Test berjalan **tanpa checkpoint** (classifier di-mock) — menguji format response,
validasi gambar, dan seluruh endpoint.
