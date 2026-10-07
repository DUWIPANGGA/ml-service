#Requires -Version 5.1
<#
.SYNOPSIS
  Ngingu ML Service — jalankan (Windows).

.DESCRIPTION
  Sekali jalan: membuat .venv, memasang requirements.txt, menyalin .env.example
  bila belum ada, lalu start uvicorn. Start berikutnya langsung jalan.

  Model dimuat di background (±100 detik) — /health sudah menjawab sejak awal,
  jadi tidak perlu menunggu untuk mengecek kesehatan service.

.EXAMPLE
  .\run.ps1
  .\run.ps1 -Port 8010 -BindHost 0.0.0.0    # buka ke jaringan lokal
#>
[CmdletBinding()]
param(
    [string]$Py = '',
    [string]$BindHost = '',
    [int]$Port = 0
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

# ── 1. Python + virtualenv ──────────────────────────────────────────────────
$python = $Py
if (-not $python) {
    foreach ($candidate in @('.venv\Scripts\python.exe', 'python', 'py')) {
        if ($candidate -like '*python.exe') {
            if (Test-Path (Join-Path $PSScriptRoot $candidate)) { $python = $candidate; break }
        } elseif (Get-Command $candidate -ErrorAction SilentlyContinue) {
            $python = $candidate; break
        }
    }
}
if (-not $python) {
    Write-Host '  [X] Python tidak ditemukan di PATH. Install Python 3.11+ dulu.' -ForegroundColor Red
    exit 1
}

if (-not (Test-Path '.venv')) {
    Write-Host '[1/3] Membuat virtualenv ...' -ForegroundColor Cyan
    & $python -m venv .venv
    $python = '.venv\Scripts\python.exe'
} else {
    $python = '.venv\Scripts\python.exe'
}

# ── 2. Dependensi (sekali saja) ─────────────────────────────────────────────
$stamp = Join-Path '.venv' '.deps-installed'
if (-not (Test-Path $stamp)) {
    Write-Host '[2/3] Memasang dependensi (torch + transformers, bisa 5-10 menit) ...' -ForegroundColor Cyan
    & $python -m pip install --upgrade pip
    & $python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { Write-Host '  [X] pip install gagal.' -ForegroundColor Red; exit 1 }
    New-Item -ItemType File -Path $stamp -Force | Out-Null
} else {
    Write-Host '[2/3] Dependensi sudah terpasang.' -ForegroundColor Cyan
}

if (-not (Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
    Write-Host '      .env dibuat dari .env.example' -ForegroundColor DarkGray
}

# ── 3. Override CLI ─────────────────────────────────────────────────────────
if ($Port -gt 0)  { $env:PORT = "$Port" }
if ($BindHost)    { $env:HOST = $BindHost }

$portShown = if ($env:PORT) { $env:PORT } else { '8009' }
$hostShown = if ($env:HOST) { $env:HOST } else { '127.0.0.1' }

Write-Host ''
Write-Host '[3/3] Menjalankan uvicorn (model dimuat di background ±100 detik) ...' -ForegroundColor Cyan
Write-Host "      health  : http://${hostShown}:${portShown}/health" -ForegroundColor Yellow
Write-Host "      docs    : http://${hostShown}:${portShown}/docs" -ForegroundColor Yellow
Write-Host "      siap    : http://${hostShown}:${portShown}/health?ready=1" -ForegroundColor Yellow
Write-Host ''

& $python serve.py
