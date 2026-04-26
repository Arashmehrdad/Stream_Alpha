# Getting Started

## Prerequisites

- Windows PowerShell.
- Docker Desktop with the Linux engine running.
- Python 3.12 for local tests and scripts.
- A local `.env` file based on `.env.example`.
- Network access to Kraken public endpoints when running the producer.

## Clone and Setup

From the repository root:

```powershell
Copy-Item .env.example .env
```

Edit `.env` before starting services. Do not commit secrets or machine-specific values.

For a paper runtime, set:

```powershell
STREAMALPHA_RUNTIME_PROFILE=paper
STREAMALPHA_TRADING_CONFIG_PATH=configs/paper_trading.paper.yaml
```

If using the helper script, it sets runtime profile and trading config environment variables for the process while it runs.

## Docker Startup Flow

Start the paper stack:

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  up -d --build
```

Or use the helper:

```powershell
.\scripts\start-stack.ps1 `
  -Profile paper
```

The helper can seed recent OHLC history unless `-SkipOhlcBackfill` is used.

## Minimal Local Validation

Check containers:

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  ps
```

Check inference health:

```powershell
(Invoke-WebRequest `
  -UseBasicParsing `
  http://127.0.0.1:8000/health).Content
```

Expected high-level health fields include:

- `status = "ok"`
- `model_loaded = true`
- `startup_validation_passed = true`
- `health_overall_status = "HEALTHY"`

## Optional Documentation Build

Install the documentation dependency only when building docs locally:

```powershell
python -m pip install `
  -r requirements-docs.txt

python -m mkdocs build

python -m mkdocs serve
```

## Common First-Run Issues

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Compose warns about missing env vars | `.env` is missing profile or config values | Copy `.env.example` to `.env` and set `STREAMALPHA_RUNTIME_PROFILE` |
| `config-check` exits non-zero | Startup validation failed | Inspect `artifacts/runtime/startup_report.json` |
| Inference health is not healthy | Model, DB, or startup validation issue | Run `docker compose --profile paper --env-file .env logs inference config-check` |
| Producer cannot connect | Redpanda not healthy or network issue | Check `streamalpha-redpanda` health and producer logs |
| Ports already in use | Local service already owns 8000, 8501, 5432, 8080, or 19092 | Change ports in `.env` or stop the conflicting service |
