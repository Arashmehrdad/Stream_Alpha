# Stream Alpha Deployment Guide

This document describes the accepted M16 local deployment foundation.

## Goal

M16 makes the stack reproducible and easy to run without changing any accepted authority boundary from M4, M10, M11, M12, M13, M14, or M15.

## What M16 Adds

- one reusable Python app image at `docker/app.Dockerfile`
- Docker Compose profiles for `dev`, `paper`, `shadow`, and `live`
- one-shot startup validation at `python -m app.runtime.validate`
- env-driven trading config resolution for the trader and dashboard
- startup-report metadata on the existing inference APIs
- PowerShell helpers for start, stop, reset, and artifact pruning

## Files To Prepare

1. Copy `.env.example` to `.env`.
2. For live-only secrets and arming env, copy `.env.secrets.example` to `.env.secrets`.

`dev` does not require `.env.secrets`.

## Runtime Profiles

### `dev`

Purpose:
- infrastructure plus ingestion and features only

Compose services:
- `redpanda`
- `redpanda-console`
- `postgres`
- `config-check`
- `producer`
- `features`

Blank-clone truth:
- supported

### `paper`

Purpose:
- full local stack with simulated execution

Compose services:
- all `dev` services
- `inference`
- `trader`
- `dashboard`

Blank-clone truth:
- not supported honestly unless local model and regime artifacts already exist

### `shadow`

Purpose:
- full local stack with real execution plumbing but no broker submit

Compose services:
- all `paper` services

Blank-clone truth:
- not supported honestly unless local model and regime artifacts already exist

### `live`

Purpose:
- guarded live startup path with existing M12 controls

Compose services:
- all `paper` services

Additional requirements:
- `.env.secrets`
- `STREAMALPHA_ENABLE_LIVE=true`
- `STREAMALPHA_LIVE_CONFIRM` set to the exact guarded-live confirmation phrase
- broker credentials present
- model and regime artifacts present

Blank-clone truth:
- not supported honestly

## Startup Validation

`config-check` runs `python -m app.runtime.validate` before the dependent app services start.

It validates:

- `STREAMALPHA_RUNTIME_PROFILE`
- env parsing through the existing shared settings loader
- `STREAMALPHA_TRADING_CONFIG_PATH`
- profile to `execution.mode` alignment
- model artifact or registry-backed champion resolution for `paper`, `shadow`, and `live`
- regime runtime artifact resolution for `paper`, `shadow`, and `live`
- live secret presence and explicit live arming for `live`

It writes:

- `artifacts/runtime/startup_report.json`

It fails closed:

- dependent services wait on `config-check` success
- invalid profile or artifact state exits non-zero
- no secret values are written into the report

## One-Command Startup

### Development profile

```powershell
Copy-Item .env.example .env
.\scripts\start-stack.ps1 -Profile dev
```

### Paper profile

Requirements:
- `.env`
- local model artifact or registry-backed champion
- local regime thresholds artifact

```powershell
.\scripts\start-stack.ps1 -Profile paper
```

### Shadow profile

Requirements:
- same artifact requirements as `paper`

```powershell
.\scripts\start-stack.ps1 -Profile shadow
```

### Live profile

Requirements:
- same artifact requirements as `paper`
- `.env.secrets`
- explicit guarded-live arming env values

```powershell
Copy-Item .env.secrets.example .env.secrets
.\scripts\start-stack.ps1 -Profile live
```

## Stop And Reset

Stop without deleting state:

```powershell
.\scripts\stop-stack.ps1
```

Reset runtime state:

```powershell
.\scripts\reset-state.ps1
```

`reset-state.ps1` removes:

- Compose volumes for PostgreSQL and Redpanda
- runtime-only artifact directories:
  - `artifacts/live`
  - `artifacts/paper_trading`
  - `artifacts/rationale`
  - `artifacts/reliability`
  - `artifacts/runtime`

It intentionally keeps:

- `artifacts/training`
- `artifacts/regime`
- `artifacts/registry`

Those directories hold longer-lived evidence and promoted runtime inputs.

## Retention Rules

Persistent state:

- PostgreSQL named volume: `postgres-data`
- Redpanda named volume: `redpanda-data`
- host bind mount: `./artifacts:/workspace/artifacts`

Recommended runtime retention:

- keep `training`, `regime`, and `registry` artifacts until explicitly replaced or rolled back
- prune runtime-only artifacts periodically

Prune helper:

```powershell
.\scripts\prune-runtime-artifacts.ps1 -RetentionDays 14
```

## Compose Behavior

Key runtime behavior:

- `dev` starts infra plus producer/features only
- `paper`, `shadow`, and `live` start the full stack
- `producer`, `features`, `inference`, `trader`, and `dashboard` all wait for `config-check` success
- `trader` and `dashboard` load trading config from `STREAMALPHA_TRADING_CONFIG_PATH`
- `inference` exposes additive `runtime_profile`, `execution_mode`, `startup_validation_passed`, and `startup_report_path` on:
  - `/health`
  - `/metrics`
  - `/reliability/system`

## Known Limitations

- Blank-clone `paper`, `shadow`, and `live` still fail honestly until model and regime artifacts are available locally.
- This guide does not add broker fill import, deployment automation, or alert routing.
- Live mode still inherits the accepted M12 fail-closed behavior rather than adding a new orchestration or recovery layer.
