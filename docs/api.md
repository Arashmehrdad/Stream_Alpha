# API

The inference API is defined in `app/inference/main.py` and served by the `inference` service on port `8000`.

Base URL for local host access:

```text
http://127.0.0.1:8000
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service, model, runtime, DB, and operational health. |
| GET | `/latest-features` | Latest feature row for a symbol and optional interval. |
| GET | `/predict` | Model prediction for a symbol and optional interval. |
| GET | `/regime` | Regime classification for a symbol and optional interval. |
| GET | `/signal` | Trading signal for a symbol and optional interval. |
| GET | `/freshness` | Freshness evaluation for a symbol and optional interval. |
| GET | `/metrics` | In-memory request counters and latency metrics. |
| GET | `/adaptation/summary` | Adaptation summary. |
| GET | `/adaptation/drift` | Adaptation drift rows. |
| GET | `/adaptation/performance` | Adaptation performance rows. |
| GET | `/adaptation/profiles` | Adaptation profiles. |
| GET | `/adaptation/promotions` | Adaptation promotion decisions. |
| GET | `/continual-learning/summary` | Continual-learning summary. |
| GET | `/continual-learning/experiments` | Continual-learning experiments. |
| GET | `/continual-learning/profiles` | Continual-learning profiles. |
| GET | `/continual-learning/drift-caps` | Continual-learning drift caps. |
| GET | `/continual-learning/promotions` | Continual-learning promotions. |
| GET | `/continual-learning/events` | Continual-learning events. |
| POST | `/continual-learning/promotions/promote-profile` | Operator/internal endpoint for guarded continual-learning profile promotion. |
| POST | `/continual-learning/promotions/rollback-active-profile` | Operator/internal endpoint for guarded continual-learning profile rollback. |
| GET | `/reliability/system` | Cross-service reliability state. |
| GET | `/alerts/active` | Active operational alerts. |
| GET | `/alerts/timeline` | Alert timeline. |
| GET | `/operations/daily-summary` | Daily operations summary artifact. |
| GET | `/operations/startup-safety` | Startup safety artifact. |

## Common Query Parameters

| Endpoint | Parameters |
| --- | --- |
| `/latest-features`, `/predict`, `/regime`, `/signal`, `/freshness` | `symbol`, optional `interval_begin` |
| `/adaptation/summary` | optional `symbol`, `regime_label` |
| `/adaptation/drift` | optional `symbol`, `regime_label`, `limit` |
| `/adaptation/performance` | optional `execution_mode`, `symbol`, `regime_label`, `limit` |
| `/adaptation/profiles`, `/adaptation/promotions` | optional `limit` |
| `/continual-learning/*` GET endpoints | optional filters visible in `app/inference/main.py` |
| `/alerts/timeline` | optional `limit`, `category`, `severity`, `symbol`, `active_only` |
| `/operations/daily-summary` | optional `date` |

## Examples

Health:

```powershell
(Invoke-WebRequest `
  -UseBasicParsing `
  http://127.0.0.1:8000/health).Content
```

Prediction:

```powershell
(Invoke-WebRequest `
  -UseBasicParsing `
  "http://127.0.0.1:8000/predict?symbol=BTC%2FUSD").Content
```

Signal:

```powershell
(Invoke-WebRequest `
  -UseBasicParsing `
  "http://127.0.0.1:8000/signal?symbol=BTC%2FUSD").Content
```

## Operator/Internal Continual-Learning Writes

The continual-learning POST endpoints are documented as operator/internal endpoints, not normal public-user API endpoints.

Confirmed endpoints:

- `POST /continual-learning/promotions/promote-profile`
- `POST /continual-learning/promotions/rollback-active-profile`

The request schemas are named `ContinualLearningPromoteProfileRequest` and `ContinualLearningRollbackRequest`, and the code describes them as guarded operator requests. These calls can affect persisted continual-learning/runtime state by saving promotion or rollback decisions, events, and profile artifacts when guards pass.

Use only with explicit operator intent. TODO: verify operator access policy before exposing these routes beyond trusted local/operator workflows.

## Error Cases

| Status | Cause |
| --- | --- |
| 400 | Invalid symbol or invalid `interval_begin`. |
| 404 | No feature row found for the requested symbol/time, or missing operation artifact. |
| 500 | Artifact schema mismatch or unexpected workflow error. |
| 503 | Database unavailable or workflow temporarily unavailable. |

Request and response schemas are defined in `app/inference/schemas.py`, `app/adaptation/schemas.py`, `app/continual_learning/schemas.py`, and related modules.
