# Stream Alpha

Stream Alpha is a local-first crypto market data pipeline built with Docker Compose, Redpanda Community Edition, PostgreSQL, and Python services.

Milestone `M1` ingests Kraken public WebSocket v2 market data for `BTC/USD`, `ETH/USD`, and `SOL/USD`, normalizes `trade` and `ohlc` events, publishes them into Redpanda, and writes raw rows into PostgreSQL.

Milestone `M2` adds an OHLC-only feature consumer. It reads `raw.ohlc`, finalizes closed candles safely, computes a minimal rolling feature set using only current and past finalized candles, and upserts typed rows into PostgreSQL table `feature_ohlc`.

Milestone `M3` adds an offline training pipeline. It builds a labeled dataset from `feature_ohlc`, evaluates naive and learned classifiers with purged expanding walk-forward validation, and saves model plus evaluation artifacts under `artifacts/training/m3/<run_id>/`.

Milestone `M4` adds a minimal FastAPI inference API. It loads the accepted saved M3 artifact, reads the latest canonical feature row from `feature_ohlc`, and serves health, latest-feature, prediction, signal, and JSON metrics endpoints.

Milestone `M5` adds a minimum correct paper-trading engine. It polls finalized canonical candles from `feature_ohlc`, asks the existing M4 `/signal` endpoint for the authoritative BUY/SELL/HOLD decision for each exact candle, simulates long-only spot fills, persists positions and trade ledger rows in PostgreSQL, and writes rolling paper-trading summaries under `artifacts/paper_trading/`.

Milestone `M6` adds a read-only Streamlit operator dashboard. It reads only from the accepted M4 inference API and PostgreSQL tables, then shows health, latest signals, canonical feature snapshots, open positions, recent trades, ledger activity, and paper-trading PnL/drawdown.

Milestone `M7` adds a local-first retraining and model-comparison workflow. It retrains challengers from canonical `feature_ohlc`, compares them to the current promoted champion with explicit file-based policy checks, promotes only passing models into a local registry, and supports rollback without retraining.

Milestone `M8` foundation adds an explicit offline regime workflow. It reads canonical `feature_ohlc`, fits deterministic per-symbol volatility and trend thresholds, classifies rows into `TREND_UP`, `TREND_DOWN`, `RANGE`, or `HIGH_VOL`, and writes explicit artifacts under `artifacts/regime/m8/<run_id>/`.

Milestone `M9` adds a minimum regime-aware live signal foundation. It loads the saved M8 `thresholds.json`, resolves regime labels for the exact canonical feature row used by M4, extends the accepted inference and paper-trading contracts additively, and surfaces by-regime performance in the existing dashboard and trading artifacts.

## Repository Tree

```text
.
|-- .env.example
|-- .gitignore
|-- README.md
|-- app
|   |-- __init__.py
|   |-- common
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- logging.py
|   |   |-- models.py
|   |   |-- serialization.py
|   |   `-- time.py
|   |-- features
|   |   |-- __init__.py
|   |   |-- __main__.py
|   |   |-- db.py
|   |   |-- engine.py
|   |   |-- main.py
|   |   |-- models.py
|   |   |-- service.py
|   |   `-- state.py
|   |-- inference
|   |   |-- __init__.py
|   |   |-- __main__.py
|   |   |-- db.py
|   |   |-- main.py
|   |   |-- schemas.py
|   |   `-- service.py
|   |-- regime
|   |   |-- __init__.py
|   |   |-- __main__.py
|   |   |-- artifacts.py
|   |   |-- config.py
|   |   |-- dataset.py
|   |   |-- live.py
|   |   `-- service.py
|   |-- trading
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- engine.py
|   |   |-- metrics.py
|   |   |-- repository.py
|   |   |-- risk.py
|   |   |-- runner.py
|   |   |-- schemas.py
|   |   `-- signal_client.py
|   |-- training
|   |   |-- __init__.py
|   |   |-- __main__.py
|   |   |-- baselines.py
|   |   |-- compare.py
|   |   |-- dataset.py
|   |   |-- promote.py
|   |   |-- registry.py
|   |   |-- retrain.py
|   |   |-- rollback.py
|   |   |-- service.py
|   |   `-- splits.py
|   `-- ingestion
|       |-- __init__.py
|       |-- __main__.py
|       |-- backfill_ohlc.py
|       |-- db.py
|       |-- kraken.py
|       |-- main.py
|       |-- normalizers.py
|       |-- publisher.py
|       `-- service.py
|-- configs
|   |-- paper_trading.yaml
|   |-- regime.m8.json
|   |-- regime_signal_policy.json
|   |-- redpanda-console-config.yml
|   |-- training.m3.json
|   `-- training.m7.json
|-- dashboards
|   |-- __init__.py
|   |-- data_sources.py
|   |-- streamlit_app.py
|   |-- view_models.py
|   `-- widgets.py
|-- docker
|   `-- producer.Dockerfile
|-- docker-compose.yml
|-- docs
|   |-- m1-overview.md
|   `-- screenshots
|       `-- README.md
|-- requirements.txt
|-- scripts
|   |-- check-db.ps1
|   |-- check-topics.ps1
|   |-- m3-smoke-run.sh
|   |-- run_paper_trader.py
|   `-- tail-producer.ps1
`-- tests
    |-- __init__.py
    |-- test_dashboard_data_sources.py
    |-- test_dashboard_view_models.py
    |-- test_inference_api.py
    |-- test_inference_db.py
    |-- test_inference_model_loader.py
    |-- test_feature_engine.py
    |-- test_feature_state.py
    |-- test_normalizers.py
    |-- test_publish_smoke.py
    |-- test_training_compare.py
    |-- test_training_labels.py
    |-- test_training_registry.py
    |-- test_training_splits.py
    |-- training_workflow_helpers.py
    `-- trading
        |-- test_engine.py
        |-- test_metrics.py
        |-- test_risk.py
        `-- test_runner_idempotency.py
```

## Services

- `redpanda`: local Redpanda broker for topics and consumer groups
- `redpanda-console`: topic and cluster inspection UI
- `postgres`: local PostgreSQL for raw and feature tables
- `producer`: Kraken public WebSocket v2 ingestion service from M1
- `features`: OHLC-only feature consumer from M2
- `inference`: FastAPI prediction API from M4, run directly from the repo with the accepted M3 artifact
- `paper-trader`: long-only spot paper-trading engine from M5, run directly from the repo against canonical features plus authoritative M4/M9 signals
- `dashboard`: read-only Streamlit UI from M6, run directly from the repo against the accepted API and PostgreSQL sources, including additive M9 regime fields

## M2 Scope

M2 does:
- consume only `raw.ohlc`
- finalize candles on next-interval arrival or grace-based sweep
- bootstrap rolling state from `raw_ohlc`
- compute typed feature rows for finalized candles only
- upsert rows into `feature_ohlc`

M2 does not do:
- consume `raw.trades`
- publish a feature topic
- create dashboards, APIs, training jobs, or inference services
- add trading, signals, or paper trading logic

## M3 Scope

M3 does:
- use `feature_ohlc` as the canonical offline training source
- construct next-3-candle binary direction labels
- evaluate `persistence_3` and `DummyClassifier(strategy="most_frequent")` baselines
- train `LogisticRegression` and `HistGradientBoostingClassifier`
- use a global purged expanding walk-forward split with 5 test folds
- save explicit run artifacts under `artifacts/training/m3/<run_id>/`

M3 does not do:
- serve inference online
- generate signals
- paper trade
- add FastAPI, dashboards, MLflow, RL, or sentiment/news modules

## M4 Scope

M4 does:
- load the accepted saved M3 model artifact directly from disk
- read the latest finalized canonical row from `feature_ohlc`
- serve `GET /health`, `GET /latest-features`, `GET /predict`, `GET /signal`, and `GET /metrics`
- return BUY, SELL, or HOLD using explicit probability thresholds
- maintain simple in-memory request and latency counters since startup

M4 does not do:
- retrain models
- accept custom feature payloads
- add batch endpoints, dashboards, paper trading, or signal-history storage
- introduce MLflow, Prometheus, Grafana, or deployment automation

## M5 Scope

M5 does:
- poll newly finalized canonical candles from `feature_ohlc`
- call the existing M4 `/signal` endpoint for each exact candle
- simulate long-only spot BUY/SELL/HOLD execution with next-open fills
- apply fixed-fraction sizing, cooldown, stop loss, take profit, max open positions, and max exposure per asset
- persist `paper_positions`, `paper_trade_ledger`, and `paper_engine_state`
- write rolling summary artifacts under `artifacts/paper_trading/`

M5 does not do:
- place live orders
- short, pyramid, or use leverage
- add broker integrations, dashboards, monitoring stacks, or retraining hooks

## M6 Scope

M6 does:
- run one read-only Streamlit dashboard locally from the repo
- call the accepted M4 `GET /health` and `GET /signal` endpoints
- read canonical `feature_ohlc` rows plus M5 trading tables from PostgreSQL
- show health, latest signals, latest canonical feature snapshots, open positions, recent trades, recent ledger activity, and paper-trading PnL/drawdown
- surface degraded states clearly when the inference API or PostgreSQL is unavailable

M6 does not do:
- retrain or reload models from any new workflow
- add a new backend service for the dashboard
- write dashboard state back into PostgreSQL
- add dashboards beyond Streamlit, live trading, RL, sentiment/news, MLflow, Grafana, Prometheus, or alerting

## M7 Scope

M7 does:
- retrain challengers from canonical `feature_ohlc` using the accepted M3 evaluation flow
- compare challengers to the current champion with explicit file-based policy checks
- store challenger artifacts under `artifacts/training/m7/<run_id>/`
- maintain a local immutable registry under `artifacts/registry/`
- support explicit rollback by `model_version`
- let M4 inference resolve the current champion from the registry when `INFERENCE_MODEL_PATH` is empty

M7 does not do:
- add MLflow, schedulers, or background retraining daemons
- add dashboard controls for retraining or promotion
- change M3 labels, models, splits, or artifact schema
- change M4 response contracts, add champion/challenger online serving, or retrain during rollback

## M8 Scope

M8 foundation does:
- read only from canonical `feature_ohlc`
- fit per-symbol `high_vol_threshold` from the 75th percentile of `realized_vol_12`
- fit per-symbol `trend_abs_threshold` from the 60th percentile of `abs(momentum_3)`
- classify rows with explicit deterministic rules into `TREND_UP`, `TREND_DOWN`, `RANGE`, or `HIGH_VOL`
- write explicit offline artifacts under `artifacts/regime/m8/<run_id>/`

M8 foundation does not do:
- change M4, M5, M6, or M7 behavior
- add a `/regime` endpoint, schedulers, background workers, or new PostgreSQL tables
- add clustering, RL, sentiment, anomaly detection, MLflow, or notebooks

## M9 Scope

M9 does:
- load the latest saved M8 `thresholds.json` by default, or an explicit threshold artifact path when configured
- resolve the regime for the exact canonical `feature_ohlc` row used by `GET /predict`, `GET /signal`, and `GET /regime`
- extend `GET /predict` with `regime_label` and `regime_run_id`
- extend `GET /signal` with `regime_label`, `regime_run_id`, and `trade_allowed`
- add a read-only `GET /regime` endpoint
- apply regime-dependent BUY/SELL thresholds and BUY no-trade rules in M4 only
- persist additive regime labels in `paper_engine_state`, `paper_positions`, and `paper_trade_ledger`
- extend paper-trading summaries with by-regime hit rate and PnL outputs
- surface regime fields and a compact by-regime performance table in the existing M6 dashboard

M9 does not do:
- add live trading, shorting, portfolio allocation, or position-sizing changes
- add a new PostgreSQL table, orchestration stack, scheduler, control plane, or background daemon
- duplicate regime logic in M5 or change M7 registry model resolution
- add RL, sentiment/news, anomaly detection, MLflow, or adaptive threshold tuning

## Environment Variables

Copy `.env.example` to `.env` before running the stack.

| Variable | Purpose | Default |
| --- | --- | --- |
| `APP_NAME` | Logical app name embedded in events | `streamalpha` |
| `LOG_LEVEL` | Python log level | `INFO` |
| `KRAKEN_WS_URL` | Kraken public WebSocket v2 endpoint | `wss://ws.kraken.com/v2` |
| `KRAKEN_REST_OHLC_URL` | Kraken public REST OHLC endpoint used for explicit backfills | `https://api.kraken.com/0/public/OHLC` |
| `KRAKEN_SYMBOLS` | Comma-separated symbols | `BTC/USD,ETH/USD,SOL/USD` |
| `KRAKEN_OHLC_INTERVAL_MINUTES` | Candle interval in minutes | `5` |
| `KAFKA_BOOTSTRAP_SERVERS` | Redpanda bootstrap servers | `redpanda:9092` |
| `KAFKA_CLIENT_ID` | Kafka producer client id | `streamalpha-producer` |
| `POSTGRES_HOST` | PostgreSQL host | `postgres` |
| `POSTGRES_PORT` | PostgreSQL port | `5432` |
| `POSTGRES_DB` | Database name | `streamalpha` |
| `POSTGRES_USER` | Database user | `streamalpha` |
| `POSTGRES_PASSWORD` | Database password | `change-me-local-only` |
| `TOPIC_RAW_TRADES` | Trade topic name | `raw.trades` |
| `TOPIC_RAW_OHLC` | OHLC topic name | `raw.ohlc` |
| `TOPIC_RAW_HEALTH` | Health topic name | `raw.health` |
| `TABLE_RAW_TRADES` | Trade table name | `raw_trades` |
| `TABLE_RAW_OHLC` | Raw OHLC table name | `raw_ohlc` |
| `TABLE_FEATURE_OHLC` | Finalized OHLC feature table name | `feature_ohlc` |
| `TABLE_PRODUCER_HEARTBEAT` | Heartbeat table name | `producer_heartbeat` |
| `PRODUCER_SERVICE_NAME` | Producer service name | `producer` |
| `PRODUCER_HEARTBEAT_INTERVAL_SECONDS` | Producer heartbeat cadence | `15` |
| `FEATURE_CONSUMER_GROUP_ID` | Kafka consumer group id for M2 | `streamalpha-feature-consumer` |
| `FEATURE_SERVICE_NAME` | Feature service name used in logs and client id | `features` |
| `FEATURE_FINALIZATION_GRACE_SECONDS` | Grace period before stale open-candle finalization | `30` |
| `FEATURE_BOOTSTRAP_CANDLES` | Raw OHLC candles loaded per symbol on startup | `64` |
| `INFERENCE_MODEL_PATH` | Optional explicit `model.joblib` override; when empty M4 resolves the current promoted champion from the local registry | empty |
| `INFERENCE_SERVICE_NAME` | Inference API service name used in logs and responses | `inference` |
| `INFERENCE_SIGNAL_BUY_PROB_UP` | BUY threshold for `prob_up` | `0.55` |
| `INFERENCE_SIGNAL_SELL_PROB_UP` | SELL threshold for `prob_up` | `0.45` |
| `INFERENCE_REGIME_THRESHOLDS_PATH` | Optional explicit M8 `thresholds.json`; when empty M9 uses the latest run under `artifacts/regime/m8/` | empty |
| `INFERENCE_REGIME_SIGNAL_POLICY_PATH` | Optional explicit M9 regime signal policy path | `configs/regime_signal_policy.json` |
| `INFERENCE_API_BASE_URL` | Base URL for the accepted local M4 inference API | `http://127.0.0.1:8000` |
| `DASHBOARD_REFRESH_SECONDS` | Browser refresh cadence for the M6 Streamlit UI | `15` |
| `DASHBOARD_RECENT_TRADES_LIMIT` | Recent closed trades shown in the dashboard | `20` |
| `DASHBOARD_RECENT_LEDGER_LIMIT` | Recent ledger rows shown in the dashboard | `20` |
| `RECONNECT_INITIAL_DELAY_SECONDS` | First reconnect delay | `1` |
| `RECONNECT_MAX_DELAY_SECONDS` | Reconnect delay cap | `30` |
| `RECONNECT_BACKOFF_MULTIPLIER` | Backoff multiplier | `2.0` |
| `RECONNECT_JITTER_SECONDS` | Random jitter added to backoff | `0.5` |

## Run Locally

### 1. Prepare the environment

```powershell
Copy-Item .env.example .env
```

### 2. Start the full local stack

```powershell
docker compose up --build -d
```

### 3. Follow the producer and feature consumer logs

```powershell
docker compose logs -f producer
docker compose logs -f features
```

### 4. Run only the feature consumer service after the shared stack is already up

```powershell
docker compose up --build -d redpanda postgres producer
docker compose up --build features
```

### 5. Run the feature consumer directly with Python

```powershell
python -m pip install -r requirements.txt
python -m app.features.main
```

### 6. Run the M3 offline training pipeline

```powershell
python -m pip install -r requirements.txt
python -m app.training --config configs/training.m3.json
```

### 7. Backfill recent Kraken REST OHLC history into the canonical raw and feature tables

```powershell
python -m app.ingestion.backfill_ohlc --symbols BTC/USD ETH/USD SOL/USD --interval 5
```

The backfill command upserts closed historical candles into `raw_ohlc` and then regenerates `feature_ohlc` directly from the existing raw/canonical flow without changing the M3 training source.

### 8. Run the WSL-based M3 smoke helper with bounded PostgreSQL readiness checks

```powershell
wsl -e bash /mnt/d/Github/Stream_Alpha/scripts/m3-smoke-run.sh
```

### 9. Run the M4 and M9 inference API from the repo

Set `INFERENCE_MODEL_PATH` to the accepted saved M3 artifact if you want a direct override. For example:

```powershell
$env:INFERENCE_MODEL_PATH = "D:\Github\Stream_Alpha\artifacts\training\m3\<run_id>\model.joblib"
$env:INFERENCE_REGIME_THRESHOLDS_PATH = "D:\Github\Stream_Alpha\artifacts\regime\m8\<run_id>\thresholds.json"
python -m app.inference
```

The API listens on `http://127.0.0.1:8000` by default.

When `INFERENCE_MODEL_PATH` is empty, M4 resolves the current champion from `artifacts/registry/current.json` instead.

When `INFERENCE_REGIME_THRESHOLDS_PATH` is empty, M9 resolves the latest saved `thresholds.json` under `artifacts/regime/m8/`.

### 10. Example inference requests

```powershell
curl "http://127.0.0.1:8000/health"
curl "http://127.0.0.1:8000/latest-features?symbol=BTC/USD"
curl "http://127.0.0.1:8000/predict?symbol=BTC/USD"
curl "http://127.0.0.1:8000/regime?symbol=BTC/USD"
curl "http://127.0.0.1:8000/signal?symbol=BTC/USD"
curl "http://127.0.0.1:8000/metrics"
```

Example `GET /predict` response:

```json
{
  "symbol": "BTC/USD",
  "model_name": "logistic_regression",
  "model_trained_at": "2026-03-19T22:30:02Z",
  "model_artifact_path": "D:/Github/Stream_Alpha/artifacts/training/m3/<run_id>/model.joblib",
  "row_id": "BTC/USD|2026-03-19T22:00:00Z",
  "interval_begin": "2026-03-19T22:00:00Z",
  "as_of_time": "2026-03-19T22:05:00Z",
  "prob_up": 0.61,
  "prob_down": 0.39,
  "predicted_class": "UP",
  "confidence": 0.61,
  "regime_label": "TREND_UP",
  "regime_run_id": "20260320T165813Z"
}
```

### 11. Run the M5 paper trader once

Make sure PostgreSQL, canonical `feature_ohlc`, and the M4 inference API are already available first.

```powershell
python scripts\run_paper_trader.py --config configs\paper_trading.yaml --once
```

### 12. Run the M5 paper trader in polling mode

```powershell
python scripts\run_paper_trader.py --config configs\paper_trading.yaml
```

The paper trader is intentionally local-first and long-only. It consumes the existing M4 `/signal` endpoint as authoritative, including M9 regime-aware decisions, and never reimplements signal logic.

### 13. Run the M6 Streamlit dashboard

Make sure PostgreSQL is reachable and the M4 inference API is running first.

```powershell
$env:POSTGRES_HOST = "127.0.0.1"
$env:INFERENCE_API_BASE_URL = "http://127.0.0.1:8000"
streamlit run dashboards/streamlit_app.py
```

The dashboard defaults to [http://localhost:8501](http://localhost:8501).

### 14. Example dashboard validation requests

```powershell
curl "http://127.0.0.1:8000/health"
curl "http://127.0.0.1:8000/signal?symbol=BTC/USD"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS feature_rows FROM feature_ohlc;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS open_positions FROM paper_positions WHERE status = 'OPEN';"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS ledger_rows FROM paper_trade_ledger;"
```

Example `GET /signal` response:

```json
{
  "symbol": "BTC/USD",
  "signal": "BUY",
  "reason": "prob_up 0.6100 >= buy threshold 0.54",
  "prob_up": 0.61,
  "prob_down": 0.39,
  "confidence": 0.61,
  "predicted_class": "UP",
  "thresholds": {
    "buy_prob_up": 0.54,
    "sell_prob_up": 0.44
  },
  "row_id": "BTC/USD|2026-03-19T22:00:00Z",
  "as_of_time": "2026-03-19T22:05:00Z",
  "model_name": "logistic_regression",
  "regime_label": "TREND_UP",
  "regime_run_id": "20260320T165813Z",
  "trade_allowed": true
}
```

Example `GET /regime` response:

```json
{
  "symbol": "BTC/USD",
  "row_id": "BTC/USD|2026-03-19T22:00:00Z",
  "interval_begin": "2026-03-19T22:00:00Z",
  "as_of_time": "2026-03-19T22:05:00Z",
  "regime_label": "TREND_UP",
  "regime_run_id": "20260320T165813Z",
  "regime_artifact_path": "D:/Github/Stream_Alpha/artifacts/regime/m8/20260320T165813Z/thresholds.json",
  "realized_vol_12": 0.03,
  "momentum_3": 0.03,
  "macd_line_12_26": 1.2,
  "high_vol_threshold": 0.05,
  "trend_abs_threshold": 0.02,
  "trade_allowed": true,
  "buy_prob_up": 0.54,
  "sell_prob_up": 0.44
}
```

### 15. Run the M7 retraining workflow

Run one challenger retraining job from the canonical source:

```powershell
python -m app.training.retrain --config configs/training.m7.json
```

This writes one timestamped challenger run under `artifacts/training/m7/<run_id>/`, including `comparison_vs_champion.json` and `run_manifest.json`.

Compare an existing challenger explicitly if needed:

```powershell
python -m app.training.compare --config configs/training.m7.json --run-dir artifacts/training/m7/<run_id>
```

Promote a passing challenger into the immutable local registry:

```powershell
python -m app.training.promote --run-dir artifacts/training/m7/<run_id> --model-version m7-<run_id>
```

Rollback to one previously promoted version without retraining:

```powershell
python -m app.training.rollback --model-version <model_version>
```

Bootstrap the registry from the current accepted artifact when you want registry-based inference before the first M7 challenger promotion:

```powershell
python -m app.training.promote --run-dir artifacts/training/m3/<run_id> --model-version m3-<run_id>
```

### 16. Run the M8 offline regime workflow

```powershell
python -m app.regime --config configs/regime.m8.json
```

Each run writes:

```text
artifacts/regime/m8/<run_id>/
|-- by_symbol_summary.csv
|-- overall_summary.json
|-- regime_predictions.csv
|-- run_config.json
|-- run_manifest.json
`-- thresholds.json
```

## Tests And Lint

```powershell
python -m pytest
python -m pylint app dashboards tests scripts\run_paper_trader.py
```

## Training Artifacts

Each M3 run writes a timestamped artifact directory:

```text
artifacts/training/m3/<run_id>/
|-- dataset_manifest.json
|-- feature_columns.json
|-- fold_metrics.csv
|-- model.joblib
|-- oof_predictions.csv
|-- run_config.json
`-- summary.json
```

Quick checks after a run:

```powershell
Get-ChildItem artifacts\training\m3
Get-Content artifacts\training\m3\<run_id>\summary.json
Get-Content artifacts\training\m3\<run_id>\dataset_manifest.json
```

## M7 Training Artifacts

Each M7 challenger run writes:

```text
artifacts/training/m7/<run_id>/
|-- comparison_vs_champion.json
|-- dataset_manifest.json
|-- feature_columns.json
|-- fold_metrics.csv
|-- model.joblib
|-- oof_predictions.csv
|-- run_config.json
|-- run_manifest.json
`-- summary.json
```

The immutable local registry lives under:

```text
artifacts/registry/
|-- current.json
|-- history.jsonl
`-- models
    `-- <model_version>/
        |-- comparison_vs_champion.json
        |-- dataset_manifest.json
        |-- feature_columns.json
        |-- fold_metrics.csv
        |-- model.joblib
        |-- oof_predictions.csv
        |-- registry_entry.json
        |-- run_config.json
        |-- run_manifest.json
        `-- summary.json
```

Quick checks after promotion:

```powershell
Get-Content artifacts\registry\current.json
Get-Content artifacts\registry\history.jsonl
Get-ChildItem artifacts\registry\models
```

## Paper-Trading Artifacts

Each M5 run updates a rolling artifact directory:

```text
artifacts/paper_trading/
|-- by_asset_summary.csv
|-- by_regime_summary.csv
|-- closed_positions.csv
|-- latest_summary.json
`-- open_positions.csv
```

Quick checks after a run:

```powershell
Get-Content artifacts\paper_trading\latest_summary.json
Get-Content artifacts\paper_trading\by_asset_summary.csv
Get-Content artifacts\paper_trading\by_regime_summary.csv
Get-Content artifacts\paper_trading\open_positions.csv
Get-Content artifacts\paper_trading\closed_positions.csv
```

## Inspect Topics

Open Redpanda Console at [http://localhost:8080](http://localhost:8080).

You can also inspect the raw topics directly:

```powershell
docker exec -it streamalpha-redpanda rpk topic list -X brokers=redpanda:9092
docker exec -it streamalpha-redpanda rpk topic consume raw.ohlc -n 5 --offset start -X brokers=redpanda:9092
docker exec -it streamalpha-redpanda rpk topic consume raw.health -n 5 --offset start -X brokers=redpanda:9092
```

## Validate Feature Rows In PostgreSQL

```powershell
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS feature_rows FROM feature_ohlc;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT symbol, interval_begin, as_of_time, close_price, log_return_1, rsi_14, macd_line_12_26 FROM feature_ohlc ORDER BY interval_begin DESC LIMIT 10;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT symbol, COUNT(*) AS rows_per_symbol FROM feature_ohlc GROUP BY symbol ORDER BY symbol;"
```

You can still validate the raw OHLC source table if needed:

```powershell
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS raw_ohlc_rows FROM raw_ohlc;"
```

## Validate Paper-Trading Rows In PostgreSQL

```powershell
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS engine_state_rows FROM paper_engine_state;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS open_positions FROM paper_positions WHERE status = 'OPEN';"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS closed_positions FROM paper_positions WHERE status = 'CLOSED';"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c \"SELECT symbol, action, reason, fill_interval_begin, fill_price, quantity, fee FROM paper_trade_ledger ORDER BY fill_time DESC LIMIT 10;\"
```

## Warmup And Finalization Rules

- Feature rows are emitted only for finalized candles.
- A newer candle closes the previous candle but is not used in the previous candle's feature values.
- A stale current candle is finalized only when `now >= interval_end + FEATURE_FINALIZATION_GRACE_SECONDS`.
- The consumer bootstraps from the latest `FEATURE_BOOTSTRAP_CANDLES` raw OHLC rows per symbol before starting Kafka consumption.
- The effective bootstrap minimum is clamped to the feature-engine minimum warmup length so restart behavior remains safe.
- No row is emitted until all selected features are available. With the current feature set and standard MACD seeding, that means at least `26` finalized candles for a symbol stream.

## Known Limitations

- The feature consumer is OHLC-only in M2 and intentionally ignores `raw.trades`.
- Gaps are not forward-filled; features are computed only from the finalized candles that actually exist.
- Rolling standard deviation and z-score calculations use population standard deviation over the fixed 12-candle window.
- Bootstrap backfills the most recent feature rows idempotently through PostgreSQL upserts rather than relying on historical Kafka replay.
- M3 training drops the first 3 feature rows per symbol from the eligible labeled dataset so the official persistence baseline can be computed from the canonical source without pulling a second table.
- M3 training will stop with a clear error if `feature_ohlc` does not yet contain enough eligible labeled rows for the configured walk-forward split. With the current checked-in split config, that means at least `9` unique eligible timestamps.
- M5 fills signal-driven entries and exits at the next finalized candle open because only finalized canonical candles are processed.
- M5 uses the existing M4 inference service as authoritative and does not re-score features locally.
- M5 is long-only spot simulation with simple next-open fills plus barrier exits; it is intentionally not a live broker or advanced execution model.
- M7 promotion is fully file-based and explicit; there is no scheduler daemon or automatic retraining loop.
- M7 rollback only switches the promoted champion pointer and does not mutate old promoted snapshots.
- M9 resolves regimes from the latest saved M8 thresholds artifact by default; there is no online threshold fitting or adaptive retuning.
- This is still a single-broker local stack for development, not a highly available deployment.

## References

- Kraken public WebSocket v2 trades: [docs.kraken.com/api/docs/websocket-v2/trade/](https://docs.kraken.com/api/docs/websocket-v2/trade/)
- Kraken public WebSocket v2 OHLC: [docs.kraken.com/api/docs/websocket-v2/ohlc/](https://docs.kraken.com/api/docs/websocket-v2/ohlc/)
- Kraken public WebSocket v2 status and heartbeat: [docs.kraken.com/api/docs/websocket-v2/status/](https://docs.kraken.com/api/docs/websocket-v2/status/) and [docs.kraken.com/api/docs/websocket-v2/heartbeat/](https://docs.kraken.com/api/docs/websocket-v2/heartbeat/)
- Redpanda single-broker Docker example: [docs.redpanda.com/current/get-started/quick-start/](https://docs.redpanda.com/current/get-started/quick-start/)
