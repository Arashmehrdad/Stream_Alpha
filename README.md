# Stream Alpha

Stream Alpha is a local-first crypto market data pipeline built with Docker Compose, Redpanda Community Edition, PostgreSQL, and Python services.

Milestone `M1` ingests Kraken public WebSocket v2 market data for `BTC/USD`, `ETH/USD`, and `SOL/USD`, normalizes `trade` and `ohlc` events, publishes them into Redpanda, and writes raw rows into PostgreSQL.

Milestone `M2` adds an OHLC-only feature consumer. It reads `raw.ohlc`, finalizes closed candles safely, computes a minimal rolling feature set using only current and past finalized candles, and upserts typed rows into PostgreSQL table `feature_ohlc`.

Milestone `M3` adds an offline training pipeline. It builds a labeled dataset from `feature_ohlc`, evaluates naive and learned classifiers with purged expanding walk-forward validation, and saves model plus evaluation artifacts under `artifacts/training/m3/<run_id>/`.

Milestone `M4` adds a minimal FastAPI inference API. It loads the accepted saved M3 artifact, reads the latest canonical feature row from `feature_ohlc`, and serves health, latest-feature, prediction, signal, and JSON metrics endpoints.

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
|   |-- training
|   |   |-- __init__.py
|   |   |-- __main__.py
|   |   |-- baselines.py
|   |   |-- dataset.py
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
|   |-- redpanda-console-config.yml
|   `-- training.m3.json
|-- docker
|   `-- producer.Dockerfile
|-- docker-compose.yml
|-- docs
|   `-- m1-overview.md
|-- requirements.txt
|-- scripts
|   |-- check-db.ps1
|   |-- check-topics.ps1
|   |-- m3-smoke-run.sh
|   `-- tail-producer.ps1
`-- tests
    |-- test_inference_api.py
    |-- test_inference_db.py
    |-- test_inference_model_loader.py
    |-- test_feature_engine.py
    |-- test_feature_state.py
    |-- test_normalizers.py
    |-- test_publish_smoke.py
    |-- test_training_labels.py
    `-- test_training_splits.py
```

## Services

- `redpanda`: local Redpanda broker for topics and consumer groups
- `redpanda-console`: topic and cluster inspection UI
- `postgres`: local PostgreSQL for raw and feature tables
- `producer`: Kraken public WebSocket v2 ingestion service from M1
- `features`: OHLC-only feature consumer from M2
- `inference`: FastAPI prediction API from M4, run directly from the repo with the accepted M3 artifact

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
| `INFERENCE_MODEL_PATH` | Path to the accepted saved M3 `model.joblib` artifact | empty |
| `INFERENCE_SERVICE_NAME` | Inference API service name used in logs and responses | `inference` |
| `INFERENCE_SIGNAL_BUY_PROB_UP` | BUY threshold for `prob_up` | `0.55` |
| `INFERENCE_SIGNAL_SELL_PROB_UP` | SELL threshold for `prob_up` | `0.45` |
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

### 9. Run the M4 inference API from the repo

Set `INFERENCE_MODEL_PATH` to the accepted saved M3 artifact first. For example:

```powershell
$env:INFERENCE_MODEL_PATH = "D:\Github\Stream_Alpha\artifacts\training\m3\<run_id>\model.joblib"
python -m app.inference
```

The API listens on `http://127.0.0.1:8000` by default.

### 10. Example inference requests

```powershell
curl "http://127.0.0.1:8000/health"
curl "http://127.0.0.1:8000/latest-features?symbol=BTC/USD"
curl "http://127.0.0.1:8000/predict?symbol=BTC/USD"
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
  "confidence": 0.61
}
```

Example `GET /signal` response:

```json
{
  "symbol": "BTC/USD",
  "signal": "BUY",
  "reason": "prob_up 0.6100 >= buy threshold 0.55",
  "prob_up": 0.61,
  "prob_down": 0.39,
  "confidence": 0.61,
  "predicted_class": "UP",
  "thresholds": {
    "buy_prob_up": 0.55,
    "sell_prob_up": 0.45
  },
  "row_id": "BTC/USD|2026-03-19T22:00:00Z",
  "as_of_time": "2026-03-19T22:05:00Z",
  "model_name": "logistic_regression"
}
```

## Tests And Lint

```powershell
python -m pytest
python -m pylint app tests
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
- This is still a single-broker local stack for development, not a highly available deployment.

## References

- Kraken public WebSocket v2 trades: [docs.kraken.com/api/docs/websocket-v2/trade/](https://docs.kraken.com/api/docs/websocket-v2/trade/)
- Kraken public WebSocket v2 OHLC: [docs.kraken.com/api/docs/websocket-v2/ohlc/](https://docs.kraken.com/api/docs/websocket-v2/ohlc/)
- Kraken public WebSocket v2 status and heartbeat: [docs.kraken.com/api/docs/websocket-v2/status/](https://docs.kraken.com/api/docs/websocket-v2/status/) and [docs.kraken.com/api/docs/websocket-v2/heartbeat/](https://docs.kraken.com/api/docs/websocket-v2/heartbeat/)
- Redpanda single-broker Docker example: [docs.redpanda.com/current/get-started/quick-start/](https://docs.redpanda.com/current/get-started/quick-start/)
