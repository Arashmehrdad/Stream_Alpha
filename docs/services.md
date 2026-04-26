# Services

## `redpanda`

- Purpose: Kafka-compatible broker for raw market data topics.
- Image: `docker.redpanda.com/redpandadata/redpanda:${REDPANDA_VERSION:-v25.3.10}`.
- Ports: `19092`, `19644`, `18081`, `18082` by default.
- Volume: `redpanda-data:/var/lib/redpanda/data`.
- Health check: `rpk cluster health -X brokers=localhost:9092`.

## `redpanda-console`

- Purpose: Redpanda browser console.
- Image: `docker.redpanda.com/redpandadata/console:${REDPANDA_CONSOLE_VERSION:-v3.5.1}`.
- Depends on: `redpanda`.
- Port: `8080` by default.
- Config: `configs/redpanda-console-config.yml`.

## `postgres`

- Purpose: PostgreSQL storage for raw data, features, trading state, reliability state, alerts, and model-operation tables.
- Image: `postgres:${POSTGRES_VERSION:-16-alpine}`.
- Port: `5432` by default.
- Volume: `postgres-data:/var/lib/postgresql/data`.
- Health check: `pg_isready`.

## `config-check`

- Purpose: one-shot runtime validation before app services start.
- Command: `python -m app.runtime.validate`.
- Outputs: `artifacts/runtime/startup_report.json`.
- Depends on: environment, configs, model registry/artifacts, regime runtime, and live safety flags when profile is `live`.

## `producer`

- Purpose: Kraken ingestion service.
- Command: `python -m app.ingestion.main`.
- Inputs: Kraken WebSocket/REST settings, Redpanda, PostgreSQL.
- Outputs: `raw.trades`, `raw.ohlc`, `raw.health`, raw DB tables, producer heartbeat table.
- Depends on: `config-check`, `redpanda`, `postgres`.

## `features`

- Purpose: feature consumer and feature table writer.
- Command: `python -m app.features.main`.
- Inputs: raw OHLC topic/table and feature settings.
- Outputs: `feature_ohlc` table.
- Depends on: `config-check`, `redpanda`, `postgres`.

## `inference`

- Purpose: FastAPI runtime for features, predictions, signals, reliability, alerts, adaptation, ensemble, and continual-learning surfaces.
- Command: `python -m app.inference`.
- Port: `8000`.
- Inputs: `feature_ohlc`, model registry/artifacts, regime runtime, startup report.
- Outputs: API responses and request metrics.
- Depends on: `config-check`, `postgres`.
- Health check: `http://127.0.0.1:8000/health`.

## `trader`

- Purpose: trading runner for the active runtime profile.
- Command: `python scripts/run_paper_trader.py`.
- Inputs: trading config, inference API, PostgreSQL.
- Outputs: positions, ledger, risk, decision trace, and order state tables.
- Depends on: `config-check`, `inference`, `postgres`.

## `dashboard`

- Purpose: Streamlit operational dashboard.
- Command: `streamlit run dashboards/streamlit_app.py --server.address=0.0.0.0 --server.port=8501`.
- Port: `8501`.
- Inputs: inference API, PostgreSQL, artifacts.
- Depends on: `config-check`, `inference`, `postgres`.

