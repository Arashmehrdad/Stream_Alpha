# Stream Alpha Documentation

Stream Alpha is a local market-data, feature, inference, and trading-operations project. It runs a Docker Compose stack with Redpanda/Kafka-compatible messaging, PostgreSQL storage, Python services, a FastAPI inference API, and a Streamlit dashboard.

The repository also contains offline training, evaluation, model registry, and operator scripts. Runtime services and training workflows are intentionally documented separately because they use different dependency sets and have different risk profiles.

## What Stream Alpha Does

- Reads Kraken market data for configured symbols.
- Publishes raw trade, OHLC, and health events to Redpanda topics.
- Persists raw and feature data in PostgreSQL.
- Generates OHLC-derived features.
- Serves model predictions and signal metadata through the inference API.
- Runs paper, shadow, or live trading workflows when the corresponding runtime profile is configured.
- Provides dashboard and operational views for health, alerts, reliability, adaptation, ensemble, and continual-learning state.

## Main Components

- `redpanda`: Kafka-compatible broker.
- `redpanda-console`: broker UI.
- `postgres`: local PostgreSQL database.
- `config-check`: one-shot startup validation.
- `producer`: Kraken ingestion service.
- `features`: feature consumer and feature-table writer.
- `inference`: FastAPI model runtime.
- `trader`: paper/shadow/live trading runner.
- `dashboard`: Streamlit dashboard.

## Current Runtime Profile

The repo supports `dev`, `paper`, `shadow`, and `live` runtime profiles through `STREAMALPHA_RUNTIME_PROFILE`.

The local Docker proof recorded in `PLANS.md` used the `paper` profile. `.env.example` leaves `STREAMALPHA_RUNTIME_PROFILE` blank so each operator must choose the profile intentionally.

## Quick Links

- [Getting Started](getting-started.md)
- [Architecture](architecture.md)
- [Docker](docker.md)
- [Configuration](configuration.md)
- [Services](services.md)
- [Runtime vs Training](runtime-vs-training.md)
- [Model Runtime](model-runtime.md)
- [Training](training.md)
- [Project Report](report/index.md)
- [API](api.md)
- [Operations Runbook](operations-runbook.md)
- [Testing and Validation](testing-and-validation.md)
- [Troubleshooting](troubleshooting.md)
- [Release and Maintenance](release-and-maintenance.md)
