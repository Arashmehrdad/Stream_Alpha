# Configuration

Configuration comes from `.env`, `.env.example`, Compose interpolation, and environment parsing in `app/common/config.py` and `app/runtime/config.py`.

Do not commit real secrets. `.env.secrets.example` is present as a placeholder for secret-style configuration.

## Core Environment Variables

| Variable | Purpose | Visible default | Required | Used by |
| --- | --- | --- | --- | --- |
| `APP_NAME` | Application name used in logs/settings | `streamalpha` | Optional | common settings |
| `LOG_LEVEL` | Logging level | `INFO` | Optional | all Python services |
| `KRAKEN_WS_URL` | Kraken WebSocket URL | `wss://ws.kraken.com/v2` | Optional | producer |
| `KRAKEN_REST_OHLC_URL` | Kraken OHLC REST URL | `https://api.kraken.com/0/public/OHLC` | Optional | producer/backfill |
| `KRAKEN_SYMBOLS` | Comma-separated symbols | `BTC/USD,ETH/USD,SOL/USD` | Optional | producer/features/inference validation |
| `KRAKEN_OHLC_INTERVAL_MINUTES` | OHLC interval | `5` | Optional | producer/backfill |
| `KAFKA_BOOTSTRAP_SERVERS` | Broker connection string | `redpanda:9092` | Optional | producer/features |
| `KAFKA_CLIENT_ID` | Kafka client ID | `streamalpha-producer` | Optional | producer |
| `POSTGRES_HOST` | PostgreSQL host | `postgres` | Optional | services using DB |
| `POSTGRES_PORT` | PostgreSQL port | `5432` | Optional | services using DB |
| `POSTGRES_DB` | PostgreSQL database | `streamalpha` | Optional | services using DB |
| `POSTGRES_USER` | PostgreSQL user | `streamalpha` | Optional | services using DB |
| `POSTGRES_PASSWORD` | PostgreSQL password | `change-me-local-only` | Optional locally | services using DB |
| `TOPIC_RAW_TRADES` | Raw trades topic | `raw.trades` | Optional | producer/features |
| `TOPIC_RAW_OHLC` | Raw OHLC topic | `raw.ohlc` | Optional | producer/features |
| `TOPIC_RAW_HEALTH` | Raw health topic | `raw.health` | Optional | producer/ops scripts |
| `TABLE_RAW_TRADES` | Raw trades table | `raw_trades` | Optional | ingestion/training |
| `TABLE_RAW_OHLC` | Raw OHLC table | `raw_ohlc` | Optional | ingestion/features/training |
| `TABLE_FEATURE_OHLC` | Feature table | `feature_ohlc` | Optional | features/inference/training |
| `TABLE_PRODUCER_HEARTBEAT` | Producer heartbeat table | `producer_heartbeat` | Optional | producer/reliability |
| `PRODUCER_SERVICE_NAME` | Producer service name | `producer` | Optional | producer |
| `PRODUCER_HEARTBEAT_INTERVAL_SECONDS` | Producer heartbeat interval | `15` | Optional | producer |
| `FEATURE_CONSUMER_GROUP_ID` | Kafka consumer group | `streamalpha-feature-consumer` | Optional | features |
| `FEATURE_SERVICE_NAME` | Feature service name | `features` | Optional | features |
| `FEATURE_FINALIZATION_GRACE_SECONDS` | Feature finalization delay | `30` | Optional | features |
| `FEATURE_BOOTSTRAP_CANDLES` | Bootstrap candle count | `64` | Optional | features |
| `INFERENCE_MODEL_PATH` | Explicit inference model path | blank | Optional | inference |
| `INFERENCE_REGIME_THRESHOLDS_PATH` | Explicit regime thresholds path | blank | Optional | inference |
| `INFERENCE_REGIME_SIGNAL_POLICY_PATH` | Regime signal policy path | `configs/regime_signal_policy.json` | Optional | inference |
| `INFERENCE_SERVICE_NAME` | Inference service name | `inference` | Optional | inference |
| `INFERENCE_SIGNAL_BUY_PROB_UP` | Buy signal threshold | `0.55` | Optional | inference |
| `INFERENCE_SIGNAL_SELL_PROB_UP` | Sell signal threshold | `0.45` | Optional | inference |
| `INFERENCE_API_BASE_URL` | Dashboard/trader inference base URL | `http://127.0.0.1:8000` | Optional | dashboard/trader |
| `DASHBOARD_REFRESH_SECONDS` | Dashboard refresh interval | `15` | Optional | dashboard |
| `DASHBOARD_RECENT_TRADES_LIMIT` | Dashboard recent trade limit | `20` | Optional | dashboard |
| `DASHBOARD_RECENT_LEDGER_LIMIT` | Dashboard recent ledger limit | `20` | Optional | dashboard |
| `STREAMALPHA_RUNTIME_PROFILE` | Runtime profile: `dev`, `paper`, `shadow`, `live` | blank in `.env.example`; runtime helper default is paper in some code paths | Required for deployed startup validation | config-check/runtime |
| `STREAMALPHA_TRADING_CONFIG_PATH` | Trading config override | blank | Required for deployed profile unless profile default is used by validation | config-check/trader/inference metadata |
| `STREAMALPHA_STARTUP_REPORT_PATH` | Startup report output path | `artifacts/runtime/startup_report.json` | Optional | config-check/inference |
| `STREAMALPHA_ENABLE_LIVE` | Live startup arming flag | blank | Required only for live | config-check/live |
| `STREAMALPHA_LIVE_CONFIRM` | Live startup confirmation phrase | blank | Required only for live | config-check/live |
| `APCA_API_KEY_ID` | Alpaca API key ID | blank | Required only for live | trader/live validation |
| `APCA_API_SECRET_KEY` | Alpaca API secret | blank | Required only for live | trader/live validation |
| `ALPACA_BASE_URL` | Alpaca API base URL | blank | Required only for live | trader/live validation |
| `STREAMALPHA_LOCAL_TRAINING_TEMP_ROOT` | Local training temp directory override | repo-local default | Optional | training scripts/workdirs |
| `STREAMALPHA_VPS_HOST` | VPS host | blank | Optional | deployment scripts |
| `STREAMALPHA_VPS_USER` | VPS user | blank | Optional | deployment scripts |
| `STREAMALPHA_VPS_PASSWORD` | VPS password | blank | Optional | deployment scripts |
| `STREAMALPHA_VPS_PORT` | VPS SSH port | `22` | Optional | deployment scripts |
| `STREAMALPHA_VPS_APP_DIR` | Remote app directory | `~/stream_alpha_paper` | Optional | deployment scripts |
| `RECONNECT_INITIAL_DELAY_SECONDS` | Reconnect initial delay | `1` | Optional | long-running services |
| `RECONNECT_MAX_DELAY_SECONDS` | Reconnect max delay | `30` | Optional | long-running services |
| `RECONNECT_BACKOFF_MULTIPLIER` | Reconnect multiplier | `2.0` | Optional | long-running services |
| `RECONNECT_JITTER_SECONDS` | Reconnect jitter | `0.5` | Optional | long-running services |
| `REDPANDA_VERSION` | Redpanda image tag | `v25.3.10` | Optional | Compose |
| `REDPANDA_CONSOLE_VERSION` | Redpanda Console image tag | `v3.5.1` | Optional | Compose |
| `POSTGRES_VERSION` | Postgres image tag | `16-alpine` | Optional | Compose |
| `REDPANDA_CONSOLE_PORT` | Console host port | `8080` | Optional | Compose |
| `REDPANDA_EXTERNAL_KAFKA_PORT` | External Kafka host port | `19092` | Optional | Compose |
| `REDPANDA_EXTERNAL_ADMIN_PORT` | External Redpanda admin port | `19644` | Optional | Compose |
| `REDPANDA_EXTERNAL_SCHEMA_REGISTRY_PORT` | External schema registry port | `18081` | Optional | Compose |
| `REDPANDA_EXTERNAL_PANDAPROXY_PORT` | External Pandaproxy port | `18082` | Optional | Compose |
| `POSTGRES_EXTERNAL_PORT` | PostgreSQL host port | `5432` | Optional | Compose |

## Profile Config Files

Profile-specific trading configs are visible in `configs/`:

- `configs/paper_trading.paper.yaml`
- `configs/paper_trading.shadow.yaml`
- `configs/paper_trading.live.yaml`
- `configs/paper_trading.yaml`

The runtime helper maps `paper`, `shadow`, and `live` to the profile-specific files when profile defaults are enabled.

