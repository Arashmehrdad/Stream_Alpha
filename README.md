# Stream Alpha

Milestone `M1` delivers a local, Docker Compose based crypto market-data ingestion stack. It connects to Kraken public WebSocket v2, normalizes `trade` and `ohlc` messages for `BTC/USD`, `ETH/USD`, and `SOL/USD`, publishes them into Redpanda, and persists essential rows into PostgreSQL.

Kraken authentication is intentionally not used in this milestone. The producer only connects to the public endpoint `wss://ws.kraken.com/v2`.

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
|   `-- ingestion
|       |-- __init__.py
|       |-- __main__.py
|       |-- db.py
|       |-- kraken.py
|       |-- main.py
|       |-- normalizers.py
|       |-- publisher.py
|       `-- service.py
|-- configs
|   `-- redpanda-console-config.yml
|-- docker
|   `-- producer.Dockerfile
|-- docker-compose.yml
|-- docs
|   `-- m1-overview.md
|-- requirements.txt
|-- scripts
|   |-- check-db.ps1
|   |-- check-topics.ps1
|   `-- tail-producer.ps1
`-- tests
    |-- test_normalizers.py
    `-- test_publish_smoke.py
```

## Services

- `redpanda`: single-node Redpanda Community Edition broker for local development.
- `redpanda-console`: the only UI in M1 for topic inspection and cluster visibility.
- `postgres`: local PostgreSQL instance for essential raw persistence.
- `producer`: Python service that ingests, normalizes, publishes, and writes heartbeats.

## Key Design Decisions

- Config is centralized in `app/common/config.py` and sourced from environment variables only.
- Typed internal event models live in `app/common/models.py`.
- Table names and topic names are configurable from environment variables.
- PostgreSQL schema creation is owned by the producer so table naming can stay configurable.
- The producer treats malformed payloads as isolated events: it logs them, emits a health event, and keeps running.
- Reconnect logic uses exponential backoff with jitter.
- Logging is structured JSON so container logs are machine-friendly from day one.

## Environment Variables

Copy `.env.example` to `.env` before running the stack.

| Variable | Purpose | Default |
| --- | --- | --- |
| `APP_NAME` | Logical app name embedded in events | `streamalpha` |
| `LOG_LEVEL` | Python log level | `INFO` |
| `KRAKEN_WS_URL` | Kraken public WebSocket v2 endpoint | `wss://ws.kraken.com/v2` |
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
| `TABLE_RAW_OHLC` | OHLC table name | `raw_ohlc` |
| `TABLE_PRODUCER_HEARTBEAT` | Heartbeat table name | `producer_heartbeat` |
| `PRODUCER_SERVICE_NAME` | Service name for heartbeat rows | `producer` |
| `PRODUCER_HEARTBEAT_INTERVAL_SECONDS` | Producer heartbeat cadence | `15` |
| `RECONNECT_INITIAL_DELAY_SECONDS` | First reconnect delay | `1` |
| `RECONNECT_MAX_DELAY_SECONDS` | Reconnect delay cap | `30` |
| `RECONNECT_BACKOFF_MULTIPLIER` | Backoff multiplier | `2.0` |
| `RECONNECT_JITTER_SECONDS` | Random jitter added to backoff | `0.5` |

## Exact Commands To Run

### 1. Prepare the environment

```powershell
Copy-Item .env.example .env
```

### 2. Start the full local stack

```powershell
docker compose up --build -d
```

### 3. Follow the producer logs

```powershell
docker compose logs -f producer
```

### 4. Run the test suite locally

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

### 5. Run pylint

```powershell
python -m pylint app tests
```

## How To Inspect Topics

Open Redpanda Console at [http://localhost:8080](http://localhost:8080).

You can also inspect topics directly with `rpk` inside the broker container:

```powershell
docker exec -it streamalpha-redpanda rpk topic list -X brokers=redpanda:9092
docker exec -it streamalpha-redpanda rpk topic consume raw.trades -n 5 --offset start -X brokers=redpanda:9092
docker exec -it streamalpha-redpanda rpk topic consume raw.ohlc -n 5 --offset start -X brokers=redpanda:9092
docker exec -it streamalpha-redpanda rpk topic consume raw.health -n 5 --offset start -X brokers=redpanda:9092
```

## How To Verify Database Rows

```powershell
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS trade_rows FROM raw_trades;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT COUNT(*) AS ohlc_rows FROM raw_ohlc;"
docker exec -it streamalpha-postgres psql -U streamalpha -d streamalpha -c "SELECT service_name, status, last_event_at FROM producer_heartbeat;"
```

## Milestone Acceptance Checklist

- [ ] `docker compose up --build -d` starts `redpanda`, `redpanda-console`, `postgres`, and `producer`
- [ ] producer connects to Kraken public WebSocket v2 without API keys
- [ ] producer subscribes to `trade` and `ohlc` interval `5` for `BTC/USD`, `ETH/USD`, and `SOL/USD`
- [ ] normalized events arrive in `raw.trades`, `raw.ohlc`, and `raw.health`
- [ ] essential rows appear in `raw_trades`, `raw_ohlc`, and `producer_heartbeat`
- [ ] malformed payloads do not crash the producer
- [ ] reconnect logic resumes ingestion after a dropped websocket session
- [ ] `python -m pytest` passes locally

## Known Limitations

- This is a single-broker local deployment for development, not a highly available production cluster.
- The producer writes to Redpanda and PostgreSQL separately, so cross-sink atomicity is not guaranteed.
- `trade` snapshots are enabled to seed initial flow quickly; downstream consumers should expect both snapshot and update message types.
- No REST backfill, feature engineering, inference, trading logic, or custom dashboard is included in M1.

## References

- Kraken public WebSocket v2 trades: [docs.kraken.com/api/docs/websocket-v2/trade/](https://docs.kraken.com/api/docs/websocket-v2/trade/)
- Kraken public WebSocket v2 OHLC: [docs.kraken.com/api/docs/websocket-v2/ohlc/](https://docs.kraken.com/api/docs/websocket-v2/ohlc/)
- Kraken public WebSocket v2 status and heartbeat: [docs.kraken.com/api/docs/websocket-v2/status/](https://docs.kraken.com/api/docs/websocket-v2/status/) and [docs.kraken.com/api/docs/websocket-v2/heartbeat/](https://docs.kraken.com/api/docs/websocket-v2/heartbeat/)
- Redpanda single-broker Docker example: [docs.redpanda.com/current/get-started/quick-start/](https://docs.redpanda.com/current/get-started/quick-start/)
- Redpanda Console Docker config example: [docs.redpanda.com/25.2/console/config/configure-console/](https://docs.redpanda.com/25.2/console/config/configure-console/)
