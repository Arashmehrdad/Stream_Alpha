# M1 Overview

## Scope

Milestone `M1` is intentionally narrow:

- public Kraken WebSocket v2 ingestion only
- `trade` and `ohlc` interval `5`
- `BTC/USD`, `ETH/USD`, and `SOL/USD`
- local Redpanda and PostgreSQL only
- no trading logic, feature engineering, model training, inference, or custom dashboard

## Event Flow

1. `producer` connects to `wss://ws.kraken.com/v2`
2. `producer` subscribes to public `trade` and `ohlc`
3. inbound Kraken messages are normalized into typed internal models
4. normalized events are published to:
   - `raw.trades`
   - `raw.ohlc`
   - `raw.health`
5. essential rows are persisted to:
   - `raw_trades`
   - `raw_ohlc`
   - `producer_heartbeat`

## Reliability Choices

- Exponential backoff with jitter for reconnects
- Structured JSON logging for all application logs
- Best-effort health emission so observability does not take the process down
- Bad payload isolation at the per-message-entry level
- Graceful shutdown through signal handling in the producer container

## Extension Points

- Add downstream consumers without changing the producer contract
- Introduce schema validation or schema registry later if needed
- Add private Kraken modules later behind separate env-driven credentials
