#!/usr/bin/env bash

set -euo pipefail

log() {
  printf '[m3-smoke] %s\n' "$1"
}

export PATH="/snap/bin:${PATH}"

cd /mnt/d/Github/Stream_Alpha

timeout_seconds="${M3_SMOKE_PG_TIMEOUT_SECONDS:-60}"
start_time="$(date +%s)"

if ! command -v docker >/dev/null 2>&1; then
  log "docker CLI was not found in PATH"
  exit 1
fi

log "waiting for PostgreSQL readiness (timeout: ${timeout_seconds}s)"
while true; do
  if docker exec streamalpha-postgres pg_isready -U streamalpha -d streamalpha >/dev/null 2>&1; then
    break
  fi

  now="$(date +%s)"
  elapsed="$((now - start_time))"
  if [ "$elapsed" -ge "$timeout_seconds" ]; then
    log "PostgreSQL did not become ready within ${timeout_seconds}s"
    docker ps --format 'table {{.Names}}\t{{.Status}}'
    exit 1
  fi

  log "still waiting for PostgreSQL (${elapsed}s elapsed)"
  sleep 1
done

log "PostgreSQL is ready; creating temporary smoke dataset"
python3 - <<'PY'
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import asyncpg

CYCLE = [100.0, 101.0, 102.0, 99.0, 98.0, 97.0]
SYMBOL_BASES = {
    "BTC/USD": 30000.0,
    "ETH/USD": 2000.0,
    "SOL/USD": 100.0,
}


async def main():
    conn = await asyncpg.connect(
        host="127.0.0.1",
        port=5432,
        user="streamalpha",
        password="change-me-local-only",
        database="streamalpha",
    )
    await conn.execute("DROP TABLE IF EXISTS feature_ohlc_smoke_m3")
    await conn.execute("CREATE TABLE feature_ohlc_smoke_m3 (LIKE feature_ohlc INCLUDING ALL)")

    start = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    total_rows = 60
    insert_sql = """
        INSERT INTO feature_ohlc_smoke_m3 (
            source_exchange,
            symbol,
            interval_minutes,
            interval_begin,
            interval_end,
            as_of_time,
            computed_at,
            raw_event_id,
            open_price,
            high_price,
            low_price,
            close_price,
            vwap,
            trade_count,
            volume,
            log_return_1,
            log_return_3,
            momentum_3,
            return_mean_12,
            return_std_12,
            realized_vol_12,
            rsi_14,
            macd_line_12_26,
            volume_mean_12,
            volume_std_12,
            volume_zscore_12,
            close_zscore_12,
            lag_log_return_1,
            lag_log_return_2,
            lag_log_return_3,
            created_at,
            updated_at
        )
        VALUES (
            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
            $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
            $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
            $31, $32
        )
    """

    for symbol, base in SYMBOL_BASES.items():
        for index in range(total_rows):
            interval_begin = start + timedelta(minutes=5 * index)
            interval_end = interval_begin + timedelta(minutes=5)
            phase = index % len(CYCLE)
            future_up = phase >= 3
            close_price = base + CYCLE[phase]
            signed_feature = 1.0 if future_up else -1.0
            row_values = (
                "kraken",
                symbol,
                5,
                interval_begin,
                interval_end,
                interval_end,
                interval_end,
                f"smoke-{symbol}-{index}",
                close_price - 0.25,
                close_price + 0.5,
                close_price - 0.5,
                close_price,
                close_price - 0.05,
                100 + phase,
                1000.0 + (phase * 25.0),
                0.02 * signed_feature,
                0.05 * signed_feature,
                0.03 * signed_feature,
                0.01 * signed_feature,
                0.02,
                0.05,
                35.0 if future_up else 65.0,
                1.5 * signed_feature,
                1000.0,
                50.0,
                1.25 * signed_feature,
                -1.5 if future_up else 1.5,
                0.015 * signed_feature,
                0.012 * signed_feature,
                0.010 * signed_feature,
                interval_end,
                interval_end,
            )
            await conn.execute(insert_sql, *row_values)

    await conn.close()

    config = json.loads(Path("configs/training.m3.json").read_text(encoding="utf-8"))
    config["source_table"] = "feature_ohlc_smoke_m3"
    Path("/tmp/training.m3.smoke.json").write_text(json.dumps(config), encoding="utf-8")


asyncio.run(main())
PY

log "running M3 training smoke run"
POSTGRES_HOST=127.0.0.1 python3 -m app.training --config /tmp/training.m3.smoke.json
latest_run="$(ls -td artifacts/training/m3/* | head -1)"
log "latest artifact directory: ${latest_run}"
cat "${latest_run}/summary.json"
