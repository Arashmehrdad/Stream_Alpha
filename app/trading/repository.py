"""PostgreSQL persistence for the Stream Alpha M5 paper trader."""

# pylint: disable=duplicate-code,too-many-lines

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date, datetime

import asyncpg

from app.common.time import to_rfc3339
from app.trading.schemas import (
    FeatureCandle,
    PaperEngineState,
    PaperPosition,
    PendingSignalState,
    RiskDecisionLogEntry,
    ServiceRiskState,
    TradeLedgerEntry,
)


POSITIONS_TABLE = "paper_positions"
LEDGER_TABLE = "paper_trade_ledger"
STATE_TABLE = "paper_engine_state"
RISK_STATE_TABLE = "paper_risk_state"
RISK_DECISIONS_TABLE = "paper_risk_decisions"

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _quote_identifier(identifier: str) -> str:
    if not _IDENTIFIER_RE.match(identifier):
        raise ValueError(f"Unsafe SQL identifier: {identifier}")
    return f'"{identifier}"'


def _quote_table_name(name: str) -> str:
    parts = name.split(".")
    if not 1 <= len(parts) <= 2:
        raise ValueError(f"Unsupported table name format: {name}")
    return ".".join(_quote_identifier(part) for part in parts)


def _build_index_name(table_name: str, suffix: str) -> str:
    return _quote_identifier(f"{table_name}_{suffix}")


class TradingRepository:
    """Repository for M5 state, positions, ledger rows, and canonical candles."""

    def __init__(self, dsn: str, source_table: str) -> None:
        self._dsn = dsn
        self._source_table = _quote_table_name(source_table)
        self._positions_table = _quote_table_name(POSITIONS_TABLE)
        self._ledger_table = _quote_table_name(LEDGER_TABLE)
        self._state_table = _quote_table_name(STATE_TABLE)
        self._risk_state_table = _quote_table_name(RISK_STATE_TABLE)
        self._risk_decisions_table = _quote_table_name(RISK_DECISIONS_TABLE)
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Open the repository pool and ensure trading tables exist."""
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=5)
        await self._ensure_schema()

    async def close(self) -> None:
        """Close the repository pool."""
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None

    async def load_engine_states(
        self,
        *,
        service_name: str,
        symbols: Sequence[str],
    ) -> dict[str, PaperEngineState]:
        """Load persisted per-symbol engine state and fill any missing defaults."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._state_table}
            WHERE service_name = $1 AND symbol = ANY($2::text[])
            """,
            service_name,
            list(symbols),
        )
        states = {
            str(row["symbol"]): _state_from_row(row)
            for row in rows
        }
        for symbol in symbols:
            states.setdefault(
                symbol,
                PaperEngineState(service_name=service_name, symbol=symbol),
            )
        return states

    async def save_engine_state(self, state: PaperEngineState) -> None:
        """Upsert one per-symbol engine state row."""
        pool = self._require_pool()
        pending = state.pending_signal
        await pool.execute(
            f"""
            INSERT INTO {self._state_table} (
                service_name,
                symbol,
                last_processed_interval_begin,
                cooldown_until_interval_begin,
                pending_signal_action,
                pending_signal_interval_begin,
                pending_signal_as_of_time,
                pending_signal_row_id,
                pending_signal_reason,
                pending_prob_up,
                pending_prob_down,
                pending_confidence,
                pending_predicted_class,
                pending_model_name,
                pending_regime_label,
                pending_approved_notional,
                pending_risk_outcome,
                pending_risk_reason_codes,
                updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                $16, $17, $18::text[], NOW()
            )
            ON CONFLICT (service_name, symbol)
            DO UPDATE SET
                last_processed_interval_begin = EXCLUDED.last_processed_interval_begin,
                cooldown_until_interval_begin = EXCLUDED.cooldown_until_interval_begin,
                pending_signal_action = EXCLUDED.pending_signal_action,
                pending_signal_interval_begin = EXCLUDED.pending_signal_interval_begin,
                pending_signal_as_of_time = EXCLUDED.pending_signal_as_of_time,
                pending_signal_row_id = EXCLUDED.pending_signal_row_id,
                pending_signal_reason = EXCLUDED.pending_signal_reason,
                pending_prob_up = EXCLUDED.pending_prob_up,
                pending_prob_down = EXCLUDED.pending_prob_down,
                pending_confidence = EXCLUDED.pending_confidence,
                pending_predicted_class = EXCLUDED.pending_predicted_class,
                pending_model_name = EXCLUDED.pending_model_name,
                pending_regime_label = EXCLUDED.pending_regime_label,
                pending_approved_notional = EXCLUDED.pending_approved_notional,
                pending_risk_outcome = EXCLUDED.pending_risk_outcome,
                pending_risk_reason_codes = EXCLUDED.pending_risk_reason_codes,
                updated_at = NOW()
            """,
            state.service_name,
            state.symbol,
            state.last_processed_interval_begin,
            state.cooldown_until_interval_begin,
            None if pending is None else pending.signal,
            None if pending is None else pending.signal_interval_begin,
            None if pending is None else pending.signal_as_of_time,
            None if pending is None else pending.row_id,
            None if pending is None else pending.reason,
            None if pending is None else pending.prob_up,
            None if pending is None else pending.prob_down,
            None if pending is None else pending.confidence,
            None if pending is None else pending.predicted_class,
            None if pending is None else pending.model_name,
            None if pending is None else pending.regime_label,
            None if pending is None else pending.approved_notional,
            None if pending is None else pending.risk_outcome,
            None if pending is None else list(pending.risk_reason_codes),
        )

    async def load_service_risk_state(
        self,
        *,
        service_name: str,
    ) -> ServiceRiskState | None:
        """Load the persisted M10 service-level risk state, if present."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT *
            FROM {self._risk_state_table}
            WHERE service_name = $1
            """,
            service_name,
        )
        if row is None:
            return None
        return _service_risk_state_from_row(row)

    async def save_service_risk_state(self, state: ServiceRiskState) -> None:
        """Upsert the service-level M10 risk state."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._risk_state_table} (
                service_name,
                trading_day,
                realized_pnl_today,
                equity_high_watermark,
                current_equity,
                loss_streak_count,
                loss_streak_cooldown_until_interval_begin,
                kill_switch_enabled,
                updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, NOW()
            )
            ON CONFLICT (service_name)
            DO UPDATE SET
                trading_day = EXCLUDED.trading_day,
                realized_pnl_today = EXCLUDED.realized_pnl_today,
                equity_high_watermark = EXCLUDED.equity_high_watermark,
                current_equity = EXCLUDED.current_equity,
                loss_streak_count = EXCLUDED.loss_streak_count,
                loss_streak_cooldown_until_interval_begin =
                    EXCLUDED.loss_streak_cooldown_until_interval_begin,
                kill_switch_enabled = EXCLUDED.kill_switch_enabled,
                updated_at = NOW()
            """,
            state.service_name,
            state.trading_day,
            state.realized_pnl_today,
            state.equity_high_watermark,
            state.current_equity,
            state.loss_streak_count,
            state.loss_streak_cooldown_until_interval_begin,
            state.kill_switch_enabled,
        )

    async def insert_risk_decision(self, entry: RiskDecisionLogEntry) -> None:
        """Persist one M10 risk-decision audit row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._risk_decisions_table} (
                service_name,
                symbol,
                signal,
                signal_interval_begin,
                signal_as_of_time,
                signal_row_id,
                outcome,
                reason_codes,
                requested_notional,
                approved_notional,
                available_cash,
                current_equity,
                current_symbol_exposure_notional,
                total_open_exposure_notional,
                realized_vol_12,
                confidence,
                regime_label,
                regime_run_id,
                trade_allowed
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8::text[], $9, $10, $11, $12, $13, $14,
                $15, $16, $17, $18, $19
            )
            """,
            entry.service_name,
            entry.symbol,
            entry.signal,
            entry.signal_interval_begin,
            entry.signal_as_of_time,
            entry.signal_row_id,
            entry.outcome,
            list(entry.reason_codes),
            entry.requested_notional,
            entry.approved_notional,
            entry.available_cash,
            entry.current_equity,
            entry.current_symbol_exposure_notional,
            entry.total_open_exposure_notional,
            entry.realized_vol_12,
            entry.confidence,
            entry.regime_label,
            entry.regime_run_id,
            entry.trade_allowed,
        )

    async def fetch_new_feature_rows(
        self,
        *,
        symbol: str,
        source_exchange: str,
        interval_minutes: int,
        last_processed_interval_begin: datetime | None,
    ) -> list[FeatureCandle]:
        """Load new finalized canonical feature rows for one symbol."""
        pool = self._require_pool()
        if last_processed_interval_begin is None:
            rows = await pool.fetch(
                f"""
                SELECT id, source_exchange, symbol, interval_minutes, interval_begin,
                       interval_end, as_of_time, raw_event_id, open_price,
                       high_price, low_price, close_price, realized_vol_12
                FROM {self._source_table}
                WHERE source_exchange = $1
                  AND symbol = $2
                  AND interval_minutes = $3
                ORDER BY as_of_time ASC, interval_begin ASC
                """,
                source_exchange,
                symbol,
                interval_minutes,
            )
        else:
            rows = await pool.fetch(
                f"""
                SELECT id, source_exchange, symbol, interval_minutes, interval_begin,
                       interval_end, as_of_time, raw_event_id, open_price,
                       high_price, low_price, close_price, realized_vol_12
                FROM {self._source_table}
                WHERE source_exchange = $1
                  AND symbol = $2
                  AND interval_minutes = $3
                  AND interval_begin > $4
                ORDER BY as_of_time ASC, interval_begin ASC
                """,
                source_exchange,
                symbol,
                interval_minutes,
                last_processed_interval_begin,
            )
        return [_candle_from_row(row) for row in rows]

    async def load_open_positions(self, service_name: str) -> dict[str, PaperPosition]:
        """Load currently open positions keyed by symbol."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._positions_table}
            WHERE service_name = $1 AND status = 'OPEN'
            ORDER BY entry_fill_interval_begin ASC
            """,
            service_name,
        )
        return {str(row["symbol"]): _position_from_row(row) for row in rows}

    async def insert_position(self, position: PaperPosition) -> int:
        """Insert a newly opened position and return its generated id."""
        pool = self._require_pool()
        return int(
            await pool.fetchval(
                f"""
                INSERT INTO {self._positions_table} (
                    service_name,
                    symbol,
                    status,
                    entry_signal_interval_begin,
                    entry_signal_as_of_time,
                    entry_signal_row_id,
                    entry_reason,
                    entry_model_name,
                    entry_prob_up,
                    entry_confidence,
                    entry_fill_interval_begin,
                    entry_fill_time,
                    entry_price,
                    quantity,
                    entry_notional,
                    entry_fee,
                    stop_loss_price,
                    take_profit_price,
                    entry_regime_label,
                    entry_approved_notional,
                    entry_risk_outcome,
                    entry_risk_reason_codes,
                    opened_at,
                    updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                    $21, $22::text[], $23, $24
                )
                RETURNING id
                """,
                position.service_name,
                position.symbol,
                position.status,
                position.entry_signal_interval_begin,
                position.entry_signal_as_of_time,
                position.entry_signal_row_id,
                position.entry_reason,
                position.entry_model_name,
                position.entry_prob_up,
                position.entry_confidence,
                position.entry_fill_interval_begin,
                position.entry_fill_time,
                position.entry_price,
                position.quantity,
                position.entry_notional,
                position.entry_fee,
                position.stop_loss_price,
                position.take_profit_price,
                position.entry_regime_label,
                position.entry_approved_notional,
                position.entry_risk_outcome,
                list(position.entry_risk_reason_codes),
                position.opened_at,
                position.updated_at,
            )
        )

    async def close_position(self, position: PaperPosition) -> None:
        """Persist the terminal state for an existing position."""
        if position.position_id is None:
            raise ValueError("Cannot close a position without a position_id")
        pool = self._require_pool()
        await pool.execute(
            f"""
            UPDATE {self._positions_table}
            SET
                status = $2,
                exit_reason = $3,
                exit_signal_interval_begin = $4,
                exit_signal_as_of_time = $5,
                exit_signal_row_id = $6,
                exit_model_name = $7,
                exit_prob_up = $8,
                exit_confidence = $9,
                exit_fill_interval_begin = $10,
                exit_fill_time = $11,
                exit_price = $12,
                exit_notional = $13,
                exit_fee = $14,
                realized_pnl = $15,
                realized_return = $16,
                entry_regime_label = $17,
                entry_approved_notional = $18,
                entry_risk_outcome = $19,
                entry_risk_reason_codes = $20::text[],
                exit_regime_label = $21,
                closed_at = $22,
                updated_at = $23
            WHERE id = $1
            """,
            position.position_id,
            position.status,
            position.exit_reason,
            position.exit_signal_interval_begin,
            position.exit_signal_as_of_time,
            position.exit_signal_row_id,
            position.exit_model_name,
            position.exit_prob_up,
            position.exit_confidence,
            position.exit_fill_interval_begin,
            position.exit_fill_time,
            position.exit_price,
            position.exit_notional,
            position.exit_fee,
            position.realized_pnl,
            position.realized_return,
            position.entry_regime_label,
            position.entry_approved_notional,
            position.entry_risk_outcome,
            list(position.entry_risk_reason_codes),
            position.exit_regime_label,
            position.closed_at,
            position.updated_at,
        )

    async def insert_ledger_entry(self, entry: TradeLedgerEntry) -> None:
        """Persist one simulated fill ledger row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._ledger_table} (
                service_name,
                position_id,
                symbol,
                action,
                reason,
                signal_interval_begin,
                signal_as_of_time,
                signal_row_id,
                model_name,
                prob_up,
                prob_down,
                confidence,
                regime_label,
                approved_notional,
                risk_outcome,
                risk_reason_codes,
                fill_interval_begin,
                fill_time,
                fill_price,
                quantity,
                notional,
                fee,
                slippage_bps,
                cash_flow,
                realized_pnl
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                $11, $12, $13, $14, $15, $16::text[], $17, $18, $19, $20, $21,
                $22, $23, $24, $25
            )
            """,
            entry.service_name,
            entry.position_id,
            entry.symbol,
            entry.action,
            entry.reason,
            entry.signal_interval_begin,
            entry.signal_as_of_time,
            entry.signal_row_id,
            entry.model_name,
            entry.prob_up,
            entry.prob_down,
            entry.confidence,
            entry.regime_label,
            entry.approved_notional,
            entry.risk_outcome,
            list(entry.risk_reason_codes),
            entry.fill_interval_begin,
            entry.fill_time,
            entry.fill_price,
            entry.quantity,
            entry.notional,
            entry.fee,
            entry.slippage_bps,
            entry.cash_flow,
            entry.realized_pnl,
        )

    async def load_cash_balance(
        self,
        *,
        service_name: str,
        initial_cash: float,
    ) -> float:
        """Reconstruct current cash from the persisted trade ledger."""
        pool = self._require_pool()
        cash_delta = float(
            await pool.fetchval(
                f"""
                SELECT COALESCE(SUM(cash_flow), 0.0)
                FROM {self._ledger_table}
                WHERE service_name = $1
                """,
                service_name,
            )
        )
        return initial_cash + cash_delta

    async def load_positions(
        self,
        *,
        service_name: str,
        status: str | None = None,
    ) -> list[PaperPosition]:
        """Load persisted positions, optionally filtered by status."""
        pool = self._require_pool()
        if status is None:
            rows = await pool.fetch(
                f"""
                SELECT *
                FROM {self._positions_table}
                WHERE service_name = $1
                ORDER BY entry_fill_interval_begin ASC, id ASC
                """,
                service_name,
            )
        else:
            rows = await pool.fetch(
                f"""
                SELECT *
                FROM {self._positions_table}
                WHERE service_name = $1 AND status = $2
                ORDER BY entry_fill_interval_begin ASC, id ASC
                """,
                service_name,
                status,
            )
        return [_position_from_row(row) for row in rows]

    async def load_latest_prices(
        self,
        *,
        source_exchange: str,
        interval_minutes: int,
        symbols: Sequence[str],
    ) -> dict[str, float]:
        """Load the latest close price per symbol from canonical features."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT DISTINCT ON (symbol)
                symbol,
                close_price
            FROM {self._source_table}
            WHERE source_exchange = $1
              AND interval_minutes = $2
              AND symbol = ANY($3::text[])
            ORDER BY symbol ASC, as_of_time DESC, interval_begin DESC
            """,
            source_exchange,
            interval_minutes,
            list(symbols),
        )
        return {str(row["symbol"]): float(row["close_price"]) for row in rows}

    async def _ensure_schema(self) -> None:
        pool = self._require_pool()
        open_position_index = _build_index_name(
            "paper_positions",
            "one_open_position_per_symbol_idx",
        )
        state_index = _build_index_name("paper_engine_state", "service_symbol_idx")
        ledger_index = _build_index_name("paper_trade_ledger", "service_fill_time_idx")
        risk_decisions_index = _build_index_name(
            "paper_risk_decisions",
            "service_signal_time_idx",
        )
        async with pool.acquire() as connection:
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._positions_table} (
                    id BIGSERIAL PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    status TEXT NOT NULL,
                    entry_signal_interval_begin TIMESTAMPTZ NOT NULL,
                    entry_signal_as_of_time TIMESTAMPTZ NOT NULL,
                    entry_signal_row_id TEXT NOT NULL,
                    entry_reason TEXT NOT NULL,
                    entry_model_name TEXT NOT NULL,
                    entry_prob_up DOUBLE PRECISION NOT NULL,
                    entry_confidence DOUBLE PRECISION NOT NULL,
                    entry_fill_interval_begin TIMESTAMPTZ NOT NULL,
                    entry_fill_time TIMESTAMPTZ NOT NULL,
                    entry_price DOUBLE PRECISION NOT NULL,
                    quantity DOUBLE PRECISION NOT NULL,
                    entry_notional DOUBLE PRECISION NOT NULL,
                    entry_fee DOUBLE PRECISION NOT NULL,
                    stop_loss_price DOUBLE PRECISION NOT NULL,
                    take_profit_price DOUBLE PRECISION NOT NULL,
                    entry_regime_label TEXT NULL,
                    entry_approved_notional DOUBLE PRECISION NULL,
                    entry_risk_outcome TEXT NULL,
                    entry_risk_reason_codes TEXT[] NULL,
                    exit_reason TEXT NULL,
                    exit_signal_interval_begin TIMESTAMPTZ NULL,
                    exit_signal_as_of_time TIMESTAMPTZ NULL,
                    exit_signal_row_id TEXT NULL,
                    exit_model_name TEXT NULL,
                    exit_prob_up DOUBLE PRECISION NULL,
                    exit_confidence DOUBLE PRECISION NULL,
                    exit_fill_interval_begin TIMESTAMPTZ NULL,
                    exit_fill_time TIMESTAMPTZ NULL,
                    exit_price DOUBLE PRECISION NULL,
                    exit_notional DOUBLE PRECISION NULL,
                    exit_fee DOUBLE PRECISION NULL,
                    realized_pnl DOUBLE PRECISION NULL,
                    realized_return DOUBLE PRECISION NULL,
                    exit_regime_label TEXT NULL,
                    opened_at TIMESTAMPTZ NOT NULL,
                    closed_at TIMESTAMPTZ NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._positions_table}
                ADD COLUMN IF NOT EXISTS entry_regime_label TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._positions_table}
                ADD COLUMN IF NOT EXISTS entry_approved_notional DOUBLE PRECISION NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._positions_table}
                ADD COLUMN IF NOT EXISTS entry_risk_outcome TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._positions_table}
                ADD COLUMN IF NOT EXISTS entry_risk_reason_codes TEXT[] NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._positions_table}
                ADD COLUMN IF NOT EXISTS exit_regime_label TEXT NULL
                """
            )
            await connection.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {open_position_index}
                ON {self._positions_table} (service_name, symbol)
                WHERE status = 'OPEN'
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._ledger_table} (
                    id BIGSERIAL PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    position_id BIGINT NULL REFERENCES {self._positions_table}(id),
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    signal_interval_begin TIMESTAMPTZ NULL,
                    signal_as_of_time TIMESTAMPTZ NULL,
                    signal_row_id TEXT NULL,
                    model_name TEXT NULL,
                    prob_up DOUBLE PRECISION NULL,
                    prob_down DOUBLE PRECISION NULL,
                    confidence DOUBLE PRECISION NULL,
                    regime_label TEXT NULL,
                    approved_notional DOUBLE PRECISION NULL,
                    risk_outcome TEXT NULL,
                    risk_reason_codes TEXT[] NULL,
                    fill_interval_begin TIMESTAMPTZ NOT NULL,
                    fill_time TIMESTAMPTZ NOT NULL,
                    fill_price DOUBLE PRECISION NOT NULL,
                    quantity DOUBLE PRECISION NOT NULL,
                    notional DOUBLE PRECISION NOT NULL,
                    fee DOUBLE PRECISION NOT NULL,
                    slippage_bps DOUBLE PRECISION NOT NULL,
                    cash_flow DOUBLE PRECISION NOT NULL,
                    realized_pnl DOUBLE PRECISION NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._ledger_table}
                ADD COLUMN IF NOT EXISTS regime_label TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._ledger_table}
                ADD COLUMN IF NOT EXISTS approved_notional DOUBLE PRECISION NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._ledger_table}
                ADD COLUMN IF NOT EXISTS risk_outcome TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._ledger_table}
                ADD COLUMN IF NOT EXISTS risk_reason_codes TEXT[] NULL
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {ledger_index}
                ON {self._ledger_table} (service_name, fill_time DESC)
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._state_table} (
                    service_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    last_processed_interval_begin TIMESTAMPTZ NULL,
                    cooldown_until_interval_begin TIMESTAMPTZ NULL,
                    pending_signal_action TEXT NULL,
                    pending_signal_interval_begin TIMESTAMPTZ NULL,
                    pending_signal_as_of_time TIMESTAMPTZ NULL,
                    pending_signal_row_id TEXT NULL,
                    pending_signal_reason TEXT NULL,
                    pending_prob_up DOUBLE PRECISION NULL,
                    pending_prob_down DOUBLE PRECISION NULL,
                    pending_confidence DOUBLE PRECISION NULL,
                    pending_predicted_class TEXT NULL,
                    pending_model_name TEXT NULL,
                    pending_regime_label TEXT NULL,
                    pending_approved_notional DOUBLE PRECISION NULL,
                    pending_risk_outcome TEXT NULL,
                    pending_risk_reason_codes TEXT[] NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (service_name, symbol)
                )
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._state_table}
                ADD COLUMN IF NOT EXISTS pending_regime_label TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._state_table}
                ADD COLUMN IF NOT EXISTS pending_approved_notional DOUBLE PRECISION NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._state_table}
                ADD COLUMN IF NOT EXISTS pending_risk_outcome TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._state_table}
                ADD COLUMN IF NOT EXISTS pending_risk_reason_codes TEXT[] NULL
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {state_index}
                ON {self._state_table} (service_name, symbol)
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._risk_state_table} (
                    service_name TEXT PRIMARY KEY,
                    trading_day DATE NOT NULL,
                    realized_pnl_today DOUBLE PRECISION NOT NULL,
                    equity_high_watermark DOUBLE PRECISION NOT NULL,
                    current_equity DOUBLE PRECISION NOT NULL,
                    loss_streak_count INTEGER NOT NULL,
                    loss_streak_cooldown_until_interval_begin TIMESTAMPTZ NULL,
                    kill_switch_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._risk_decisions_table} (
                    id BIGSERIAL PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    signal_interval_begin TIMESTAMPTZ NOT NULL,
                    signal_as_of_time TIMESTAMPTZ NOT NULL,
                    signal_row_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    reason_codes TEXT[] NOT NULL,
                    requested_notional DOUBLE PRECISION NOT NULL,
                    approved_notional DOUBLE PRECISION NOT NULL,
                    available_cash DOUBLE PRECISION NOT NULL,
                    current_equity DOUBLE PRECISION NOT NULL,
                    current_symbol_exposure_notional DOUBLE PRECISION NOT NULL,
                    total_open_exposure_notional DOUBLE PRECISION NOT NULL,
                    realized_vol_12 DOUBLE PRECISION NOT NULL,
                    confidence DOUBLE PRECISION NOT NULL,
                    regime_label TEXT NULL,
                    regime_run_id TEXT NULL,
                    trade_allowed BOOLEAN NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {risk_decisions_index}
                ON {self._risk_decisions_table} (service_name, signal_as_of_time DESC, id DESC)
                """
            )

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("TradingRepository has not been connected")
        return self._pool


def _candle_from_row(row: asyncpg.Record) -> FeatureCandle:
    return FeatureCandle(
        id=int(row["id"]),
        source_exchange=str(row["source_exchange"]),
        symbol=str(row["symbol"]),
        interval_minutes=int(row["interval_minutes"]),
        interval_begin=row["interval_begin"],
        interval_end=row["interval_end"],
        as_of_time=row["as_of_time"],
        raw_event_id=str(row["raw_event_id"]),
        open_price=float(row["open_price"]),
        high_price=float(row["high_price"]),
        low_price=float(row["low_price"]),
        close_price=float(row["close_price"]),
        realized_vol_12=float(row["realized_vol_12"]),
    )


def _state_from_row(row: asyncpg.Record) -> PaperEngineState:
    pending_signal = None
    if row["pending_signal_action"] is not None:
        pending_signal = PendingSignalState(
            signal=str(row["pending_signal_action"]),
            signal_interval_begin=row["pending_signal_interval_begin"],
            signal_as_of_time=row["pending_signal_as_of_time"],
            row_id=str(row["pending_signal_row_id"]),
            reason=str(row["pending_signal_reason"]),
            prob_up=float(row["pending_prob_up"]),
            prob_down=float(row["pending_prob_down"]),
            confidence=float(row["pending_confidence"]),
            predicted_class=str(row["pending_predicted_class"]),
            model_name=str(row["pending_model_name"]),
            regime_label=(
                None
                if row["pending_regime_label"] is None
                else str(row["pending_regime_label"])
            ),
            approved_notional=(
                None
                if row["pending_approved_notional"] is None
                else float(row["pending_approved_notional"])
            ),
            risk_outcome=(
                None
                if row["pending_risk_outcome"] is None
                else str(row["pending_risk_outcome"])
            ),
            risk_reason_codes=_text_array_to_tuple(row["pending_risk_reason_codes"]),
        )
    return PaperEngineState(
        service_name=str(row["service_name"]),
        symbol=str(row["symbol"]),
        last_processed_interval_begin=row["last_processed_interval_begin"],
        cooldown_until_interval_begin=row["cooldown_until_interval_begin"],
        pending_signal=pending_signal,
    )


def _position_from_row(row: asyncpg.Record) -> PaperPosition:
    return PaperPosition(
        service_name=str(row["service_name"]),
        symbol=str(row["symbol"]),
        status=str(row["status"]),
        entry_signal_interval_begin=row["entry_signal_interval_begin"],
        entry_signal_as_of_time=row["entry_signal_as_of_time"],
        entry_signal_row_id=str(row["entry_signal_row_id"]),
        entry_reason=str(row["entry_reason"]),
        entry_model_name=str(row["entry_model_name"]),
        entry_prob_up=float(row["entry_prob_up"]),
        entry_confidence=float(row["entry_confidence"]),
        entry_fill_interval_begin=row["entry_fill_interval_begin"],
        entry_fill_time=row["entry_fill_time"],
        entry_price=float(row["entry_price"]),
        quantity=float(row["quantity"]),
        entry_notional=float(row["entry_notional"]),
        entry_fee=float(row["entry_fee"]),
        stop_loss_price=float(row["stop_loss_price"]),
        take_profit_price=float(row["take_profit_price"]),
        entry_regime_label=(
            None if row["entry_regime_label"] is None else str(row["entry_regime_label"])
        ),
        entry_approved_notional=(
            None
            if row["entry_approved_notional"] is None
            else float(row["entry_approved_notional"])
        ),
        entry_risk_outcome=(
            None
            if row["entry_risk_outcome"] is None
            else str(row["entry_risk_outcome"])
        ),
        entry_risk_reason_codes=_text_array_to_tuple(row["entry_risk_reason_codes"]),
        position_id=int(row["id"]),
        exit_reason=None if row["exit_reason"] is None else str(row["exit_reason"]),
        exit_signal_interval_begin=row["exit_signal_interval_begin"],
        exit_signal_as_of_time=row["exit_signal_as_of_time"],
        exit_signal_row_id=None
        if row["exit_signal_row_id"] is None
        else str(row["exit_signal_row_id"]),
        exit_model_name=None if row["exit_model_name"] is None else str(row["exit_model_name"]),
        exit_prob_up=None if row["exit_prob_up"] is None else float(row["exit_prob_up"]),
        exit_confidence=None
        if row["exit_confidence"] is None
        else float(row["exit_confidence"]),
        exit_fill_interval_begin=row["exit_fill_interval_begin"],
        exit_fill_time=row["exit_fill_time"],
        exit_price=None if row["exit_price"] is None else float(row["exit_price"]),
        exit_notional=None if row["exit_notional"] is None else float(row["exit_notional"]),
        exit_fee=None if row["exit_fee"] is None else float(row["exit_fee"]),
        realized_pnl=None if row["realized_pnl"] is None else float(row["realized_pnl"]),
        realized_return=None
        if row["realized_return"] is None
        else float(row["realized_return"]),
        exit_regime_label=(
            None if row["exit_regime_label"] is None else str(row["exit_regime_label"])
        ),
        opened_at=row["opened_at"],
        closed_at=row["closed_at"],
        updated_at=row["updated_at"],
    )


def _service_risk_state_from_row(row: asyncpg.Record) -> ServiceRiskState:
    return ServiceRiskState(
        service_name=str(row["service_name"]),
        trading_day=_coerce_date(row["trading_day"]),
        realized_pnl_today=float(row["realized_pnl_today"]),
        equity_high_watermark=float(row["equity_high_watermark"]),
        current_equity=float(row["current_equity"]),
        loss_streak_count=int(row["loss_streak_count"]),
        loss_streak_cooldown_until_interval_begin=row["loss_streak_cooldown_until_interval_begin"],
        kill_switch_enabled=bool(row["kill_switch_enabled"]),
        updated_at=row["updated_at"],
    )


def ledger_rows_to_csv(entries: Sequence[TradeLedgerEntry]) -> list[dict[str, object]]:
    """Serialize ledger entries for optional artifact writing."""
    rows: list[dict[str, object]] = []
    for entry in entries:
        rows.append(
            {
                "service_name": entry.service_name,
                "position_id": entry.position_id,
                "symbol": entry.symbol,
                "action": entry.action,
                "reason": entry.reason,
                "signal_interval_begin": (
                    None
                    if entry.signal_interval_begin is None
                    else to_rfc3339(entry.signal_interval_begin)
                ),
                "signal_as_of_time": (
                    None
                    if entry.signal_as_of_time is None
                    else to_rfc3339(entry.signal_as_of_time)
                ),
                "signal_row_id": entry.signal_row_id,
                "model_name": entry.model_name,
                "prob_up": entry.prob_up,
                "prob_down": entry.prob_down,
                "confidence": entry.confidence,
                "regime_label": entry.regime_label,
                "approved_notional": entry.approved_notional,
                "risk_outcome": entry.risk_outcome,
                "risk_reason_codes": list(entry.risk_reason_codes),
                "fill_interval_begin": to_rfc3339(entry.fill_interval_begin),
                "fill_time": to_rfc3339(entry.fill_time),
                "fill_price": entry.fill_price,
                "quantity": entry.quantity,
                "notional": entry.notional,
                "fee": entry.fee,
                "slippage_bps": entry.slippage_bps,
                "cash_flow": entry.cash_flow,
                "realized_pnl": entry.realized_pnl,
            }
        )
    return rows


def _text_array_to_tuple(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(str(item) for item in value)


def _coerce_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value
