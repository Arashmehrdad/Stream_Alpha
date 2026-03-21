"""Focused PostgreSQL repository tests for M11 persistence fixes."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import asyncpg
import pytest

from app.reliability.schemas import RecoveryEvent, ReliabilityState, ServiceHeartbeat
from app.trading.repository import TradingRepository
from app.trading.schemas import PaperEngineState, PendingSignalState, TradeLedgerEntry


def _postgres_dsn() -> str:
    host = os.getenv("POSTGRES_HOST", "127.0.0.1").strip() or "127.0.0.1"
    if host == "postgres":
        host = "127.0.0.1"
    port = int(os.getenv("POSTGRES_PORT", "5432").strip())
    database = os.getenv("POSTGRES_DB", "streamalpha").strip() or "streamalpha"
    user = os.getenv("POSTGRES_USER", "streamalpha").strip() or "streamalpha"
    password = os.getenv("POSTGRES_PASSWORD", "change-me-local-only").strip()
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def test_save_engine_state_round_trip_includes_pending_order_fields() -> None:
    service_name = f"paper-trader-test-{uuid4().hex[:10]}"
    interval_begin = datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc)
    pending = PendingSignalState(
        signal="BUY",
        signal_interval_begin=interval_begin,
        signal_as_of_time=interval_begin + timedelta(minutes=5),
        row_id="BTC/USD|2026-03-21T12:00:00Z",
        reason="buy",
        prob_up=0.71,
        prob_down=0.29,
        confidence=0.71,
        predicted_class="UP",
        model_name="logistic_regression",
        regime_label="TREND_UP",
        regime_run_id="20260321T120000Z",
        approved_notional=900.0,
        risk_outcome="APPROVED",
        order_request_id=42,
        order_request_idempotency_key="v1|paper-trader|shadow|BTC/USD|BUY|row|fill|900.00000000",
        risk_reason_codes=("BUY_APPROVED", "VOLATILITY_SIZE_ADJUSTED"),
    )
    state = PaperEngineState(
        service_name=service_name,
        symbol="BTC/USD",
        execution_mode="shadow",
        last_processed_interval_begin=interval_begin,
        cooldown_until_interval_begin=interval_begin + timedelta(minutes=5),
        pending_signal=pending,
    )

    asyncio.run(_run_save_engine_state_round_trip_test(state))


async def _run_save_engine_state_round_trip_test(state: PaperEngineState) -> None:
    repository = TradingRepository(dsn=_postgres_dsn(), source_table="feature_ohlc")
    try:
        await repository.connect()
    except (OSError, asyncpg.CannotConnectNowError) as error:
        pytest.skip(f"PostgreSQL not reachable for repository round-trip test: {error}")
        return

    try:
        await repository.save_engine_state(state)
        loaded_states = await repository.load_engine_states(
            service_name=state.service_name,
            execution_mode=state.execution_mode,
            symbols=(state.symbol,),
        )
        loaded_state = loaded_states[state.symbol]
        assert loaded_state.execution_mode == "shadow"
        assert loaded_state.pending_signal is not None
        assert loaded_state.pending_signal.signal == "BUY"
        assert loaded_state.pending_signal.order_request_id == 42
        assert (
            loaded_state.pending_signal.order_request_idempotency_key
            == "v1|paper-trader|shadow|BTC/USD|BUY|row|fill|900.00000000"
        )
        assert loaded_state.pending_signal.risk_reason_codes == (
            "BUY_APPROVED",
            "VOLATILITY_SIZE_ADJUSTED",
        )
    finally:
        pool = repository._require_pool()  # pylint: disable=protected-access
        await pool.execute(
            """
            DELETE FROM paper_engine_state
            WHERE service_name = $1 AND execution_mode = $2
            """,
            state.service_name,
            state.execution_mode,
        )
        await repository.close()


def test_insert_ledger_entry_round_trip_includes_risk_reason_codes() -> None:
    service_name = f"paper-trader-test-{uuid4().hex[:10]}"
    fill_interval_begin = datetime(2026, 3, 21, 12, 5, tzinfo=timezone.utc)
    entry = TradeLedgerEntry(
        service_name=service_name,
        execution_mode="shadow",
        symbol="BTC/USD",
        action="BUY",
        reason="entry",
        signal_interval_begin=fill_interval_begin - timedelta(minutes=5),
        signal_as_of_time=fill_interval_begin,
        signal_row_id="BTC/USD|2026-03-21T12:00:00Z",
        model_name="logistic_regression",
        prob_up=0.71,
        prob_down=0.29,
        confidence=0.71,
        regime_label="TREND_UP",
        approved_notional=900.0,
        risk_outcome="MODIFIED",
        risk_reason_codes=("BUY_APPROVED", "VOLATILITY_SIZE_ADJUSTED"),
        fill_interval_begin=fill_interval_begin,
        fill_time=fill_interval_begin,
        fill_price=101.0,
        quantity=8.9,
        notional=900.0,
        fee=1.8,
        slippage_bps=5.0,
        cash_flow=-901.8,
    )

    asyncio.run(_run_insert_ledger_entry_round_trip_test(entry))


async def _run_insert_ledger_entry_round_trip_test(entry: TradeLedgerEntry) -> None:
    repository = TradingRepository(dsn=_postgres_dsn(), source_table="feature_ohlc")
    try:
        await repository.connect()
    except (OSError, asyncpg.CannotConnectNowError) as error:
        pytest.skip(f"PostgreSQL not reachable for repository round-trip test: {error}")
        return

    try:
        await repository.insert_ledger_entry(entry)
        pool = repository._require_pool()  # pylint: disable=protected-access
        row = await pool.fetchrow(
            """
            SELECT execution_mode, risk_outcome, risk_reason_codes
            FROM paper_trade_ledger
            WHERE service_name = $1 AND execution_mode = $2
            ORDER BY id DESC
            LIMIT 1
            """,
            entry.service_name,
            entry.execution_mode,
        )
        assert row is not None
        assert row["execution_mode"] == "shadow"
        assert row["risk_outcome"] == "MODIFIED"
        assert tuple(row["risk_reason_codes"]) == (
            "BUY_APPROVED",
            "VOLATILITY_SIZE_ADJUSTED",
        )
    finally:
        pool = repository._require_pool()  # pylint: disable=protected-access
        await pool.execute(
            """
            DELETE FROM paper_trade_ledger
            WHERE service_name = $1 AND execution_mode = $2
            """,
            entry.service_name,
            entry.execution_mode,
        )
        await repository.close()


def test_reliability_round_trip_supports_heartbeat_state_and_events() -> None:
    service_name = f"reliability-test-{uuid4().hex[:10]}"
    heartbeat_at = datetime(2026, 3, 21, 12, 15, tzinfo=timezone.utc)
    heartbeat = ServiceHeartbeat(
        service_name=service_name,
        component_name="signal_client",
        heartbeat_at=heartbeat_at,
        health_overall_status="HEALTHY",
        reason_code="HEALTH_HEALTHY",
        detail="initial heartbeat",
    )
    state = ReliabilityState(
        service_name=service_name,
        component_name="signal_client",
        health_overall_status="DEGRADED",
        freshness_status="STALE",
        breaker_state="HALF_OPEN",
        failure_count=2,
        success_count=0,
        last_heartbeat_at=heartbeat_at,
        last_failure_at=heartbeat_at,
        opened_at=heartbeat_at,
        reason_code="HEALTH_DEGRADED_FRESHNESS",
        detail="feature freshness degraded",
    )
    event = RecoveryEvent(
        service_name=service_name,
        component_name="signal_client",
        event_type="BREAKER_TRANSITION",
        event_time=heartbeat_at,
        reason_code="BREAKER_HALF_OPENED",
        health_overall_status="DEGRADED",
        freshness_status="STALE",
        breaker_state="HALF_OPEN",
        detail="breaker moved to half-open",
    )

    asyncio.run(_run_reliability_round_trip_test(heartbeat, state, event))


async def _run_reliability_round_trip_test(
    heartbeat: ServiceHeartbeat,
    state: ReliabilityState,
    event: RecoveryEvent,
) -> None:
    repository = TradingRepository(dsn=_postgres_dsn(), source_table="feature_ohlc")
    try:
        await repository.connect()
    except (OSError, asyncpg.CannotConnectNowError) as error:
        pytest.skip(f"PostgreSQL not reachable for reliability round-trip test: {error}")
        return

    try:
        stored_heartbeat = await repository.save_service_heartbeat(heartbeat)
        await repository.save_reliability_state(state)
        stored_event = await repository.insert_reliability_event(event)
        loaded_state = await repository.load_reliability_state(
            service_name=state.service_name,
            component_name=state.component_name,
        )
        latest_heartbeat = await repository.load_latest_service_heartbeat(
            service_name=heartbeat.service_name,
            component_name=heartbeat.component_name,
        )
        assert stored_heartbeat.heartbeat_id is not None
        assert latest_heartbeat is not None
        assert latest_heartbeat.reason_code == "HEALTH_HEALTHY"
        assert loaded_state is not None
        assert loaded_state.breaker_state == "HALF_OPEN"
        assert loaded_state.freshness_status == "STALE"
        assert stored_event.event_id is not None
        assert stored_event.reason_code == "BREAKER_HALF_OPENED"
    finally:
        pool = repository._require_pool()  # pylint: disable=protected-access
        await pool.execute(
            """
            DELETE FROM reliability_events
            WHERE service_name = $1 AND component_name = $2
            """,
            event.service_name,
            event.component_name,
        )
        await pool.execute(
            """
            DELETE FROM reliability_state
            WHERE service_name = $1 AND component_name = $2
            """,
            state.service_name,
            state.component_name,
        )
        await pool.execute(
            """
            DELETE FROM service_heartbeats
            WHERE service_name = $1 AND component_name = $2
            """,
            heartbeat.service_name,
            heartbeat.component_name,
        )
        await repository.close()
