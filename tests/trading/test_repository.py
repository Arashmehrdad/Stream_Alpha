"""Focused PostgreSQL repository tests for M11 persistence fixes."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import asyncpg
import pytest

from app.explainability.schemas import (
    DecisionTraceBlockedTrade,
    DecisionTracePayload,
    DecisionTracePortfolioContext,
    DecisionTracePrediction,
    DecisionTraceRisk,
    DecisionTraceServiceRiskState,
    DecisionTraceSignal,
    PredictionExplanation,
    SignalExplanation,
    ThresholdSnapshot,
)
from app.reliability.schemas import RecoveryEvent, ReliabilityState, ServiceHeartbeat
from app.trading.repository import TradingRepository
from app.trading.schemas import (
    DecisionTraceRecord,
    OrderRequest,
    PaperEngineState,
    PendingSignalState,
    RiskDecisionLogEntry,
    TradeLedgerEntry,
)


def _postgres_dsn() -> str:
    host = os.getenv("POSTGRES_HOST", "127.0.0.1").strip() or "127.0.0.1"
    if host == "postgres":
        host = "127.0.0.1"
    port = int(os.getenv("POSTGRES_PORT", "5432").strip())
    database = os.getenv("POSTGRES_DB", "streamalpha").strip() or "streamalpha"
    user = os.getenv("POSTGRES_USER", "streamalpha").strip() or "streamalpha"
    password = os.getenv("POSTGRES_PASSWORD", "change-me-local-only").strip()
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def _decision_trace_payload(
    *,
    service_name: str,
    execution_mode: str,
    signal_row_id: str,
) -> DecisionTracePayload:
    return DecisionTracePayload(
        schema_version="m14_decision_trace_v1",
        service_name=service_name,
        execution_mode=execution_mode,
        symbol="BTC/USD",
        signal_row_id=signal_row_id,
        signal_interval_begin="2026-03-21T12:00:00Z",
        signal_as_of_time="2026-03-21T12:05:00Z",
        model_name="logistic_regression",
        model_version="m3-20260321T120000Z",
        prediction=DecisionTracePrediction(
            model_name="logistic_regression",
            model_version="m3-20260321T120000Z",
            prob_up=0.71,
            prob_down=0.29,
            confidence=0.71,
            predicted_class="UP",
            prediction_explanation=PredictionExplanation(
                method="ONE_AT_A_TIME_REFERENCE_ABLATION",
                available=True,
                reason_code="EXPLAINABILITY_AVAILABLE",
                summary_text="deterministic top features",
                reference_vector_path="artifacts/explainability/m3-20260321T120000Z/reference.json",
                reference_vector_source="PERSISTED_ARTIFACT",
                explainable_feature_count=3,
                top_feature_count=2,
            ),
        ),
        signal=DecisionTraceSignal(
            signal="BUY",
            reason="prob_up 0.7100 >= buy threshold 0.54",
            signal_status="MODEL_SIGNAL",
            decision_source="model",
            signal_explanation=SignalExplanation(
                signal="BUY",
                available=True,
                reason_code="SIGNAL_EXPLANATION_MODEL_DECISION",
                summary_text="BUY came from the M4 model path.",
                trade_allowed=True,
                decision_source="model",
            ),
        ),
        threshold_snapshot=ThresholdSnapshot(
            buy_prob_up=0.54,
            sell_prob_up=0.44,
            allow_new_long_entries=True,
            regime_label="TREND_UP",
            regime_run_id="20260321T120000Z",
            regime_artifact_path="artifacts/regime/m8/20260321T120000Z/thresholds.json",
            high_vol_threshold=0.05,
            trend_abs_threshold=0.02,
        ),
    )


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
        pending_decision_trace_id=77,
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
        assert loaded_state.pending_decision_trace_id == 77
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


def test_decision_trace_round_trip_supports_enrichment_and_linkage() -> None:
    service_name = f"paper-trader-trace-{uuid4().hex[:10]}"
    signal_row_id = "BTC/USD|2026-03-21T12:00:00Z"
    trace = DecisionTraceRecord(
        service_name=service_name,
        execution_mode="shadow",
        symbol="BTC/USD",
        signal="BUY",
        signal_interval_begin=datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc),
        signal_as_of_time=datetime(2026, 3, 21, 12, 5, tzinfo=timezone.utc),
        signal_row_id=signal_row_id,
        model_name="logistic_regression",
        model_version="m3-20260321T120000Z",
        payload=_decision_trace_payload(
            service_name=service_name,
            execution_mode="shadow",
            signal_row_id=signal_row_id,
        ),
    )

    asyncio.run(_run_decision_trace_round_trip_test(trace))


async def _run_decision_trace_round_trip_test(trace: DecisionTraceRecord) -> None:
    repository = TradingRepository(dsn=_postgres_dsn(), source_table="feature_ohlc")
    try:
        await repository.connect()
    except (OSError, asyncpg.CannotConnectNowError) as error:
        pytest.skip(f"PostgreSQL not reachable for repository round-trip test: {error}")
        return

    stored_trace = None
    try:
        stored_trace = await repository.ensure_decision_trace(trace)
        enriched_trace = await repository.update_decision_trace(
            replace(
                stored_trace,
                risk_outcome="BLOCKED",
                payload=stored_trace.payload.model_copy(
                    update={
                        "risk": DecisionTraceRisk(
                            outcome="BLOCKED",
                            primary_reason_code="TRADE_NOT_ALLOWED",
                            reason_codes=["TRADE_NOT_ALLOWED"],
                            reason_texts=[
                                "The resolved regime or M4 signal policy does not "
                                "allow opening a new long trade."
                            ],
                            requested_notional=2495.00998,
                            approved_notional=0.0,
                            portfolio_context=DecisionTracePortfolioContext(
                                available_cash=10_000.0,
                                open_position_count=0,
                                current_equity=10_000.0,
                                total_open_exposure_notional=0.0,
                                current_symbol_exposure_notional=0.0,
                            ),
                            service_risk_state=DecisionTraceServiceRiskState(
                                trading_day="2026-03-21",
                                realized_pnl_today=0.0,
                                equity_high_watermark=10_000.0,
                                current_equity=10_000.0,
                                loss_streak_count=0,
                                kill_switch_enabled=False,
                            ),
                        ),
                        "blocked_trade": DecisionTraceBlockedTrade(
                            blocked_stage="risk",
                            reason_code="TRADE_NOT_ALLOWED",
                            reason_texts=[
                                "The resolved regime or M4 signal policy does not "
                                "allow opening a new long trade."
                            ],
                        ),
                    },
                    deep=True,
                ),
            )
        )
        loaded_trace = await repository.load_decision_trace(
            decision_trace_id=enriched_trace.decision_trace_id,
        )
        assert loaded_trace is not None
        assert loaded_trace.payload.risk is not None
        assert loaded_trace.payload.risk.outcome == "BLOCKED"
        assert loaded_trace.payload.blocked_trade is not None
        assert loaded_trace.payload.blocked_trade.blocked_stage == "risk"

        stored_request = await repository.ensure_order_request(
            OrderRequest(
                service_name=trace.service_name,
                execution_mode=trace.execution_mode,
                symbol=trace.symbol,
                action="BUY",
                signal_interval_begin=trace.signal_interval_begin,
                signal_as_of_time=trace.signal_as_of_time,
                signal_row_id=trace.signal_row_id,
                target_fill_interval_begin=trace.signal_interval_begin + timedelta(minutes=5),
                requested_notional=2495.00998,
                approved_notional=0.0,
                idempotency_key=f"trace-test|{trace.service_name}",
                model_name=trace.model_name,
                model_version=trace.model_version,
                confidence=0.71,
                regime_label="TREND_UP",
                regime_run_id="20260321T120000Z",
                risk_outcome="BLOCKED",
                risk_reason_codes=("TRADE_NOT_ALLOWED",),
                decision_trace_id=enriched_trace.decision_trace_id,
            )
        )
        assert stored_request.decision_trace_id == enriched_trace.decision_trace_id
        assert stored_request.model_version == trace.model_version

        await repository.insert_risk_decision(
            RiskDecisionLogEntry(
                service_name=trace.service_name,
                execution_mode=trace.execution_mode,
                symbol=trace.symbol,
                signal="BUY",
                signal_interval_begin=trace.signal_interval_begin,
                signal_as_of_time=trace.signal_as_of_time,
                signal_row_id=trace.signal_row_id,
                outcome="BLOCKED",
                reason_codes=("TRADE_NOT_ALLOWED",),
                requested_notional=2495.00998,
                approved_notional=0.0,
                available_cash=10_000.0,
                current_equity=10_000.0,
                current_symbol_exposure_notional=0.0,
                total_open_exposure_notional=0.0,
                realized_vol_12=0.01,
                confidence=0.71,
                regime_label="TREND_UP",
                regime_run_id="20260321T120000Z",
                trade_allowed=False,
                decision_trace_id=enriched_trace.decision_trace_id,
                model_version=trace.model_version,
            )
        )
        pool = repository._require_pool()  # pylint: disable=protected-access
        linked_risk_row = await pool.fetchrow(
            """
            SELECT decision_trace_id, model_version
            FROM paper_risk_decisions
            WHERE service_name = $1 AND execution_mode = $2
            ORDER BY id DESC
            LIMIT 1
            """,
            trace.service_name,
            trace.execution_mode,
        )
        assert linked_risk_row is not None
        assert linked_risk_row["decision_trace_id"] == enriched_trace.decision_trace_id
        assert linked_risk_row["model_version"] == trace.model_version
    finally:
        pool = repository._require_pool()  # pylint: disable=protected-access
        await pool.execute(
            """
            DELETE FROM execution_order_requests
            WHERE service_name = $1 AND execution_mode = $2
            """,
            trace.service_name,
            trace.execution_mode,
        )
        await pool.execute(
            """
            DELETE FROM paper_risk_decisions
            WHERE service_name = $1 AND execution_mode = $2
            """,
            trace.service_name,
            trace.execution_mode,
        )
        await pool.execute(
            """
            DELETE FROM decision_traces
            WHERE service_name = $1 AND execution_mode = $2
            """,
            trace.service_name,
            trace.execution_mode,
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
