"""Read-only data access helpers for M18 evaluation generation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime

import asyncpg

from app.explainability.schemas import DecisionTracePayload
from app.reliability.schemas import RecoveryEvent, ServiceHeartbeat
from app.trading.repository import (
    DECISION_TRACES_TABLE,
    HEARTBEATS_TABLE,
    LEDGER_TABLE,
    ORDER_EVENTS_TABLE,
    POSITIONS_TABLE,
    RELIABILITY_EVENTS_TABLE,
)
from app.trading.schemas import (
    DecisionTraceRecord,
    OrderLifecycleEvent,
    PaperPosition,
    TradeLedgerEntry,
)


def _json_object(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


class EvaluationRepository:
    """Read-only repository for M18 evaluation loading."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        """Open the read-only connection pool."""
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(self._dsn, min_size=1, max_size=3)

    async def close(self) -> None:
        """Close the pool."""
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None

    async def load_decision_traces(
        self,
        *,
        service_name: str,
        execution_modes: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> list[DecisionTraceRecord]:
        """Load decision traces in the requested window across modes."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {DECISION_TRACES_TABLE}
            WHERE service_name = $1
              AND execution_mode = ANY($2::text[])
              AND signal_as_of_time >= $3
              AND signal_as_of_time <= $4
            ORDER BY execution_mode ASC, signal_as_of_time ASC, id ASC
            """,
            service_name,
            list(execution_modes),
            start,
            end,
        )
        return [_decision_trace_from_row(row) for row in rows]

    async def load_order_events_for_trace_ids(
        self,
        *,
        service_name: str,
        execution_modes: Sequence[str],
        decision_trace_ids: Sequence[int],
    ) -> list[OrderLifecycleEvent]:
        """Load lifecycle events linked to one set of decision traces."""
        if not decision_trace_ids:
            return []
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {ORDER_EVENTS_TABLE}
            WHERE service_name = $1
              AND execution_mode = ANY($2::text[])
              AND decision_trace_id = ANY($3::int[])
            ORDER BY execution_mode ASC, event_time ASC, id ASC
            """,
            service_name,
            list(execution_modes),
            list(decision_trace_ids),
        )
        return [_order_event_from_row(row) for row in rows]

    async def load_positions_for_trace_ids(
        self,
        *,
        service_name: str,
        execution_modes: Sequence[str],
        decision_trace_ids: Sequence[int],
    ) -> list[PaperPosition]:
        """Load positions linked to the requested decision traces."""
        if not decision_trace_ids:
            return []
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {POSITIONS_TABLE}
            WHERE service_name = $1
              AND execution_mode = ANY($2::text[])
              AND (
                entry_decision_trace_id = ANY($3::int[])
                OR exit_decision_trace_id = ANY($3::int[])
              )
            ORDER BY execution_mode ASC, entry_signal_as_of_time ASC, id ASC
            """,
            service_name,
            list(execution_modes),
            list(decision_trace_ids),
        )
        return [_position_from_row(row) for row in rows]

    async def load_ledger_entries_for_trace_ids(
        self,
        *,
        service_name: str,
        execution_modes: Sequence[str],
        decision_trace_ids: Sequence[int],
    ) -> list[TradeLedgerEntry]:
        """Load ledger rows linked to the requested decision traces."""
        if not decision_trace_ids:
            return []
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {LEDGER_TABLE}
            WHERE service_name = $1
              AND execution_mode = ANY($2::text[])
              AND decision_trace_id = ANY($3::int[])
            ORDER BY execution_mode ASC, fill_time ASC
            """,
            service_name,
            list(execution_modes),
            list(decision_trace_ids),
        )
        return [_ledger_entry_from_row(row) for row in rows]

    async def load_service_heartbeats(
        self,
        *,
        service_name: str,
        start: datetime,
        end: datetime,
    ) -> list[ServiceHeartbeat]:
        """Load M13 service heartbeats in the evaluation window."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {HEARTBEATS_TABLE}
            WHERE service_name = $1
              AND heartbeat_at >= $2
              AND heartbeat_at <= $3
            ORDER BY component_name ASC, heartbeat_at ASC, id ASC
            """,
            service_name,
            start,
            end,
        )
        return [_heartbeat_from_row(row) for row in rows]

    async def load_reliability_events(
        self,
        *,
        service_name: str,
        start: datetime,
        end: datetime,
    ) -> list[RecoveryEvent]:
        """Load canonical M13 recovery events in the evaluation window."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {RELIABILITY_EVENTS_TABLE}
            WHERE service_name = $1
              AND event_time >= $2
              AND event_time <= $3
            ORDER BY component_name ASC, event_time ASC, id ASC
            """,
            service_name,
            start,
            end,
        )
        return [_recovery_event_from_row(row) for row in rows]

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("EvaluationRepository has not been connected")
        return self._pool


def _decision_trace_from_row(row: asyncpg.Record) -> DecisionTraceRecord:
    return DecisionTraceRecord(
        service_name=str(row["service_name"]),
        execution_mode=str(row["execution_mode"]),
        symbol=str(row["symbol"]),
        signal=str(row["signal"]),
        signal_interval_begin=row["signal_interval_begin"],
        signal_as_of_time=row["signal_as_of_time"],
        signal_row_id=str(row["signal_row_id"]),
        model_name=str(row["model_name"]),
        model_version=str(row["model_version"]),
        payload=DecisionTracePayload.model_validate(_json_object(row["trace_payload"])),
        risk_outcome=None if row["risk_outcome"] is None else str(row["risk_outcome"]),
        json_report_path=(
            None if row["json_report_path"] is None else str(row["json_report_path"])
        ),
        markdown_report_path=(
            None
            if row["markdown_report_path"] is None
            else str(row["markdown_report_path"])
        ),
        decision_trace_id=int(row["id"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _order_event_from_row(row: asyncpg.Record) -> OrderLifecycleEvent:
    return OrderLifecycleEvent(
        order_request_id=int(row["order_request_id"]),
        service_name=str(row["service_name"]),
        execution_mode=str(row["execution_mode"]),
        symbol=str(row["symbol"]),
        action=str(row["action"]),
        lifecycle_state=str(row["lifecycle_state"]),
        event_time=row["event_time"],
        reason_code=None if row["reason_code"] is None else str(row["reason_code"]),
        details=None if row["details"] is None else str(row["details"]),
        external_order_id=(
            None if row["external_order_id"] is None else str(row["external_order_id"])
        ),
        external_status=(
            None if row["external_status"] is None else str(row["external_status"])
        ),
        account_id=None if row["account_id"] is None else str(row["account_id"]),
        environment_name=(
            None if row["environment_name"] is None else str(row["environment_name"])
        ),
        broker_name=None if row["broker_name"] is None else str(row["broker_name"]),
        probe_policy_active=bool(row["probe_policy_active"]),
        probe_symbol=None if row["probe_symbol"] is None else str(row["probe_symbol"]),
        probe_qty=None if row["probe_qty"] is None else int(row["probe_qty"]),
        decision_trace_id=(
            None if row["decision_trace_id"] is None else int(row["decision_trace_id"])
        ),
        event_id=int(row["id"]),
        created_at=row["created_at"],
    )


def _ledger_entry_from_row(row: asyncpg.Record) -> TradeLedgerEntry:
    return TradeLedgerEntry(
        service_name=str(row["service_name"]),
        execution_mode=str(row["execution_mode"]),
        position_id=None if row["position_id"] is None else int(row["position_id"]),
        order_request_id=(
            None if row["order_request_id"] is None else int(row["order_request_id"])
        ),
        decision_trace_id=(
            None if row["decision_trace_id"] is None else int(row["decision_trace_id"])
        ),
        symbol=str(row["symbol"]),
        action=str(row["action"]),
        reason=str(row["reason"]),
        signal_interval_begin=row["signal_interval_begin"],
        signal_as_of_time=row["signal_as_of_time"],
        signal_row_id=(
            None if row["signal_row_id"] is None else str(row["signal_row_id"])
        ),
        model_name=None if row["model_name"] is None else str(row["model_name"]),
        prob_up=None if row["prob_up"] is None else float(row["prob_up"]),
        prob_down=None if row["prob_down"] is None else float(row["prob_down"]),
        confidence=None if row["confidence"] is None else float(row["confidence"]),
        regime_label=(
            None if row["regime_label"] is None else str(row["regime_label"])
        ),
        approved_notional=(
            None if row["approved_notional"] is None else float(row["approved_notional"])
        ),
        risk_outcome=(
            None if row["risk_outcome"] is None else str(row["risk_outcome"])
        ),
        risk_reason_codes=tuple(str(value) for value in row["risk_reason_codes"] or []),
        fill_interval_begin=row["fill_interval_begin"],
        fill_time=row["fill_time"],
        fill_price=float(row["fill_price"]),
        quantity=float(row["quantity"]),
        notional=float(row["notional"]),
        fee=float(row["fee"]),
        slippage_bps=float(row["slippage_bps"]),
        cash_flow=float(row["cash_flow"]),
        realized_pnl=None if row["realized_pnl"] is None else float(row["realized_pnl"]),
        created_at=row["created_at"],
    )


def _position_from_row(row: asyncpg.Record) -> PaperPosition:
    return PaperPosition(
        service_name=str(row["service_name"]),
        execution_mode=str(row["execution_mode"]),
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
        entry_order_request_id=(
            None if row["entry_order_request_id"] is None else int(row["entry_order_request_id"])
        ),
        entry_decision_trace_id=(
            None if row["entry_decision_trace_id"] is None else int(row["entry_decision_trace_id"])
        ),
        entry_regime_label=(
            None if row["entry_regime_label"] is None else str(row["entry_regime_label"])
        ),
        entry_approved_notional=(
            None
            if row["entry_approved_notional"] is None
            else float(row["entry_approved_notional"])
        ),
        entry_risk_outcome=(
            None if row["entry_risk_outcome"] is None else str(row["entry_risk_outcome"])
        ),
        entry_risk_reason_codes=tuple(
            str(value) for value in row["entry_risk_reason_codes"] or []
        ),
        position_id=int(row["id"]),
        exit_reason=None if row["exit_reason"] is None else str(row["exit_reason"]),
        exit_signal_interval_begin=row["exit_signal_interval_begin"],
        exit_signal_as_of_time=row["exit_signal_as_of_time"],
        exit_signal_row_id=(
            None if row["exit_signal_row_id"] is None else str(row["exit_signal_row_id"])
        ),
        exit_model_name=(
            None if row["exit_model_name"] is None else str(row["exit_model_name"])
        ),
        exit_prob_up=None if row["exit_prob_up"] is None else float(row["exit_prob_up"]),
        exit_confidence=(
            None if row["exit_confidence"] is None else float(row["exit_confidence"])
        ),
        exit_fill_interval_begin=row["exit_fill_interval_begin"],
        exit_fill_time=row["exit_fill_time"],
        exit_price=None if row["exit_price"] is None else float(row["exit_price"]),
        exit_notional=(
            None if row["exit_notional"] is None else float(row["exit_notional"])
        ),
        exit_fee=None if row["exit_fee"] is None else float(row["exit_fee"]),
        realized_pnl=(
            None if row["realized_pnl"] is None else float(row["realized_pnl"])
        ),
        realized_return=(
            None if row["realized_return"] is None else float(row["realized_return"])
        ),
        exit_regime_label=(
            None if row["exit_regime_label"] is None else str(row["exit_regime_label"])
        ),
        exit_order_request_id=(
            None if row["exit_order_request_id"] is None else int(row["exit_order_request_id"])
        ),
        exit_decision_trace_id=(
            None if row["exit_decision_trace_id"] is None else int(row["exit_decision_trace_id"])
        ),
        opened_at=row["opened_at"],
        closed_at=row["closed_at"],
        updated_at=row["updated_at"],
    )


def _heartbeat_from_row(row: asyncpg.Record) -> ServiceHeartbeat:
    return ServiceHeartbeat(
        service_name=str(row["service_name"]),
        component_name=str(row["component_name"]),
        heartbeat_at=row["heartbeat_at"],
        health_overall_status=str(row["health_overall_status"]),
        reason_code=str(row["reason_code"]),
        detail=None if row["details"] is None else str(row["details"]),
        heartbeat_id=int(row["id"]),
        created_at=row["created_at"],
    )


def _recovery_event_from_row(row: asyncpg.Record) -> RecoveryEvent:
    return RecoveryEvent(
        service_name=str(row["service_name"]),
        component_name=str(row["component_name"]),
        event_type=str(row["event_type"]),
        event_time=row["event_time"],
        reason_code=str(row["reason_code"]),
        health_overall_status=(
            None if row["health_overall_status"] is None else str(row["health_overall_status"])
        ),
        freshness_status=(
            None if row["freshness_status"] is None else str(row["freshness_status"])
        ),
        breaker_state=(
            None if row["breaker_state"] is None else str(row["breaker_state"])
        ),
        detail=None if row["details"] is None else str(row["details"]),
        event_id=int(row["id"]),
        created_at=row["created_at"],
    )
