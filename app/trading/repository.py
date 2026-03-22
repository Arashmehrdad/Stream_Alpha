"""PostgreSQL persistence for the Stream Alpha M5 paper trader."""

# pylint: disable=duplicate-code,too-many-lines

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import date, datetime

import asyncpg

from app.adaptation.schemas import (
    AdaptiveChallengerRunRecord,
    AdaptiveDriftRecord,
    AdaptivePerformanceWindow,
    AdaptiveProfileRecord,
    AdaptivePromotionDecisionRecord,
)
from app.common.time import to_rfc3339
from app.ensemble.schemas import (
    EnsembleChallengerRunRecord,
    EnsembleProfileRecord,
    EnsemblePromotionDecisionRecord,
)
from app.explainability.schemas import DecisionTracePayload
from app.reliability.schemas import RecoveryEvent, ReliabilityState, ServiceHeartbeat
from app.trading.schemas import (
    DecisionTraceRecord,
    FeatureCandle,
    LiveSafetyState,
    OrderLifecycleEvent,
    OrderRequest,
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
DECISION_TRACES_TABLE = "decision_traces"
ORDER_REQUESTS_TABLE = "execution_order_requests"
ORDER_EVENTS_TABLE = "execution_order_events"
LIVE_SAFETY_TABLE = "execution_live_safety_state"
HEARTBEATS_TABLE = "service_heartbeats"
RELIABILITY_STATE_TABLE = "reliability_state"
RELIABILITY_EVENTS_TABLE = "reliability_events"
ADAPTIVE_DRIFT_STATE_TABLE = "adaptive_drift_state"
ADAPTIVE_PERFORMANCE_WINDOWS_TABLE = "adaptive_performance_windows"
ADAPTIVE_PROFILES_TABLE = "adaptive_profiles"
ADAPTIVE_CHALLENGER_RUNS_TABLE = "adaptive_challenger_runs"
ADAPTIVE_PROMOTION_DECISIONS_TABLE = "adaptive_promotion_decisions"
ENSEMBLE_PROFILES_TABLE = "ensemble_profiles"
ENSEMBLE_CHALLENGER_RUNS_TABLE = "ensemble_challenger_runs"
ENSEMBLE_PROMOTION_DECISIONS_TABLE = "ensemble_promotion_decisions"

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


def _table_basename(name: str) -> str:
    return name.split(".")[-1]


class TradingRepository:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """Repository for M5 state, positions, ledger rows, and canonical candles."""

    def __init__(self, dsn: str, source_table: str) -> None:
        self._dsn = dsn
        self._source_table = _quote_table_name(source_table)
        self._positions_table = _quote_table_name(POSITIONS_TABLE)
        self._ledger_table = _quote_table_name(LEDGER_TABLE)
        self._state_table = _quote_table_name(STATE_TABLE)
        self._risk_state_table = _quote_table_name(RISK_STATE_TABLE)
        self._risk_decisions_table = _quote_table_name(RISK_DECISIONS_TABLE)
        self._decision_traces_table = _quote_table_name(DECISION_TRACES_TABLE)
        self._order_requests_table = _quote_table_name(ORDER_REQUESTS_TABLE)
        self._order_events_table = _quote_table_name(ORDER_EVENTS_TABLE)
        self._live_safety_table = _quote_table_name(LIVE_SAFETY_TABLE)
        self._heartbeats_table = _quote_table_name(HEARTBEATS_TABLE)
        self._reliability_state_table = _quote_table_name(RELIABILITY_STATE_TABLE)
        self._reliability_events_table = _quote_table_name(RELIABILITY_EVENTS_TABLE)
        self._adaptive_drift_state_table = _quote_table_name(ADAPTIVE_DRIFT_STATE_TABLE)
        self._adaptive_performance_windows_table = _quote_table_name(
            ADAPTIVE_PERFORMANCE_WINDOWS_TABLE
        )
        self._adaptive_profiles_table = _quote_table_name(ADAPTIVE_PROFILES_TABLE)
        self._adaptive_challenger_runs_table = _quote_table_name(
            ADAPTIVE_CHALLENGER_RUNS_TABLE
        )
        self._adaptive_promotion_decisions_table = _quote_table_name(
            ADAPTIVE_PROMOTION_DECISIONS_TABLE
        )
        self._ensemble_profiles_table = _quote_table_name(ENSEMBLE_PROFILES_TABLE)
        self._ensemble_challenger_runs_table = _quote_table_name(
            ENSEMBLE_CHALLENGER_RUNS_TABLE
        )
        self._ensemble_promotion_decisions_table = _quote_table_name(
            ENSEMBLE_PROMOTION_DECISIONS_TABLE
        )
        self._pool: asyncpg.Pool | None = None

    @property
    def dsn(self) -> str:
        """Expose the configured DSN for additive helper repositories."""
        return self._dsn

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
        execution_mode: str,
        symbols: Sequence[str],
    ) -> dict[str, PaperEngineState]:
        """Load persisted per-symbol engine state and fill any missing defaults."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._state_table}
            WHERE service_name = $1
              AND execution_mode = $2
              AND symbol = ANY($3::text[])
            """,
            service_name,
            execution_mode,
            list(symbols),
        )
        states = {
            str(row["symbol"]): _state_from_row(row)
            for row in rows
        }
        for symbol in symbols:
            states.setdefault(
                symbol,
                PaperEngineState(
                    service_name=service_name,
                    symbol=symbol,
                    execution_mode=execution_mode,
                ),
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
                execution_mode,
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
                pending_order_request_id,
                pending_order_idempotency_key,
                pending_risk_reason_codes,
                pending_decision_trace_id,
                updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                $16, $17, $18, $19, $20, $21::text[], $22, NOW()
            )
            ON CONFLICT (service_name, execution_mode, symbol)
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
                pending_order_request_id = EXCLUDED.pending_order_request_id,
                pending_order_idempotency_key = EXCLUDED.pending_order_idempotency_key,
                pending_risk_reason_codes = EXCLUDED.pending_risk_reason_codes,
                pending_decision_trace_id = EXCLUDED.pending_decision_trace_id,
                updated_at = NOW()
            """,
            state.service_name,
            state.execution_mode,
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
            None if pending is None else pending.order_request_id,
            None if pending is None else pending.order_request_idempotency_key,
            None if pending is None else list(pending.risk_reason_codes),
            None if pending is None else state.pending_decision_trace_id,
        )

    async def load_service_risk_state(
        self,
        *,
        service_name: str,
        execution_mode: str,
    ) -> ServiceRiskState | None:
        """Load the persisted M10 service-level risk state, if present."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT *
            FROM {self._risk_state_table}
            WHERE service_name = $1 AND execution_mode = $2
            """,
            service_name,
            execution_mode,
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
                execution_mode,
                trading_day,
                realized_pnl_today,
                equity_high_watermark,
                current_equity,
                loss_streak_count,
                loss_streak_cooldown_until_interval_begin,
                kill_switch_enabled,
                updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, NOW()
            )
            ON CONFLICT (service_name, execution_mode)
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
            state.execution_mode,
            state.trading_day,
            state.realized_pnl_today,
            state.equity_high_watermark,
            state.current_equity,
            state.loss_streak_count,
            state.loss_streak_cooldown_until_interval_begin,
            state.kill_switch_enabled,
        )

    async def ensure_decision_trace(
        self,
        trace: DecisionTraceRecord,
    ) -> DecisionTraceRecord:
        """Insert one canonical M14 decision trace or return the existing row."""
        pool = self._require_pool()
        payload_json = json.dumps(trace.payload.model_dump(mode="json"))
        row = await pool.fetchrow(
            f"""
            INSERT INTO {self._decision_traces_table} (
                service_name,
                execution_mode,
                symbol,
                signal,
                signal_interval_begin,
                signal_as_of_time,
                signal_row_id,
                model_name,
                model_version,
                risk_outcome,
                trace_payload,
                json_report_path,
                markdown_report_path
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::jsonb, $12, $13
            )
            ON CONFLICT (service_name, execution_mode, signal_row_id) DO NOTHING
            RETURNING *
            """,
            trace.service_name,
            trace.execution_mode,
            trace.symbol,
            trace.signal,
            trace.signal_interval_begin,
            trace.signal_as_of_time,
            trace.signal_row_id,
            trace.model_name,
            trace.model_version,
            trace.risk_outcome,
            payload_json,
            trace.json_report_path,
            trace.markdown_report_path,
        )
        if row is None:
            existing_row = await pool.fetchrow(
                f"""
                SELECT *
                FROM {self._decision_traces_table}
                WHERE service_name = $1
                  AND execution_mode = $2
                  AND signal_row_id = $3
                """,
                trace.service_name,
                trace.execution_mode,
                trace.signal_row_id,
            )
            if existing_row is None:
                raise RuntimeError("Decision trace insert returned no row and no existing record")
            row = existing_row
        return _decision_trace_from_row(row)

    async def update_decision_trace(
        self,
        trace: DecisionTraceRecord,
    ) -> DecisionTraceRecord:
        """Persist one enriched canonical M14 decision trace payload."""
        if trace.decision_trace_id is None:
            raise ValueError("DecisionTraceRecord must have decision_trace_id before update")
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            UPDATE {self._decision_traces_table}
            SET
                symbol = $1,
                signal = $2,
                signal_interval_begin = $3,
                signal_as_of_time = $4,
                model_name = $5,
                model_version = $6,
                risk_outcome = $7,
                trace_payload = $8::jsonb,
                json_report_path = $9,
                markdown_report_path = $10,
                updated_at = NOW()
            WHERE id = $11
            RETURNING *
            """,
            trace.symbol,
            trace.signal,
            trace.signal_interval_begin,
            trace.signal_as_of_time,
            trace.model_name,
            trace.model_version,
            trace.risk_outcome,
            json.dumps(trace.payload.model_dump(mode="json")),
            trace.json_report_path,
            trace.markdown_report_path,
            trace.decision_trace_id,
        )
        if row is None:
            raise RuntimeError(f"Decision trace {trace.decision_trace_id} could not be updated")
        return _decision_trace_from_row(row)

    async def load_decision_trace(
        self,
        *,
        decision_trace_id: int,
    ) -> DecisionTraceRecord | None:
        """Load one canonical M14 decision trace by its persisted identifier."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT *
            FROM {self._decision_traces_table}
            WHERE id = $1
            """,
            decision_trace_id,
        )
        if row is None:
            return None
        return _decision_trace_from_row(row)

    async def insert_risk_decision(self, entry: RiskDecisionLogEntry) -> None:
        """Persist one M10 risk-decision audit row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._risk_decisions_table} (
                service_name,
                execution_mode,
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
                trade_allowed,
                decision_trace_id,
                model_version
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9::text[], $10, $11, $12, $13, $14, $15,
                $16, $17, $18, $19, $20, $21, $22
            )
            """,
            entry.service_name,
            entry.execution_mode,
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
            entry.decision_trace_id,
            entry.model_version,
        )

    async def load_live_safety_state(
        self,
        *,
        service_name: str,
        execution_mode: str,
    ) -> LiveSafetyState | None:
        """Load the persisted M12 live safety state, if present."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT *
            FROM {self._live_safety_table}
            WHERE service_name = $1 AND execution_mode = $2
            """,
            service_name,
            execution_mode,
        )
        if row is None:
            return None
        return _live_safety_state_from_row(row)

    async def save_live_safety_state(self, state: LiveSafetyState) -> None:
        """Upsert the persisted M12 live safety state."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._live_safety_table} (
                service_name,
                execution_mode,
                broker_name,
                live_enabled,
                startup_checks_passed,
                startup_checks_passed_at,
                account_validated,
                account_id,
                environment_name,
                manual_disable_active,
                consecutive_live_failures,
                failure_hard_stop_active,
                last_failure_reason,
                system_health_status,
                system_health_reason_code,
                system_health_checked_at,
                health_gate_status,
                health_gate_reason_code,
                health_gate_detail,
                broker_cash,
                broker_equity,
                reconciliation_status,
                reconciliation_reason_code,
                reconciliation_checked_at,
                unresolved_incident_count,
                can_submit_live_now,
                primary_block_reason_code,
                block_detail,
                updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, NOW()
            )
            ON CONFLICT (service_name, execution_mode)
            DO UPDATE SET
                broker_name = EXCLUDED.broker_name,
                live_enabled = EXCLUDED.live_enabled,
                startup_checks_passed = EXCLUDED.startup_checks_passed,
                startup_checks_passed_at = EXCLUDED.startup_checks_passed_at,
                account_validated = EXCLUDED.account_validated,
                account_id = EXCLUDED.account_id,
                environment_name = EXCLUDED.environment_name,
                manual_disable_active = EXCLUDED.manual_disable_active,
                consecutive_live_failures = EXCLUDED.consecutive_live_failures,
                failure_hard_stop_active = EXCLUDED.failure_hard_stop_active,
                last_failure_reason = EXCLUDED.last_failure_reason,
                system_health_status = EXCLUDED.system_health_status,
                system_health_reason_code = EXCLUDED.system_health_reason_code,
                system_health_checked_at = EXCLUDED.system_health_checked_at,
                health_gate_status = EXCLUDED.health_gate_status,
                health_gate_reason_code = EXCLUDED.health_gate_reason_code,
                health_gate_detail = EXCLUDED.health_gate_detail,
                broker_cash = EXCLUDED.broker_cash,
                broker_equity = EXCLUDED.broker_equity,
                reconciliation_status = EXCLUDED.reconciliation_status,
                reconciliation_reason_code = EXCLUDED.reconciliation_reason_code,
                reconciliation_checked_at = EXCLUDED.reconciliation_checked_at,
                unresolved_incident_count = EXCLUDED.unresolved_incident_count,
                can_submit_live_now = EXCLUDED.can_submit_live_now,
                primary_block_reason_code = EXCLUDED.primary_block_reason_code,
                block_detail = EXCLUDED.block_detail,
                updated_at = NOW()
            """,
            state.service_name,
            state.execution_mode,
            state.broker_name,
            state.live_enabled,
            state.startup_checks_passed,
            state.startup_checks_passed_at,
            state.account_validated,
            state.account_id,
            state.environment_name,
            state.manual_disable_active,
            state.consecutive_live_failures,
            state.failure_hard_stop_active,
            state.last_failure_reason,
            state.system_health_status,
            state.system_health_reason_code,
            state.system_health_checked_at,
            state.health_gate_status,
            state.health_gate_reason_code,
            state.health_gate_detail,
            state.broker_cash,
            state.broker_equity,
            state.reconciliation_status,
            state.reconciliation_reason_code,
            state.reconciliation_checked_at,
            state.unresolved_incident_count,
            state.can_submit_live_now,
            state.primary_block_reason_code,
            state.block_detail,
        )

    async def save_service_heartbeat(
        self,
        heartbeat: ServiceHeartbeat,
    ) -> ServiceHeartbeat:
        """Insert one reliability heartbeat row."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            INSERT INTO {self._heartbeats_table} (
                service_name,
                component_name,
                heartbeat_at,
                health_overall_status,
                reason_code,
                details
            ) VALUES (
                $1, $2, $3, $4, $5, $6
            )
            RETURNING *
            """,
            heartbeat.service_name,
            heartbeat.component_name,
            heartbeat.heartbeat_at,
            heartbeat.health_overall_status,
            heartbeat.reason_code,
            heartbeat.detail,
        )
        if row is None:
            raise RuntimeError("Heartbeat insert returned no row")
        return _heartbeat_from_row(row)

    async def load_latest_service_heartbeat(
        self,
        *,
        service_name: str,
        component_name: str,
    ) -> ServiceHeartbeat | None:
        """Load the latest heartbeat row for one service component."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT *
            FROM {self._heartbeats_table}
            WHERE service_name = $1 AND component_name = $2
            ORDER BY heartbeat_at DESC, id DESC
            LIMIT $3
            """,
            service_name,
            component_name,
            1,
        )
        if row is None:
            return None
        return _heartbeat_from_row(row)

    async def save_reliability_state(self, state: ReliabilityState) -> None:
        """Upsert one reliability-state row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._reliability_state_table} (
                service_name,
                component_name,
                health_overall_status,
                freshness_status,
                breaker_state,
                failure_count,
                success_count,
                last_heartbeat_at,
                last_success_at,
                last_failure_at,
                opened_at,
                reason_code,
                details,
                updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, NOW()
            )
            ON CONFLICT (service_name, component_name)
            DO UPDATE SET
                health_overall_status = EXCLUDED.health_overall_status,
                freshness_status = EXCLUDED.freshness_status,
                breaker_state = EXCLUDED.breaker_state,
                failure_count = EXCLUDED.failure_count,
                success_count = EXCLUDED.success_count,
                last_heartbeat_at = EXCLUDED.last_heartbeat_at,
                last_success_at = EXCLUDED.last_success_at,
                last_failure_at = EXCLUDED.last_failure_at,
                opened_at = EXCLUDED.opened_at,
                reason_code = EXCLUDED.reason_code,
                details = EXCLUDED.details,
                updated_at = NOW()
            """,
            state.service_name,
            state.component_name,
            state.health_overall_status,
            state.freshness_status,
            state.breaker_state,
            state.failure_count,
            state.success_count,
            state.last_heartbeat_at,
            state.last_success_at,
            state.last_failure_at,
            state.opened_at,
            state.reason_code,
            state.detail,
        )

    async def load_reliability_state(
        self,
        *,
        service_name: str,
        component_name: str,
    ) -> ReliabilityState | None:
        """Load one persisted reliability-state row."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT *
            FROM {self._reliability_state_table}
            WHERE service_name = $1 AND component_name = $2
            """,
            service_name,
            component_name,
        )
        if row is None:
            return None
        return _reliability_state_from_row(row)

    async def insert_reliability_event(
        self,
        event: RecoveryEvent,
    ) -> RecoveryEvent:
        """Insert one reliability event audit row."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            INSERT INTO {self._reliability_events_table} (
                service_name,
                component_name,
                event_type,
                event_time,
                reason_code,
                health_overall_status,
                freshness_status,
                breaker_state,
                details
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9
            )
            RETURNING *
            """,
            event.service_name,
            event.component_name,
            event.event_type,
            event.event_time,
            event.reason_code,
            event.health_overall_status,
            event.freshness_status,
            event.breaker_state,
            event.detail,
        )
        if row is None:
            raise RuntimeError("Reliability event insert returned no row")
        return _recovery_event_from_row(row)

    async def ensure_order_request(self, order_request: OrderRequest) -> OrderRequest:
        """Insert one deterministic M11 order request or return the existing row."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            INSERT INTO {self._order_requests_table} (
                service_name,
                execution_mode,
                symbol,
                action,
                signal_interval_begin,
                signal_as_of_time,
                signal_row_id,
                target_fill_interval_begin,
                requested_notional,
                approved_notional,
                idempotency_key,
                model_name,
                model_version,
                confidence,
                regime_label,
                regime_run_id,
                risk_outcome,
                risk_reason_codes,
                decision_trace_id
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                $12, $13, $14, $15, $16, $17, $18::text[], $19
            )
            ON CONFLICT (idempotency_key)
            DO UPDATE SET
                decision_trace_id = COALESCE(
                    {self._order_requests_table}.decision_trace_id,
                    EXCLUDED.decision_trace_id
                ),
                model_version = COALESCE(
                    {self._order_requests_table}.model_version,
                    EXCLUDED.model_version
                )
            RETURNING *
            """,
            order_request.service_name,
            order_request.execution_mode,
            order_request.symbol,
            order_request.action,
            order_request.signal_interval_begin,
            order_request.signal_as_of_time,
            order_request.signal_row_id,
            order_request.target_fill_interval_begin,
            order_request.requested_notional,
            order_request.approved_notional,
            order_request.idempotency_key,
            order_request.model_name,
            order_request.model_version,
            order_request.confidence,
            order_request.regime_label,
            order_request.regime_run_id,
            order_request.risk_outcome,
            list(order_request.risk_reason_codes),
            order_request.decision_trace_id,
        )
        if row is None:
            existing_row = await pool.fetchrow(
                f"""
                SELECT *
                FROM {self._order_requests_table}
                WHERE idempotency_key = $1
                """,
                order_request.idempotency_key,
            )
            if existing_row is None:
                raise RuntimeError("Order request insert returned no row and no existing record")
            row = existing_row
        return _order_request_from_row(row)

    async def load_order_request_by_idempotency_key(
        self,
        *,
        idempotency_key: str,
    ) -> OrderRequest | None:
        """Load one order request by its deterministic idempotency key."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT *
            FROM {self._order_requests_table}
            WHERE idempotency_key = $1
            """,
            idempotency_key,
        )
        if row is None:
            return None
        return _order_request_from_row(row)

    async def insert_order_event_if_absent(
        self,
        event: OrderLifecycleEvent,
    ) -> OrderLifecycleEvent:
        """Insert one lifecycle event unless that request already has the same state."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            INSERT INTO {self._order_events_table} (
                order_request_id,
                service_name,
                execution_mode,
                symbol,
                action,
                lifecycle_state,
                event_time,
                external_order_id,
                external_status,
                account_id,
                environment_name,
                broker_name,
                reason_code,
                details,
                probe_policy_active,
                probe_symbol,
                probe_qty,
                decision_trace_id
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                $16, $17, $18
            )
            ON CONFLICT (order_request_id, lifecycle_state)
            DO UPDATE SET
                decision_trace_id = COALESCE(
                    {self._order_events_table}.decision_trace_id,
                    EXCLUDED.decision_trace_id
                )
            RETURNING *
            """,
            event.order_request_id,
            event.service_name,
            event.execution_mode,
            event.symbol,
            event.action,
            event.lifecycle_state,
            event.event_time,
            event.external_order_id,
            event.external_status,
            event.account_id,
            event.environment_name,
            event.broker_name,
            event.reason_code,
            event.details,
            event.probe_policy_active,
            event.probe_symbol,
            event.probe_qty,
            event.decision_trace_id,
        )
        if row is None:
            existing_row = await pool.fetchrow(
                f"""
                SELECT *
                FROM {self._order_events_table}
                WHERE order_request_id = $1 AND lifecycle_state = $2
                """,
                event.order_request_id,
                event.lifecycle_state,
            )
            if existing_row is None:
                raise RuntimeError("Order lifecycle insert returned no row and no existing record")
            row = existing_row
        return _order_event_from_row(row)

    async def load_recent_order_events(
        self,
        *,
        service_name: str,
        execution_mode: str,
        limit: int,
    ) -> list[OrderLifecycleEvent]:
        """Load recent order lifecycle rows for dashboard and manual inspection."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._order_events_table}
            WHERE service_name = $1 AND execution_mode = $2
            ORDER BY event_time DESC, id DESC
            LIMIT $3
            """,
            service_name,
            execution_mode,
            limit,
        )
        return [_order_event_from_row(row) for row in rows]

    async def load_order_events_since(
        self,
        *,
        service_name: str,
        execution_mode: str,
        since: datetime,
    ) -> list[OrderLifecycleEvent]:
        """Load order lifecycle rows since one explicit timestamp."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._order_events_table}
            WHERE service_name = $1
              AND execution_mode = $2
              AND event_time >= $3
            ORDER BY event_time ASC, id ASC
            """,
            service_name,
            execution_mode,
            since,
        )
        return [_order_event_from_row(row) for row in rows]

    async def load_decision_traces_since(
        self,
        *,
        service_name: str,
        execution_mode: str,
        since: datetime,
    ) -> list[DecisionTraceRecord]:
        """Load canonical decision traces since one explicit timestamp."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._decision_traces_table}
            WHERE service_name = $1
              AND execution_mode = $2
              AND signal_as_of_time >= $3
            ORDER BY signal_as_of_time ASC, id ASC
            """,
            service_name,
            execution_mode,
            since,
        )
        return [_decision_trace_from_row(row) for row in rows]

    async def save_adaptive_drift_state(self, record: AdaptiveDriftRecord) -> None:
        """Upsert one adaptive drift state row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._adaptive_drift_state_table} (
                symbol,
                regime_label,
                detector_name,
                window_id,
                reference_window_start,
                reference_window_end,
                live_window_start,
                live_window_end,
                drift_score,
                warning_threshold,
                breach_threshold,
                status,
                reason_code,
                detail,
                created_at,
                updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                COALESCE($15, NOW()), COALESCE($16, NOW())
            )
            ON CONFLICT (symbol, regime_label, detector_name, window_id)
            DO UPDATE SET
                reference_window_start = EXCLUDED.reference_window_start,
                reference_window_end = EXCLUDED.reference_window_end,
                live_window_start = EXCLUDED.live_window_start,
                live_window_end = EXCLUDED.live_window_end,
                drift_score = EXCLUDED.drift_score,
                warning_threshold = EXCLUDED.warning_threshold,
                breach_threshold = EXCLUDED.breach_threshold,
                status = EXCLUDED.status,
                reason_code = EXCLUDED.reason_code,
                detail = EXCLUDED.detail,
                updated_at = NOW()
            """,
            record.symbol,
            record.regime_label,
            record.detector_name,
            record.window_id,
            record.reference_window_start,
            record.reference_window_end,
            record.live_window_start,
            record.live_window_end,
            record.drift_score,
            record.warning_threshold,
            record.breach_threshold,
            record.status,
            record.reason_code,
            record.detail,
            record.created_at,
            record.updated_at,
        )

    async def load_adaptive_drift_states(
        self,
        *,
        symbol: str,
        regime_label: str,
        limit: int,
    ) -> list[AdaptiveDriftRecord]:
        """Load recent adaptive drift state rows for one symbol and regime."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._adaptive_drift_state_table}
            WHERE symbol = $1
              AND regime_label IN ($2, 'ALL')
            ORDER BY updated_at DESC, created_at DESC
            LIMIT $3
            """,
            symbol,
            regime_label,
            limit,
        )
        return [_adaptive_drift_state_from_row(row) for row in rows]

    async def load_latest_adaptive_drift_state(
        self,
        *,
        symbol: str,
        regime_label: str,
    ) -> AdaptiveDriftRecord | None:
        """Load the latest adaptive drift state row for one symbol and regime."""
        rows = await self.load_adaptive_drift_states(
            symbol=symbol,
            regime_label=regime_label,
            limit=1,
        )
        return None if not rows else rows[0]

    async def save_adaptive_performance_window(
        self,
        record: AdaptivePerformanceWindow,
    ) -> None:
        """Upsert one adaptive performance window row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._adaptive_performance_windows_table} (
                execution_mode,
                symbol,
                regime_label,
                window_id,
                window_type,
                window_start,
                window_end,
                trade_count,
                net_pnl_after_costs,
                max_drawdown,
                profit_factor,
                expectancy,
                win_rate,
                precision,
                avg_slippage_bps,
                blocked_trade_rate,
                shadow_divergence_rate,
                health_context,
                created_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14,
                $15, $16, $17, $18::jsonb, COALESCE($19, NOW())
            )
            ON CONFLICT (execution_mode, symbol, regime_label, window_id)
            DO UPDATE SET
                window_type = EXCLUDED.window_type,
                window_start = EXCLUDED.window_start,
                window_end = EXCLUDED.window_end,
                trade_count = EXCLUDED.trade_count,
                net_pnl_after_costs = EXCLUDED.net_pnl_after_costs,
                max_drawdown = EXCLUDED.max_drawdown,
                profit_factor = EXCLUDED.profit_factor,
                expectancy = EXCLUDED.expectancy,
                win_rate = EXCLUDED.win_rate,
                precision = EXCLUDED.precision,
                avg_slippage_bps = EXCLUDED.avg_slippage_bps,
                blocked_trade_rate = EXCLUDED.blocked_trade_rate,
                shadow_divergence_rate = EXCLUDED.shadow_divergence_rate,
                health_context = EXCLUDED.health_context,
                created_at = COALESCE(EXCLUDED.created_at, NOW())
            """,
            record.execution_mode,
            record.symbol,
            record.regime_label,
            record.window_id,
            record.window_type,
            record.window_start,
            record.window_end,
            record.trade_count,
            record.net_pnl_after_costs,
            record.max_drawdown,
            record.profit_factor,
            record.expectancy,
            record.win_rate,
            record.precision,
            record.avg_slippage_bps,
            record.blocked_trade_rate,
            record.shadow_divergence_rate,
            json.dumps(record.health_context),
            record.created_at,
        )

    async def load_adaptive_performance_windows(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        limit: int,
    ) -> list[AdaptivePerformanceWindow]:
        """Load recent adaptive performance windows for one mode, symbol, and regime."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._adaptive_performance_windows_table}
            WHERE execution_mode = $1
              AND symbol = $2
              AND regime_label IN ($3, 'ALL')
            ORDER BY window_end DESC, created_at DESC
            LIMIT $4
            """,
            execution_mode,
            symbol,
            regime_label,
            limit,
        )
        return [_adaptive_performance_window_from_row(row) for row in rows]

    async def load_latest_adaptive_performance_window(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
    ) -> AdaptivePerformanceWindow | None:
        """Load the latest adaptive performance window for one mode, symbol, and regime."""
        rows = await self.load_adaptive_performance_windows(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
            limit=1,
        )
        return None if not rows else rows[0]

    async def save_adaptive_profile(self, record: AdaptiveProfileRecord) -> None:
        """Upsert one adaptive profile row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._adaptive_profiles_table} (
                profile_id,
                status,
                execution_mode_scope,
                symbol_scope,
                regime_scope,
                threshold_policy_json,
                sizing_policy_json,
                calibration_profile_json,
                source_evidence_json,
                rollback_target_profile_id,
                created_at,
                approved_at,
                activated_at,
                superseded_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb, $9::jsonb,
                $10, COALESCE($11, NOW()), $12, $13, $14
            )
            ON CONFLICT (profile_id)
            DO UPDATE SET
                status = EXCLUDED.status,
                execution_mode_scope = EXCLUDED.execution_mode_scope,
                symbol_scope = EXCLUDED.symbol_scope,
                regime_scope = EXCLUDED.regime_scope,
                threshold_policy_json = EXCLUDED.threshold_policy_json,
                sizing_policy_json = EXCLUDED.sizing_policy_json,
                calibration_profile_json = EXCLUDED.calibration_profile_json,
                source_evidence_json = EXCLUDED.source_evidence_json,
                rollback_target_profile_id = EXCLUDED.rollback_target_profile_id,
                approved_at = EXCLUDED.approved_at,
                activated_at = EXCLUDED.activated_at,
                superseded_at = EXCLUDED.superseded_at
            """,
            record.profile_id,
            record.status,
            record.execution_mode_scope,
            record.symbol_scope,
            record.regime_scope,
            json.dumps(record.threshold_policy_json.model_dump(mode="json")),
            json.dumps(record.sizing_policy_json.model_dump(mode="json")),
            json.dumps(record.calibration_profile_json.model_dump(mode="json")),
            json.dumps(record.source_evidence_json),
            record.rollback_target_profile_id,
            record.created_at,
            record.approved_at,
            record.activated_at,
            record.superseded_at,
        )

    async def load_adaptive_profiles(self, *, limit: int) -> list[AdaptiveProfileRecord]:
        """Load recent adaptive profile rows."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._adaptive_profiles_table}
            ORDER BY created_at DESC, profile_id DESC
            LIMIT $1
            """,
            limit,
        )
        return [_adaptive_profile_from_row(row) for row in rows]

    async def load_adaptive_profile(
        self,
        *,
        profile_id: str,
    ) -> AdaptiveProfileRecord | None:
        """Load one adaptive profile by id."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT *
            FROM {self._adaptive_profiles_table}
            WHERE profile_id = $1
            """,
            profile_id,
        )
        if row is None:
            return None
        return _adaptive_profile_from_row(row)

    async def load_active_adaptive_profile(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
    ) -> AdaptiveProfileRecord | None:
        """Load the best matching active adaptive profile for one runtime scope."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT *
            FROM {self._adaptive_profiles_table}
            WHERE status = 'ACTIVE'
              AND execution_mode_scope IN ($1, 'ALL')
              AND symbol_scope IN ($2, 'ALL')
              AND regime_scope IN ($3, 'ALL')
            ORDER BY
                CASE WHEN execution_mode_scope = $1 THEN 0 ELSE 1 END,
                CASE WHEN symbol_scope = $2 THEN 0 ELSE 1 END,
                CASE WHEN regime_scope = $3 THEN 0 ELSE 1 END,
                activated_at DESC NULLS LAST,
                created_at DESC,
                profile_id DESC
            LIMIT 1
            """,
            execution_mode,
            symbol,
            regime_label,
        )
        if row is None:
            return None
        return _adaptive_profile_from_row(row)

    async def rollback_adaptive_profile(
        self,
        *,
        active_profile_id: str,
        rollback_target_profile_id: str,
        changed_at: datetime,
    ) -> None:
        """Switch the active profile back to its rollback target in one transaction."""
        pool = self._require_pool()
        async with pool.acquire() as connection:
            async with connection.transaction():
                await connection.execute(
                    f"""
                    UPDATE {self._adaptive_profiles_table}
                    SET status = 'ROLLED_BACK',
                        superseded_at = $2
                    WHERE profile_id = $1
                    """,
                    active_profile_id,
                    changed_at,
                )
                await connection.execute(
                    f"""
                    UPDATE {self._adaptive_profiles_table}
                    SET status = 'ACTIVE',
                        activated_at = $2,
                        superseded_at = NULL
                    WHERE profile_id = $1
                    """,
                    rollback_target_profile_id,
                    changed_at,
                )

    async def save_adaptive_challenger_run(
        self,
        record: AdaptiveChallengerRunRecord,
    ) -> None:
        """Upsert one adaptive challenger run row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._adaptive_challenger_runs_table} (
                challenger_run_id,
                status,
                train_window_start,
                train_window_end,
                validation_window_start,
                validation_window_end,
                shadow_window_start,
                shadow_window_end,
                candidate_model_version,
                config_json,
                metrics_json,
                shadow_summary_json,
                artifact_paths_json,
                reason_codes,
                created_at,
                updated_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10::jsonb, $11::jsonb,
                $12::jsonb, $13::jsonb, $14::text[], COALESCE($15, NOW()),
                COALESCE($16, NOW())
            )
            ON CONFLICT (challenger_run_id)
            DO UPDATE SET
                status = EXCLUDED.status,
                train_window_start = EXCLUDED.train_window_start,
                train_window_end = EXCLUDED.train_window_end,
                validation_window_start = EXCLUDED.validation_window_start,
                validation_window_end = EXCLUDED.validation_window_end,
                shadow_window_start = EXCLUDED.shadow_window_start,
                shadow_window_end = EXCLUDED.shadow_window_end,
                candidate_model_version = EXCLUDED.candidate_model_version,
                config_json = EXCLUDED.config_json,
                metrics_json = EXCLUDED.metrics_json,
                shadow_summary_json = EXCLUDED.shadow_summary_json,
                artifact_paths_json = EXCLUDED.artifact_paths_json,
                reason_codes = EXCLUDED.reason_codes,
                updated_at = NOW()
            """,
            record.challenger_run_id,
            record.status,
            record.train_window_start,
            record.train_window_end,
            record.validation_window_start,
            record.validation_window_end,
            record.shadow_window_start,
            record.shadow_window_end,
            record.candidate_model_version,
            json.dumps(record.config_json),
            json.dumps(record.metrics_json),
            json.dumps(record.shadow_summary_json),
            json.dumps(record.artifact_paths_json),
            list(record.reason_codes),
            record.created_at,
            record.updated_at,
        )

    async def load_adaptive_challenger_runs(
        self,
        *,
        limit: int,
    ) -> list[AdaptiveChallengerRunRecord]:
        """Load recent adaptive challenger runs."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._adaptive_challenger_runs_table}
            ORDER BY updated_at DESC, created_at DESC
            LIMIT $1
            """,
            limit,
        )
        return [_adaptive_challenger_run_from_row(row) for row in rows]

    async def save_adaptive_promotion_decision(
        self,
        record: AdaptivePromotionDecisionRecord,
    ) -> None:
        """Upsert one adaptive promotion decision row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._adaptive_promotion_decisions_table} (
                decision_id,
                target_type,
                target_id,
                incumbent_id,
                decision,
                metrics_delta_json,
                safety_checks_json,
                research_integrity_json,
                reason_codes,
                summary_text,
                decided_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::jsonb, $9::text[],
                $10, $11
            )
            ON CONFLICT (decision_id)
            DO UPDATE SET
                target_type = EXCLUDED.target_type,
                target_id = EXCLUDED.target_id,
                incumbent_id = EXCLUDED.incumbent_id,
                decision = EXCLUDED.decision,
                metrics_delta_json = EXCLUDED.metrics_delta_json,
                safety_checks_json = EXCLUDED.safety_checks_json,
                research_integrity_json = EXCLUDED.research_integrity_json,
                reason_codes = EXCLUDED.reason_codes,
                summary_text = EXCLUDED.summary_text,
                decided_at = EXCLUDED.decided_at
            """,
            record.decision_id,
            record.target_type,
            record.target_id,
            record.incumbent_id,
            record.decision,
            json.dumps(record.metrics_delta_json),
            json.dumps(record.safety_checks_json),
            json.dumps(record.research_integrity_json),
            list(record.reason_codes),
            record.summary_text,
            record.decided_at,
        )

    async def load_adaptive_promotion_decisions(
        self,
        *,
        limit: int,
    ) -> list[AdaptivePromotionDecisionRecord]:
        """Load recent adaptive promotion decisions."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._adaptive_promotion_decisions_table}
            ORDER BY decided_at DESC, decision_id DESC
            LIMIT $1
            """,
            limit,
        )
        return [_adaptive_promotion_decision_from_row(row) for row in rows]

    # ------------------------------------------------------------------
    # M20 ensemble CRUD
    # ------------------------------------------------------------------

    async def save_ensemble_profile(self, record: EnsembleProfileRecord) -> None:
        """Upsert one ensemble profile row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._ensemble_profiles_table} (
                profile_id, status, approval_stage,
                candidate_roster_json, regime_weight_matrix_json,
                agreement_policy_json, evidence_summary_json,
                rollback_target_profile_id,
                created_at, approved_at, activated_at, superseded_at
            ) VALUES (
                $1, $2, $3, $4::jsonb, $5::jsonb, $6::jsonb, $7::jsonb,
                $8, $9, $10, $11, $12
            )
            ON CONFLICT (profile_id)
            DO UPDATE SET
                status = EXCLUDED.status,
                approval_stage = EXCLUDED.approval_stage,
                candidate_roster_json = EXCLUDED.candidate_roster_json,
                regime_weight_matrix_json = EXCLUDED.regime_weight_matrix_json,
                agreement_policy_json = EXCLUDED.agreement_policy_json,
                evidence_summary_json = EXCLUDED.evidence_summary_json,
                rollback_target_profile_id = EXCLUDED.rollback_target_profile_id,
                approved_at = EXCLUDED.approved_at,
                activated_at = EXCLUDED.activated_at,
                superseded_at = EXCLUDED.superseded_at
            """,
            record.profile_id,
            record.status,
            record.approval_stage,
            json.dumps(record.candidate_roster_json),
            json.dumps(record.regime_weight_matrix_json),
            json.dumps(record.agreement_policy_json),
            json.dumps(record.evidence_summary_json),
            record.rollback_target_profile_id,
            record.created_at,
            record.approved_at,
            record.activated_at,
            record.superseded_at,
        )

    async def load_active_ensemble_profile(self) -> EnsembleProfileRecord | None:
        """Load the single active ensemble profile, if any."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT *
            FROM {self._ensemble_profiles_table}
            WHERE status = 'ACTIVE'
            ORDER BY activated_at DESC
            LIMIT 1
            """
        )
        if row is None:
            return None
        return _ensemble_profile_from_row(row)

    async def load_ensemble_profile(
        self,
        *,
        profile_id: str,
    ) -> EnsembleProfileRecord | None:
        """Load one ensemble profile by id."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT *
            FROM {self._ensemble_profiles_table}
            WHERE profile_id = $1
            """,
            profile_id,
        )
        if row is None:
            return None
        return _ensemble_profile_from_row(row)

    async def load_ensemble_profiles(
        self,
        *,
        limit: int = 50,
    ) -> list[EnsembleProfileRecord]:
        """Load recent ensemble profiles."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._ensemble_profiles_table}
            ORDER BY created_at DESC, profile_id DESC
            LIMIT $1
            """,
            limit,
        )
        return [_ensemble_profile_from_row(row) for row in rows]

    async def save_ensemble_challenger_run(
        self,
        record: EnsembleChallengerRunRecord,
    ) -> None:
        """Upsert one ensemble challenger run row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._ensemble_challenger_runs_table} (
                challenger_run_id, status, config_json, metrics_json,
                reason_codes, created_at, updated_at
            ) VALUES (
                $1, $2, $3::jsonb, $4::jsonb, $5::text[], $6, $7
            )
            ON CONFLICT (challenger_run_id)
            DO UPDATE SET
                status = EXCLUDED.status,
                config_json = EXCLUDED.config_json,
                metrics_json = EXCLUDED.metrics_json,
                reason_codes = EXCLUDED.reason_codes,
                updated_at = EXCLUDED.updated_at
            """,
            record.challenger_run_id,
            record.status,
            json.dumps(record.config_json),
            json.dumps(record.metrics_json),
            list(record.reason_codes),
            record.created_at,
            record.updated_at,
        )

    async def save_ensemble_promotion_decision(
        self,
        record: EnsemblePromotionDecisionRecord,
    ) -> None:
        """Upsert one ensemble promotion decision row."""
        pool = self._require_pool()
        await pool.execute(
            f"""
            INSERT INTO {self._ensemble_promotion_decisions_table} (
                decision_id, target_type, target_id, incumbent_id,
                decision, metrics_delta_json, safety_checks_json,
                reason_codes, summary_text, decided_at
            ) VALUES (
                $1, $2, $3, $4, $5, $6::jsonb, $7::jsonb, $8::text[], $9, $10
            )
            ON CONFLICT (decision_id)
            DO UPDATE SET
                target_type = EXCLUDED.target_type,
                target_id = EXCLUDED.target_id,
                incumbent_id = EXCLUDED.incumbent_id,
                decision = EXCLUDED.decision,
                metrics_delta_json = EXCLUDED.metrics_delta_json,
                safety_checks_json = EXCLUDED.safety_checks_json,
                reason_codes = EXCLUDED.reason_codes,
                summary_text = EXCLUDED.summary_text,
                decided_at = EXCLUDED.decided_at
            """,
            record.decision_id,
            record.target_type,
            record.target_id,
            record.incumbent_id,
            record.decision,
            json.dumps(record.metrics_delta_json),
            json.dumps(record.safety_checks_json),
            list(record.reason_codes),
            record.summary_text,
            record.decided_at,
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

    async def fetch_latest_feature_row(
        self,
        *,
        symbol: str,
        source_exchange: str,
        interval_minutes: int,
    ) -> FeatureCandle | None:
        """Load the latest finalized canonical feature row for one symbol."""
        pool = self._require_pool()
        row = await pool.fetchrow(
            f"""
            SELECT id, source_exchange, symbol, interval_minutes, interval_begin,
                   interval_end, as_of_time, raw_event_id, open_price,
                   high_price, low_price, close_price, realized_vol_12
            FROM {self._source_table}
            WHERE source_exchange = $1
              AND symbol = $2
              AND interval_minutes = $3
            ORDER BY as_of_time DESC, interval_begin DESC
            LIMIT 1
            """,
            source_exchange,
            symbol,
            interval_minutes,
        )
        if row is None:
            return None
        return _candle_from_row(row)

    async def load_open_positions(
        self,
        service_name: str,
        *,
        execution_mode: str,
    ) -> dict[str, PaperPosition]:
        """Load currently open positions keyed by symbol."""
        pool = self._require_pool()
        rows = await pool.fetch(
            f"""
            SELECT *
            FROM {self._positions_table}
            WHERE service_name = $1
              AND execution_mode = $2
              AND status = 'OPEN'
            ORDER BY entry_fill_interval_begin ASC
            """,
            service_name,
            execution_mode,
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
                    execution_mode,
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
                    entry_order_request_id,
                    entry_decision_trace_id,
                    entry_regime_label,
                    entry_approved_notional,
                    entry_risk_outcome,
                    entry_risk_reason_codes,
                    exit_order_request_id,
                    exit_decision_trace_id,
                    opened_at,
                    updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                    $21, $22, $23, $24, $25::text[], $26, $27, $28, $29
                )
                RETURNING id
                """,
                position.service_name,
                position.execution_mode,
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
                position.entry_order_request_id,
                position.entry_decision_trace_id,
                position.entry_regime_label,
                position.entry_approved_notional,
                position.entry_risk_outcome,
                list(position.entry_risk_reason_codes),
                position.exit_order_request_id,
                position.exit_decision_trace_id,
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
                entry_order_request_id = $17,
                entry_decision_trace_id = $18,
                entry_regime_label = $19,
                entry_approved_notional = $20,
                entry_risk_outcome = $21,
                entry_risk_reason_codes = $22::text[],
                exit_regime_label = $23,
                exit_order_request_id = $24,
                exit_decision_trace_id = $25,
                closed_at = $26,
                updated_at = $27
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
            position.entry_order_request_id,
            position.entry_decision_trace_id,
            position.entry_regime_label,
            position.entry_approved_notional,
            position.entry_risk_outcome,
            list(position.entry_risk_reason_codes),
            position.exit_regime_label,
            position.exit_order_request_id,
            position.exit_decision_trace_id,
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
                execution_mode,
                position_id,
                order_request_id,
                decision_trace_id,
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
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12,
                $13, $14, $15, $16, $17, $18, $19::text[], $20, $21, $22, $23,
                $24, $25, $26, $27, $28
            )
            """,
            entry.service_name,
            entry.execution_mode,
            entry.position_id,
            entry.order_request_id,
            entry.decision_trace_id,
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
        execution_mode: str,
        initial_cash: float,
    ) -> float:
        """Reconstruct current cash from the persisted trade ledger."""
        pool = self._require_pool()
        cash_delta = float(
            await pool.fetchval(
                f"""
                SELECT COALESCE(SUM(cash_flow), 0.0)
                FROM {self._ledger_table}
                WHERE service_name = $1 AND execution_mode = $2
                """,
                service_name,
                execution_mode,
            )
        )
        return initial_cash + cash_delta

    async def load_positions(
        self,
        *,
        service_name: str,
        execution_mode: str,
        status: str | None = None,
    ) -> list[PaperPosition]:
        """Load persisted positions, optionally filtered by status."""
        pool = self._require_pool()
        if status is None:
            rows = await pool.fetch(
                f"""
                SELECT *
                FROM {self._positions_table}
                WHERE service_name = $1 AND execution_mode = $2
                ORDER BY entry_fill_interval_begin ASC, id ASC
                """,
                service_name,
                execution_mode,
            )
        else:
            rows = await pool.fetch(
                f"""
                SELECT *
                FROM {self._positions_table}
                WHERE service_name = $1 AND execution_mode = $2 AND status = $3
                ORDER BY entry_fill_interval_begin ASC, id ASC
                """,
                service_name,
                execution_mode,
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

    async def _ensure_schema(self) -> None:  # pylint: disable=too-many-locals,too-many-statements
        pool = self._require_pool()
        open_position_index = _build_index_name(
            _table_basename(POSITIONS_TABLE),
            "one_open_position_per_symbol_idx",
        )
        state_index = _build_index_name(
            _table_basename(STATE_TABLE),
            "service_mode_symbol_idx",
        )
        ledger_index = _build_index_name(
            _table_basename(LEDGER_TABLE),
            "service_mode_fill_time_idx",
        )
        risk_decisions_index = _build_index_name(
            _table_basename(RISK_DECISIONS_TABLE),
            "service_mode_signal_time_idx",
        )
        decision_traces_index = _build_index_name(
            _table_basename(DECISION_TRACES_TABLE),
            "service_mode_signal_time_idx",
        )
        decision_traces_unique = _build_index_name(
            _table_basename(DECISION_TRACES_TABLE),
            "service_mode_signal_row_uidx",
        )
        order_requests_index = _build_index_name(
            _table_basename(ORDER_REQUESTS_TABLE),
            "service_mode_target_fill_idx",
        )
        order_requests_unique = _build_index_name(
            _table_basename(ORDER_REQUESTS_TABLE),
            "idempotency_key_uidx",
        )
        order_events_index = _build_index_name(
            _table_basename(ORDER_EVENTS_TABLE),
            "service_mode_event_time_idx",
        )
        order_events_unique = _build_index_name(
            _table_basename(ORDER_EVENTS_TABLE),
            "request_state_uidx",
        )
        heartbeats_index = _build_index_name(
            _table_basename(HEARTBEATS_TABLE),
            "service_component_heartbeat_idx",
        )
        reliability_state_primary_key = _quote_identifier(
            f"{_table_basename(RELIABILITY_STATE_TABLE)}_pkey"
        )
        reliability_events_index = _build_index_name(
            _table_basename(RELIABILITY_EVENTS_TABLE),
            "service_component_event_time_idx",
        )
        adaptive_drift_unique = _build_index_name(
            _table_basename(ADAPTIVE_DRIFT_STATE_TABLE),
            "symbol_regime_detector_window_uidx",
        )
        adaptive_drift_index = _build_index_name(
            _table_basename(ADAPTIVE_DRIFT_STATE_TABLE),
            "symbol_regime_updated_idx",
        )
        adaptive_performance_unique = _build_index_name(
            _table_basename(ADAPTIVE_PERFORMANCE_WINDOWS_TABLE),
            "scope_window_uidx",
        )
        adaptive_performance_index = _build_index_name(
            _table_basename(ADAPTIVE_PERFORMANCE_WINDOWS_TABLE),
            "scope_window_end_idx",
        )
        adaptive_profiles_status_index = _build_index_name(
            _table_basename(ADAPTIVE_PROFILES_TABLE),
            "status_scope_idx",
        )
        adaptive_challengers_index = _build_index_name(
            _table_basename(ADAPTIVE_CHALLENGER_RUNS_TABLE),
            "updated_idx",
        )
        adaptive_promotions_index = _build_index_name(
            _table_basename(ADAPTIVE_PROMOTION_DECISIONS_TABLE),
            "decided_idx",
        )
        ensemble_profiles_status_index = _build_index_name(
            _table_basename(ENSEMBLE_PROFILES_TABLE),
            "status_activated_idx",
        )
        ensemble_challengers_index = _build_index_name(
            _table_basename(ENSEMBLE_CHALLENGER_RUNS_TABLE),
            "updated_idx",
        )
        ensemble_promotions_index = _build_index_name(
            _table_basename(ENSEMBLE_PROMOTION_DECISIONS_TABLE),
            "decided_idx",
        )
        live_safety_primary_key = _quote_identifier(
            f"{_table_basename(LIVE_SAFETY_TABLE)}_pkey"
        )
        state_primary_key = _quote_identifier(f"{_table_basename(STATE_TABLE)}_pkey")
        risk_state_primary_key = _quote_identifier(
            f"{_table_basename(RISK_STATE_TABLE)}_pkey"
        )
        async with pool.acquire() as connection:
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._positions_table} (
                    id BIGSERIAL PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    execution_mode TEXT NOT NULL DEFAULT 'paper',
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
                    entry_order_request_id BIGINT NULL,
                    entry_decision_trace_id BIGINT NULL,
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
                    exit_order_request_id BIGINT NULL,
                    exit_decision_trace_id BIGINT NULL,
                    opened_at TIMESTAMPTZ NOT NULL,
                    closed_at TIMESTAMPTZ NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._positions_table}
                ADD COLUMN IF NOT EXISTS execution_mode TEXT NOT NULL DEFAULT 'paper'
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
                ADD COLUMN IF NOT EXISTS entry_order_request_id BIGINT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._positions_table}
                ADD COLUMN IF NOT EXISTS entry_decision_trace_id BIGINT NULL
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
                ALTER TABLE {self._positions_table}
                ADD COLUMN IF NOT EXISTS exit_order_request_id BIGINT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._positions_table}
                ADD COLUMN IF NOT EXISTS exit_decision_trace_id BIGINT NULL
                """
            )
            await connection.execute(f"DROP INDEX IF EXISTS {open_position_index}")
            await connection.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {open_position_index}
                ON {self._positions_table} (service_name, execution_mode, symbol)
                WHERE status = 'OPEN'
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._ledger_table} (
                    id BIGSERIAL PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    execution_mode TEXT NOT NULL DEFAULT 'paper',
                    position_id BIGINT NULL REFERENCES {self._positions_table}(id),
                    order_request_id BIGINT NULL,
                    decision_trace_id BIGINT NULL,
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
                ADD COLUMN IF NOT EXISTS execution_mode TEXT NOT NULL DEFAULT 'paper'
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._ledger_table}
                ADD COLUMN IF NOT EXISTS order_request_id BIGINT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._ledger_table}
                ADD COLUMN IF NOT EXISTS decision_trace_id BIGINT NULL
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
            await connection.execute(f"DROP INDEX IF EXISTS {ledger_index}")
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {ledger_index}
                ON {self._ledger_table} (service_name, execution_mode, fill_time DESC)
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._state_table} (
                    service_name TEXT NOT NULL,
                    execution_mode TEXT NOT NULL DEFAULT 'paper',
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
                    pending_order_request_id BIGINT NULL,
                    pending_order_idempotency_key TEXT NULL,
                    pending_risk_reason_codes TEXT[] NULL,
                    pending_decision_trace_id BIGINT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (service_name, execution_mode, symbol)
                )
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._state_table}
                ADD COLUMN IF NOT EXISTS execution_mode TEXT NOT NULL DEFAULT 'paper'
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
                ADD COLUMN IF NOT EXISTS pending_order_request_id BIGINT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._state_table}
                ADD COLUMN IF NOT EXISTS pending_order_idempotency_key TEXT NULL
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
                ALTER TABLE {self._state_table}
                ADD COLUMN IF NOT EXISTS pending_decision_trace_id BIGINT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._state_table}
                DROP CONSTRAINT IF EXISTS {state_primary_key}
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._state_table}
                ADD PRIMARY KEY (service_name, execution_mode, symbol)
                """
            )
            await connection.execute(f"DROP INDEX IF EXISTS {state_index}")
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {state_index}
                ON {self._state_table} (service_name, execution_mode, symbol)
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._risk_state_table} (
                    service_name TEXT NOT NULL,
                    execution_mode TEXT NOT NULL DEFAULT 'paper',
                    trading_day DATE NOT NULL,
                    realized_pnl_today DOUBLE PRECISION NOT NULL,
                    equity_high_watermark DOUBLE PRECISION NOT NULL,
                    current_equity DOUBLE PRECISION NOT NULL,
                    loss_streak_count INTEGER NOT NULL,
                    loss_streak_cooldown_until_interval_begin TIMESTAMPTZ NULL,
                    kill_switch_enabled BOOLEAN NOT NULL DEFAULT FALSE,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (service_name, execution_mode)
                )
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._risk_state_table}
                ADD COLUMN IF NOT EXISTS execution_mode TEXT NOT NULL DEFAULT 'paper'
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._risk_state_table}
                DROP CONSTRAINT IF EXISTS {risk_state_primary_key}
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._risk_state_table}
                ADD PRIMARY KEY (service_name, execution_mode)
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._decision_traces_table} (
                    id BIGSERIAL PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    execution_mode TEXT NOT NULL DEFAULT 'paper',
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    signal_interval_begin TIMESTAMPTZ NOT NULL,
                    signal_as_of_time TIMESTAMPTZ NOT NULL,
                    signal_row_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    risk_outcome TEXT NULL,
                    trace_payload JSONB NOT NULL,
                    json_report_path TEXT NULL,
                    markdown_report_path TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._decision_traces_table}
                ADD COLUMN IF NOT EXISTS risk_outcome TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._decision_traces_table}
                ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._decision_traces_table}
                ADD COLUMN IF NOT EXISTS json_report_path TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._decision_traces_table}
                ADD COLUMN IF NOT EXISTS markdown_report_path TEXT NULL
                """
            )
            await connection.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {decision_traces_unique}
                ON {self._decision_traces_table} (
                    service_name,
                    execution_mode,
                    signal_row_id
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {decision_traces_index}
                ON {self._decision_traces_table} (
                    service_name,
                    execution_mode,
                    signal_as_of_time DESC,
                    id DESC
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._risk_decisions_table} (
                    id BIGSERIAL PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    execution_mode TEXT NOT NULL DEFAULT 'paper',
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
                    decision_trace_id BIGINT NULL,
                    model_version TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._risk_decisions_table}
                ADD COLUMN IF NOT EXISTS execution_mode TEXT NOT NULL DEFAULT 'paper'
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._risk_decisions_table}
                ADD COLUMN IF NOT EXISTS decision_trace_id BIGINT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._risk_decisions_table}
                ADD COLUMN IF NOT EXISTS model_version TEXT NULL
                """
            )
            await connection.execute(f"DROP INDEX IF EXISTS {risk_decisions_index}")
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {risk_decisions_index}
                ON {self._risk_decisions_table} (
                    service_name,
                    execution_mode,
                    signal_as_of_time DESC,
                    id DESC
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._order_requests_table} (
                    id BIGSERIAL PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    signal_interval_begin TIMESTAMPTZ NOT NULL,
                    signal_as_of_time TIMESTAMPTZ NOT NULL,
                    signal_row_id TEXT NOT NULL,
                    target_fill_interval_begin TIMESTAMPTZ NOT NULL,
                    requested_notional DOUBLE PRECISION NOT NULL,
                    approved_notional DOUBLE PRECISION NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    model_name TEXT NULL,
                    model_version TEXT NULL,
                    confidence DOUBLE PRECISION NULL,
                    regime_label TEXT NULL,
                    regime_run_id TEXT NULL,
                    risk_outcome TEXT NULL,
                    risk_reason_codes TEXT[] NULL,
                    decision_trace_id BIGINT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._order_requests_table}
                ADD COLUMN IF NOT EXISTS model_version TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._order_requests_table}
                ADD COLUMN IF NOT EXISTS decision_trace_id BIGINT NULL
                """
            )
            await connection.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {order_requests_unique}
                ON {self._order_requests_table} (idempotency_key)
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {order_requests_index}
                ON {self._order_requests_table} (
                    service_name,
                    execution_mode,
                    target_fill_interval_begin DESC,
                    id DESC
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._order_events_table} (
                    id BIGSERIAL PRIMARY KEY,
                    order_request_id BIGINT NOT NULL REFERENCES {self._order_requests_table}(id),
                    service_name TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    lifecycle_state TEXT NOT NULL,
                    event_time TIMESTAMPTZ NOT NULL,
                    external_order_id TEXT NULL,
                    external_status TEXT NULL,
                    account_id TEXT NULL,
                    environment_name TEXT NULL,
                    broker_name TEXT NULL,
                    reason_code TEXT NULL,
                    details TEXT NULL,
                    probe_policy_active BOOLEAN NOT NULL DEFAULT FALSE,
                    probe_symbol TEXT NULL,
                    probe_qty INTEGER NULL,
                    decision_trace_id BIGINT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._order_events_table}
                ADD COLUMN IF NOT EXISTS external_order_id TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._order_events_table}
                ADD COLUMN IF NOT EXISTS external_status TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._order_events_table}
                ADD COLUMN IF NOT EXISTS account_id TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._order_events_table}
                ADD COLUMN IF NOT EXISTS environment_name TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._order_events_table}
                ADD COLUMN IF NOT EXISTS broker_name TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._order_events_table}
                ADD COLUMN IF NOT EXISTS probe_policy_active BOOLEAN NOT NULL DEFAULT FALSE
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._order_events_table}
                ADD COLUMN IF NOT EXISTS probe_symbol TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._order_events_table}
                ADD COLUMN IF NOT EXISTS probe_qty INTEGER NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._order_events_table}
                ADD COLUMN IF NOT EXISTS decision_trace_id BIGINT NULL
                """
            )
            await connection.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {order_events_unique}
                ON {self._order_events_table} (order_request_id, lifecycle_state)
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {order_events_index}
                ON {self._order_events_table} (
                    service_name,
                    execution_mode,
                    event_time DESC,
                    id DESC
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._live_safety_table} (
                    service_name TEXT NOT NULL,
                    execution_mode TEXT NOT NULL,
                    broker_name TEXT NOT NULL,
                    live_enabled BOOLEAN NOT NULL,
                    startup_checks_passed BOOLEAN NOT NULL,
                    startup_checks_passed_at TIMESTAMPTZ NULL,
                    account_validated BOOLEAN NOT NULL,
                    account_id TEXT NULL,
                    environment_name TEXT NULL,
                manual_disable_active BOOLEAN NOT NULL,
                consecutive_live_failures INTEGER NOT NULL,
                failure_hard_stop_active BOOLEAN NOT NULL,
                last_failure_reason TEXT NULL,
                system_health_status TEXT NOT NULL DEFAULT 'UNKNOWN',
                system_health_reason_code TEXT NULL,
                system_health_checked_at TIMESTAMPTZ NULL,
                health_gate_status TEXT NOT NULL DEFAULT 'UNKNOWN',
                health_gate_reason_code TEXT NULL,
                health_gate_detail TEXT NULL,
                broker_cash DOUBLE PRECISION NULL,
                broker_equity DOUBLE PRECISION NULL,
                reconciliation_status TEXT NOT NULL DEFAULT 'UNKNOWN',
                reconciliation_reason_code TEXT NULL,
                reconciliation_checked_at TIMESTAMPTZ NULL,
                unresolved_incident_count INTEGER NOT NULL DEFAULT 0,
                can_submit_live_now BOOLEAN NOT NULL DEFAULT FALSE,
                primary_block_reason_code TEXT NULL,
                block_detail TEXT NULL,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (service_name, execution_mode)
                )
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD COLUMN IF NOT EXISTS system_health_status TEXT NOT NULL DEFAULT 'UNKNOWN'
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD COLUMN IF NOT EXISTS system_health_reason_code TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD COLUMN IF NOT EXISTS system_health_checked_at TIMESTAMPTZ NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD COLUMN IF NOT EXISTS health_gate_status TEXT NOT NULL DEFAULT 'UNKNOWN'
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD COLUMN IF NOT EXISTS health_gate_reason_code TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD COLUMN IF NOT EXISTS health_gate_detail TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD COLUMN IF NOT EXISTS broker_cash DOUBLE PRECISION NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD COLUMN IF NOT EXISTS broker_equity DOUBLE PRECISION NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD COLUMN IF NOT EXISTS reconciliation_status TEXT NOT NULL DEFAULT 'UNKNOWN'
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD COLUMN IF NOT EXISTS reconciliation_reason_code TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD COLUMN IF NOT EXISTS reconciliation_checked_at TIMESTAMPTZ NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD COLUMN IF NOT EXISTS unresolved_incident_count INTEGER NOT NULL DEFAULT 0
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD COLUMN IF NOT EXISTS can_submit_live_now BOOLEAN NOT NULL DEFAULT FALSE
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD COLUMN IF NOT EXISTS primary_block_reason_code TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD COLUMN IF NOT EXISTS block_detail TEXT NULL
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                DROP CONSTRAINT IF EXISTS {live_safety_primary_key}
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._live_safety_table}
                ADD PRIMARY KEY (service_name, execution_mode)
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._heartbeats_table} (
                    id BIGSERIAL PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    heartbeat_at TIMESTAMPTZ NOT NULL,
                    health_overall_status TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    details TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {heartbeats_index}
                ON {self._heartbeats_table} (
                    service_name,
                    component_name,
                    heartbeat_at DESC,
                    id DESC
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._reliability_state_table} (
                    service_name TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    health_overall_status TEXT NOT NULL,
                    freshness_status TEXT NULL,
                    breaker_state TEXT NOT NULL,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    success_count INTEGER NOT NULL DEFAULT 0,
                    last_heartbeat_at TIMESTAMPTZ NULL,
                    last_success_at TIMESTAMPTZ NULL,
                    last_failure_at TIMESTAMPTZ NULL,
                    opened_at TIMESTAMPTZ NULL,
                    reason_code TEXT NULL,
                    details TEXT NULL,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    PRIMARY KEY (service_name, component_name)
                )
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._reliability_state_table}
                DROP CONSTRAINT IF EXISTS {reliability_state_primary_key}
                """
            )
            await connection.execute(
                f"""
                ALTER TABLE {self._reliability_state_table}
                ADD PRIMARY KEY (service_name, component_name)
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._reliability_events_table} (
                    id BIGSERIAL PRIMARY KEY,
                    service_name TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_time TIMESTAMPTZ NOT NULL,
                    reason_code TEXT NOT NULL,
                    health_overall_status TEXT NULL,
                    freshness_status TEXT NULL,
                    breaker_state TEXT NULL,
                    details TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {reliability_events_index}
                ON {self._reliability_events_table} (
                    service_name,
                    component_name,
                    event_time DESC,
                    id DESC
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._adaptive_drift_state_table} (
                    symbol TEXT NOT NULL,
                    regime_label TEXT NOT NULL,
                    detector_name TEXT NOT NULL,
                    window_id TEXT NOT NULL,
                    reference_window_start TIMESTAMPTZ NOT NULL,
                    reference_window_end TIMESTAMPTZ NOT NULL,
                    live_window_start TIMESTAMPTZ NOT NULL,
                    live_window_end TIMESTAMPTZ NOT NULL,
                    drift_score DOUBLE PRECISION NOT NULL,
                    warning_threshold DOUBLE PRECISION NOT NULL,
                    breach_threshold DOUBLE PRECISION NOT NULL,
                    status TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    detail TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {adaptive_drift_unique}
                ON {self._adaptive_drift_state_table} (
                    symbol,
                    regime_label,
                    detector_name,
                    window_id
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {adaptive_drift_index}
                ON {self._adaptive_drift_state_table} (symbol, regime_label, updated_at DESC)
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._adaptive_performance_windows_table} (
                    execution_mode TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    regime_label TEXT NOT NULL,
                    window_id TEXT NOT NULL,
                    window_type TEXT NOT NULL,
                    window_start TIMESTAMPTZ NOT NULL,
                    window_end TIMESTAMPTZ NOT NULL,
                    trade_count INTEGER NOT NULL,
                    net_pnl_after_costs DOUBLE PRECISION NOT NULL,
                    max_drawdown DOUBLE PRECISION NOT NULL,
                    profit_factor DOUBLE PRECISION NOT NULL,
                    expectancy DOUBLE PRECISION NOT NULL,
                    win_rate DOUBLE PRECISION NOT NULL,
                    precision DOUBLE PRECISION NOT NULL,
                    avg_slippage_bps DOUBLE PRECISION NOT NULL,
                    blocked_trade_rate DOUBLE PRECISION NOT NULL,
                    shadow_divergence_rate DOUBLE PRECISION NOT NULL,
                    health_context JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                CREATE UNIQUE INDEX IF NOT EXISTS {adaptive_performance_unique}
                ON {self._adaptive_performance_windows_table} (
                    execution_mode,
                    symbol,
                    regime_label,
                    window_id
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {adaptive_performance_index}
                ON {self._adaptive_performance_windows_table} (
                    execution_mode,
                    symbol,
                    regime_label,
                    window_end DESC
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._adaptive_profiles_table} (
                    profile_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    execution_mode_scope TEXT NOT NULL,
                    symbol_scope TEXT NOT NULL,
                    regime_scope TEXT NOT NULL,
                    threshold_policy_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    sizing_policy_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    calibration_profile_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    source_evidence_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    rollback_target_profile_id TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    approved_at TIMESTAMPTZ NULL,
                    activated_at TIMESTAMPTZ NULL,
                    superseded_at TIMESTAMPTZ NULL
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {adaptive_profiles_status_index}
                ON {self._adaptive_profiles_table} (
                    status,
                    execution_mode_scope,
                    symbol_scope,
                    regime_scope,
                    activated_at DESC
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._adaptive_challenger_runs_table} (
                    challenger_run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    train_window_start TIMESTAMPTZ NOT NULL,
                    train_window_end TIMESTAMPTZ NOT NULL,
                    validation_window_start TIMESTAMPTZ NOT NULL,
                    validation_window_end TIMESTAMPTZ NOT NULL,
                    shadow_window_start TIMESTAMPTZ NOT NULL,
                    shadow_window_end TIMESTAMPTZ NOT NULL,
                    candidate_model_version TEXT NOT NULL,
                    config_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    metrics_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    shadow_summary_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    artifact_paths_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    reason_codes TEXT[] NOT NULL DEFAULT '{{}}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {adaptive_challengers_index}
                ON {self._adaptive_challenger_runs_table} (updated_at DESC, created_at DESC)
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._adaptive_promotion_decisions_table} (
                    decision_id TEXT PRIMARY KEY,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    incumbent_id TEXT NULL,
                    decision TEXT NOT NULL,
                    metrics_delta_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    safety_checks_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    research_integrity_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    reason_codes TEXT[] NOT NULL DEFAULT '{{}}',
                    summary_text TEXT NOT NULL,
                    decided_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {adaptive_promotions_index}
                ON {self._adaptive_promotion_decisions_table} (decided_at DESC, decision_id DESC)
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._ensemble_profiles_table} (
                    profile_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    approval_stage TEXT NOT NULL,
                    candidate_roster_json JSONB NOT NULL DEFAULT '[]'::jsonb,
                    regime_weight_matrix_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    agreement_policy_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    evidence_summary_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    rollback_target_profile_id TEXT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    approved_at TIMESTAMPTZ NULL,
                    activated_at TIMESTAMPTZ NULL,
                    superseded_at TIMESTAMPTZ NULL
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {ensemble_profiles_status_index}
                ON {self._ensemble_profiles_table} (
                    status,
                    activated_at DESC
                )
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._ensemble_challenger_runs_table} (
                    challenger_run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    config_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    metrics_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    reason_codes TEXT[] NOT NULL DEFAULT '{{}}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {ensemble_challengers_index}
                ON {self._ensemble_challenger_runs_table} (updated_at DESC)
                """
            )
            await connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._ensemble_promotion_decisions_table} (
                    decision_id TEXT PRIMARY KEY,
                    target_type TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    incumbent_id TEXT NULL,
                    decision TEXT NOT NULL,
                    metrics_delta_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    safety_checks_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    reason_codes TEXT[] NOT NULL DEFAULT '{{}}',
                    summary_text TEXT NOT NULL,
                    decided_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            await connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {ensemble_promotions_index}
                ON {self._ensemble_promotion_decisions_table} (decided_at DESC, decision_id DESC)
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
            order_request_id=(
                None
                if row["pending_order_request_id"] is None
                else int(row["pending_order_request_id"])
            ),
            order_request_idempotency_key=(
                None
                if row["pending_order_idempotency_key"] is None
                else str(row["pending_order_idempotency_key"])
            ),
            risk_reason_codes=_text_array_to_tuple(row["pending_risk_reason_codes"]),
        )
    return PaperEngineState(
        service_name=str(row["service_name"]),
        symbol=str(row["symbol"]),
        execution_mode=str(row["execution_mode"]),
        last_processed_interval_begin=row["last_processed_interval_begin"],
        cooldown_until_interval_begin=row["cooldown_until_interval_begin"],
        pending_signal=pending_signal,
        pending_decision_trace_id=(
            None
            if row["pending_decision_trace_id"] is None
            else int(row["pending_decision_trace_id"])
        ),
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
        execution_mode=str(row["execution_mode"]),
        entry_order_request_id=(
            None
            if row["entry_order_request_id"] is None
            else int(row["entry_order_request_id"])
        ),
        entry_decision_trace_id=(
            None
            if row["entry_decision_trace_id"] is None
            else int(row["entry_decision_trace_id"])
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
        exit_order_request_id=(
            None
            if row["exit_order_request_id"] is None
            else int(row["exit_order_request_id"])
        ),
        exit_decision_trace_id=(
            None
            if row["exit_decision_trace_id"] is None
            else int(row["exit_decision_trace_id"])
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
        execution_mode=str(row["execution_mode"]),
        loss_streak_cooldown_until_interval_begin=row["loss_streak_cooldown_until_interval_begin"],
        kill_switch_enabled=bool(row["kill_switch_enabled"]),
        updated_at=row["updated_at"],
    )


def _live_safety_state_from_row(row: asyncpg.Record) -> LiveSafetyState:
    return LiveSafetyState(
        service_name=str(row["service_name"]),
        execution_mode=str(row["execution_mode"]),
        broker_name=str(row["broker_name"]),
        live_enabled=bool(row["live_enabled"]),
        startup_checks_passed=bool(row["startup_checks_passed"]),
        startup_checks_passed_at=row["startup_checks_passed_at"],
        account_validated=bool(row["account_validated"]),
        account_id=None if row["account_id"] is None else str(row["account_id"]),
        environment_name=(
            None if row["environment_name"] is None else str(row["environment_name"])
        ),
        manual_disable_active=bool(row["manual_disable_active"]),
        consecutive_live_failures=int(row["consecutive_live_failures"]),
        failure_hard_stop_active=bool(row["failure_hard_stop_active"]),
        last_failure_reason=(
            None
            if row["last_failure_reason"] is None
            else str(row["last_failure_reason"])
        ),
        system_health_status=str(row["system_health_status"]),
        system_health_reason_code=(
            None
            if row["system_health_reason_code"] is None
            else str(row["system_health_reason_code"])
        ),
        system_health_checked_at=row["system_health_checked_at"],
        health_gate_status=str(row["health_gate_status"]),
        health_gate_reason_code=(
            None
            if row["health_gate_reason_code"] is None
            else str(row["health_gate_reason_code"])
        ),
        health_gate_detail=(
            None
            if row["health_gate_detail"] is None
            else str(row["health_gate_detail"])
        ),
        broker_cash=None if row["broker_cash"] is None else float(row["broker_cash"]),
        broker_equity=(
            None if row["broker_equity"] is None else float(row["broker_equity"])
        ),
        reconciliation_status=str(row["reconciliation_status"]),
        reconciliation_reason_code=(
            None
            if row["reconciliation_reason_code"] is None
            else str(row["reconciliation_reason_code"])
        ),
        reconciliation_checked_at=row["reconciliation_checked_at"],
        unresolved_incident_count=int(row["unresolved_incident_count"]),
        can_submit_live_now=bool(row["can_submit_live_now"]),
        primary_block_reason_code=(
            None
            if row["primary_block_reason_code"] is None
            else str(row["primary_block_reason_code"])
        ),
        block_detail=(
            None if row["block_detail"] is None else str(row["block_detail"])
        ),
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


def _reliability_state_from_row(row: asyncpg.Record) -> ReliabilityState:
    return ReliabilityState(
        service_name=str(row["service_name"]),
        component_name=str(row["component_name"]),
        health_overall_status=str(row["health_overall_status"]),
        freshness_status=(
            None if row["freshness_status"] is None else str(row["freshness_status"])
        ),
        breaker_state=str(row["breaker_state"]),
        failure_count=int(row["failure_count"]),
        success_count=int(row["success_count"]),
        last_heartbeat_at=row["last_heartbeat_at"],
        last_success_at=row["last_success_at"],
        last_failure_at=row["last_failure_at"],
        opened_at=row["opened_at"],
        reason_code=None if row["reason_code"] is None else str(row["reason_code"]),
        detail=None if row["details"] is None else str(row["details"]),
        updated_at=row["updated_at"],
    )


def _recovery_event_from_row(row: asyncpg.Record) -> RecoveryEvent:
    return RecoveryEvent(
        service_name=str(row["service_name"]),
        component_name=str(row["component_name"]),
        event_type=str(row["event_type"]),
        event_time=row["event_time"],
        reason_code=str(row["reason_code"]),
        health_overall_status=(
            None
            if row["health_overall_status"] is None
            else str(row["health_overall_status"])
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


def _order_request_from_row(row: asyncpg.Record) -> OrderRequest:
    return OrderRequest(
        service_name=str(row["service_name"]),
        execution_mode=str(row["execution_mode"]),
        symbol=str(row["symbol"]),
        action=str(row["action"]),
        signal_interval_begin=row["signal_interval_begin"],
        signal_as_of_time=row["signal_as_of_time"],
        signal_row_id=str(row["signal_row_id"]),
        target_fill_interval_begin=row["target_fill_interval_begin"],
        requested_notional=float(row["requested_notional"]),
        approved_notional=float(row["approved_notional"]),
        idempotency_key=str(row["idempotency_key"]),
        model_name=None if row["model_name"] is None else str(row["model_name"]),
        model_version=(
            None if row["model_version"] is None else str(row["model_version"])
        ),
        confidence=None if row["confidence"] is None else float(row["confidence"]),
        regime_label=None if row["regime_label"] is None else str(row["regime_label"]),
        regime_run_id=None if row["regime_run_id"] is None else str(row["regime_run_id"]),
        risk_outcome=None if row["risk_outcome"] is None else str(row["risk_outcome"]),
        risk_reason_codes=_text_array_to_tuple(row["risk_reason_codes"]),
        decision_trace_id=(
            None
            if row["decision_trace_id"] is None
            else int(row["decision_trace_id"])
        ),
        order_request_id=int(row["id"]),
        created_at=row["created_at"],
    )


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
        payload=DecisionTracePayload.model_validate(_jsonb_to_object(row["trace_payload"])),
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


def _adaptive_drift_state_from_row(row: asyncpg.Record) -> AdaptiveDriftRecord:
    return AdaptiveDriftRecord(
        symbol=str(row["symbol"]),
        regime_label=str(row["regime_label"]),
        detector_name=str(row["detector_name"]),
        window_id=str(row["window_id"]),
        reference_window_start=row["reference_window_start"],
        reference_window_end=row["reference_window_end"],
        live_window_start=row["live_window_start"],
        live_window_end=row["live_window_end"],
        drift_score=float(row["drift_score"]),
        warning_threshold=float(row["warning_threshold"]),
        breach_threshold=float(row["breach_threshold"]),
        status=str(row["status"]),
        reason_code=str(row["reason_code"]),
        detail=None if row["detail"] is None else str(row["detail"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _adaptive_performance_window_from_row(
    row: asyncpg.Record,
) -> AdaptivePerformanceWindow:
    return AdaptivePerformanceWindow(
        execution_mode=str(row["execution_mode"]),
        symbol=str(row["symbol"]),
        regime_label=str(row["regime_label"]),
        window_id=str(row["window_id"]),
        window_type=str(row["window_type"]),
        window_start=row["window_start"],
        window_end=row["window_end"],
        trade_count=int(row["trade_count"]),
        net_pnl_after_costs=float(row["net_pnl_after_costs"]),
        max_drawdown=float(row["max_drawdown"]),
        profit_factor=float(row["profit_factor"]),
        expectancy=float(row["expectancy"]),
        win_rate=float(row["win_rate"]),
        precision=float(row["precision"]),
        avg_slippage_bps=float(row["avg_slippage_bps"]),
        blocked_trade_rate=float(row["blocked_trade_rate"]),
        shadow_divergence_rate=float(row["shadow_divergence_rate"]),
        health_context=dict(_jsonb_to_object(row["health_context"])),
        created_at=row["created_at"],
    )


def _adaptive_profile_from_row(row: asyncpg.Record) -> AdaptiveProfileRecord:
    return AdaptiveProfileRecord.model_validate(
        {
            "profile_id": str(row["profile_id"]),
            "status": str(row["status"]),
            "execution_mode_scope": str(row["execution_mode_scope"]),
            "symbol_scope": str(row["symbol_scope"]),
            "regime_scope": str(row["regime_scope"]),
            "threshold_policy_json": _jsonb_to_object(row["threshold_policy_json"]),
            "sizing_policy_json": _jsonb_to_object(row["sizing_policy_json"]),
            "calibration_profile_json": _jsonb_to_object(row["calibration_profile_json"]),
            "source_evidence_json": _jsonb_to_object(row["source_evidence_json"]),
            "rollback_target_profile_id": row["rollback_target_profile_id"],
            "created_at": row["created_at"],
            "approved_at": row["approved_at"],
            "activated_at": row["activated_at"],
            "superseded_at": row["superseded_at"],
        }
    )


def _adaptive_challenger_run_from_row(
    row: asyncpg.Record,
) -> AdaptiveChallengerRunRecord:
    return AdaptiveChallengerRunRecord.model_validate(
        {
            "challenger_run_id": str(row["challenger_run_id"]),
            "status": str(row["status"]),
            "train_window_start": row["train_window_start"],
            "train_window_end": row["train_window_end"],
            "validation_window_start": row["validation_window_start"],
            "validation_window_end": row["validation_window_end"],
            "shadow_window_start": row["shadow_window_start"],
            "shadow_window_end": row["shadow_window_end"],
            "candidate_model_version": str(row["candidate_model_version"]),
            "config_json": _jsonb_to_object(row["config_json"]),
            "metrics_json": _jsonb_to_object(row["metrics_json"]),
            "shadow_summary_json": _jsonb_to_object(row["shadow_summary_json"]),
            "artifact_paths_json": _jsonb_to_object(row["artifact_paths_json"]),
            "reason_codes": list(_text_array_to_tuple(row["reason_codes"])),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
    )


def _adaptive_promotion_decision_from_row(
    row: asyncpg.Record,
) -> AdaptivePromotionDecisionRecord:
    return AdaptivePromotionDecisionRecord.model_validate(
        {
            "decision_id": str(row["decision_id"]),
            "target_type": str(row["target_type"]),
            "target_id": str(row["target_id"]),
            "incumbent_id": row["incumbent_id"],
            "decision": str(row["decision"]),
            "metrics_delta_json": _jsonb_to_object(row["metrics_delta_json"]),
            "safety_checks_json": _jsonb_to_object(row["safety_checks_json"]),
            "research_integrity_json": _jsonb_to_object(row["research_integrity_json"]),
            "reason_codes": list(_text_array_to_tuple(row["reason_codes"])),
            "summary_text": str(row["summary_text"]),
            "decided_at": row["decided_at"],
        }
    )


def _ensemble_profile_from_row(row: asyncpg.Record) -> EnsembleProfileRecord:
    roster_raw = row["candidate_roster_json"]
    if isinstance(roster_raw, str):
        roster_raw = json.loads(roster_raw)
    return EnsembleProfileRecord.model_validate(
        {
            "profile_id": str(row["profile_id"]),
            "status": str(row["status"]),
            "approval_stage": str(row["approval_stage"]),
            "candidate_roster_json": roster_raw if isinstance(roster_raw, list) else [],
            "regime_weight_matrix_json": _jsonb_to_object(row["regime_weight_matrix_json"]),
            "agreement_policy_json": _jsonb_to_object(row["agreement_policy_json"]),
            "evidence_summary_json": _jsonb_to_object(row["evidence_summary_json"]),
            "rollback_target_profile_id": row["rollback_target_profile_id"],
            "created_at": row["created_at"],
            "approved_at": row["approved_at"],
            "activated_at": row["activated_at"],
            "superseded_at": row["superseded_at"],
        }
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
            None
            if row["external_order_id"] is None
            else str(row["external_order_id"])
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
            None
            if row["decision_trace_id"] is None
            else int(row["decision_trace_id"])
        ),
        event_id=int(row["id"]),
        created_at=row["created_at"],
    )


def ledger_rows_to_csv(entries: Sequence[TradeLedgerEntry]) -> list[dict[str, object]]:
    """Serialize ledger entries for optional artifact writing."""
    rows: list[dict[str, object]] = []
    for entry in entries:
        rows.append(
            {
                "service_name": entry.service_name,
                "execution_mode": entry.execution_mode,
                "position_id": entry.position_id,
                "order_request_id": entry.order_request_id,
                "decision_trace_id": entry.decision_trace_id,
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


def _jsonb_to_object(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _coerce_date(value: date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    return value
