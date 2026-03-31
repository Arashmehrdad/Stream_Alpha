"""M17 operational alert evaluation and artifact writers."""

# pylint: disable=too-many-lines,too-many-instance-attributes
# pylint: disable=too-many-arguments,too-many-locals
# pylint: disable=too-many-branches,too-many-statements

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from app.alerting.config import AlertingConfig
from app.alerting.repository import OperationalAlertRepository
from app.alerting.schemas import (
    DailyOperationsSummary,
    OperationalAlertEvent,
    OperationalAlertState,
    StartupSafetyReport,
    StartupSafetySection,
)
from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339, utc_now
from app.reliability.service import FEATURE_LAG_BREACH, FEED_STALE
from app.trading.live import LIVE_HEALTH_GATE_CLEAR, LIVE_RECONCILIATION_CLEAR
from app.trading.risk_engine import MAX_DRAWDOWN_BREACHED, current_drawdown_pct
from app.trading.schemas import (
    CanonicalSystemReliability,
    DecisionTraceRecord,
    LiveSafetyState,
    OrderLifecycleEvent,
    ServiceRiskState,
)


DAILY_SUMMARY_SCHEMA_VERSION = "m17_daily_operations_summary_v1"
STARTUP_SAFETY_REPORT_SCHEMA_VERSION = "m17_startup_safety_report_v1"
ORDER_FAILURE_SPIKE_WARNING = "ORDER_FAILURE_SPIKE_WARNING"
ORDER_FAILURE_SPIKE_CRITICAL = "ORDER_FAILURE_SPIKE_CRITICAL"
ORDER_FAILURE_SPIKE_CLEARED = "ORDER_FAILURE_SPIKE_CLEARED"
SIGNAL_SILENCE_DETECTED = "SIGNAL_SILENCE_DETECTED"
SIGNAL_SILENCE_CLEARED = "SIGNAL_SILENCE_CLEARED"
SIGNAL_FLOOD_WARNING = "SIGNAL_FLOOD_WARNING"
SIGNAL_FLOOD_CRITICAL = "SIGNAL_FLOOD_CRITICAL"
SIGNAL_FLOOD_CLEARED = "SIGNAL_FLOOD_CLEARED"
DRAWDOWN_BREACH_CLEARED = "DRAWDOWN_BREACH_CLEARED"
LIVE_MODE_ACTIVATED = "LIVE_MODE_ACTIVATED"
STARTUP_SAFETY_PASSED = "STARTUP_SAFETY_PASSED"
STARTUP_VALIDATION_MISSING = "STARTUP_VALIDATION_MISSING"
STARTUP_VALIDATION_FAILED = "STARTUP_VALIDATION_FAILED"
STARTUP_SAFETY_LIVE_STATE_MISSING = "STARTUP_SAFETY_LIVE_STATE_MISSING"
_FAILURE_LIFECYCLE_STATES = {"REJECTED", "FAILED"}
_ACTIONABLE_SIGNALS = {"BUY", "SELL"}
_CATEGORY_ORDER = (
    "FEED_STALE",
    "CONSUMER_LAG",
    "ORDER_FAILURE_SPIKE",
    "DRAWDOWN_BREACH",
    "SIGNAL_SILENCE",
    "SIGNAL_FLOOD",
    "LIVE_MODE_ACTIVATION",
    "STARTUP_SAFETY",
)
_SEVERITY_RANK = {"INFO": 0, "WARNING": 1, "CRITICAL": 2}


@dataclass(frozen=True, slots=True)
class AlertObservation:
    """One evaluated alert condition before OPEN, UPDATE, or CLEAR persistence."""

    service_name: str
    execution_mode: str
    category: str
    severity: str
    reason_code: str
    source_component: str
    fingerprint: str
    summary_text: str
    event_time: datetime
    is_active: bool
    symbol: str | None = None
    detail: str | None = None
    related_order_request_id: int | None = None
    related_decision_trace_id: int | None = None
    payload_json: dict[str, Any] | None = None


class OperationalAlertService:
    """Evaluate existing runtime truths into normalized M17 alert events and state."""

    def __init__(
        self,
        *,
        config: AlertingConfig,
        repository: OperationalAlertRepository | None = None,
    ) -> None:
        self.config = config
        self.repository = repository

    def required_lookback_start(
        self,
        *,
        evaluated_at: datetime,
        interval_minutes: int,
    ) -> datetime:
        """Return the earliest timestamp needed for M17 daily and windowed checks."""
        signal_window_minutes = max(
            self.config.signals.silence_window_intervals,
            self.config.signals.flood_window_intervals,
        ) * interval_minutes
        max_window_minutes = max(
            self.config.order_failure_spike.window_minutes,
            signal_window_minutes,
        )
        day_start = datetime.combine(
            evaluated_at.date(),
            time.min,
            tzinfo=evaluated_at.tzinfo or timezone.utc,
        )
        return min(day_start, evaluated_at - timedelta(minutes=max_window_minutes))

    async def evaluate_cycle(
        self,
        *,
        service_name: str,
        execution_mode: str,
        interval_minutes: int,
        evaluated_at: datetime,
        system_reliability: CanonicalSystemReliability | None,
        service_risk_state: ServiceRiskState | None,
        order_events: Sequence[OrderLifecycleEvent],
        decision_traces: Sequence[DecisionTraceRecord],
        max_drawdown_pct: float,
    ) -> tuple[OperationalAlertEvent, ...]:
        """Evaluate the current runner truths into normalized alert timeline events."""
        observations = self._build_cycle_observations(
            service_name=service_name,
            execution_mode=execution_mode,
            interval_minutes=interval_minutes,
            evaluated_at=evaluated_at,
            system_reliability=system_reliability,
            service_risk_state=service_risk_state,
            order_events=order_events,
            decision_traces=decision_traces,
            max_drawdown_pct=max_drawdown_pct,
        )
        return await self._persist_observations(observations)

    async def record_live_mode_activation(
        self,
        *,
        service_name: str,
        execution_mode: str,
        runtime_profile: str,
        live_safety_state: LiveSafetyState,
        event_time: datetime,
    ) -> tuple[OperationalAlertEvent, ...]:
        """Record one INFO alert when guarded live mode is activated."""
        observation = AlertObservation(
            service_name=service_name,
            execution_mode=execution_mode,
            category="LIVE_MODE_ACTIVATION",
            severity="INFO",
            reason_code=LIVE_MODE_ACTIVATED,
            source_component="live_safety",
            fingerprint=build_alert_fingerprint(
                service_name=service_name,
                execution_mode=execution_mode,
                category="LIVE_MODE_ACTIVATION",
                source_component="live_safety",
            ),
            summary_text=(
                "Guarded live mode has been activated "
                f"for account {live_safety_state.account_id or 'unknown'} "
                f"in {live_safety_state.environment_name or 'unknown'}."
            ),
            detail=live_safety_state.block_detail,
            event_time=event_time,
            is_active=False,
            payload_json={
                "runtime_profile": runtime_profile,
                "account_id": live_safety_state.account_id,
                "environment_name": live_safety_state.environment_name,
                "startup_checks_passed": live_safety_state.startup_checks_passed,
                "reconciliation_status": live_safety_state.reconciliation_status,
                "health_gate_status": live_safety_state.health_gate_status,
                "can_submit_live_now": live_safety_state.can_submit_live_now,
                "primary_block_reason_code": live_safety_state.primary_block_reason_code,
            },
        )
        if self.repository is None:
            return ()
        event = await self._persist_info_event(observation)
        return () if event is None else (event,)

    async def write_startup_safety_artifact(
        self,
        *,
        service_name: str,
        execution_mode: str,
        runtime_profile: str,
        startup_validation_passed: bool | None,
        startup_report_path: str,
        live_safety_state: LiveSafetyState | None,
        live_startup_checklist_path: str | None,
    ) -> tuple[StartupSafetyReport, tuple[OperationalAlertEvent, ...]]:
        """Compose the canonical startup-safety artifact and persist its alert state."""
        report = self.compose_startup_safety_report(
            service_name=service_name,
            execution_mode=execution_mode,
            runtime_profile=runtime_profile,
            startup_validation_passed=startup_validation_passed,
            startup_report_path=startup_report_path,
            live_safety_state=live_safety_state,
            live_startup_checklist_path=live_startup_checklist_path,
        )
        self._write_json(Path(self.config.artifacts.startup_safety_path), asdict(report))
        events = await self._record_startup_safety_alert(report=report)
        return report, events

    async def write_daily_summary(
        self,
        *,
        service_name: str,
        execution_mode: str,
        runtime_profile: str,
        evaluated_at: datetime,
        service_risk_state: ServiceRiskState | None,
        max_drawdown_pct: float,
        order_events: Sequence[OrderLifecycleEvent],
        decision_traces: Sequence[DecisionTraceRecord],
        startup_safety_report: StartupSafetyReport,
    ) -> DailyOperationsSummary:
        """Write the canonical M17 daily JSON summary."""
        summary_date = evaluated_at.date()
        daily_events = (
            []
            if self.repository is None
            else await self.repository.load_events_for_day(
                service_name=service_name,
                execution_mode=execution_mode,
                summary_date=summary_date,
            )
        )
        active_states = (
            []
            if self.repository is None
            else await self.repository.load_active_states(
                service_name=service_name,
                execution_mode=execution_mode,
            )
        )
        daily_order_events = [
            event for event in order_events if event.event_time.date() == summary_date
        ]
        daily_decision_traces = [
            trace for trace in decision_traces if trace.signal_as_of_time.date() == summary_date
        ]
        actionable_traces = [
            trace for trace in daily_decision_traces if trace.signal in _ACTIONABLE_SIGNALS
        ]
        order_failure_events = [
            event
            for event in daily_order_events
            if event.lifecycle_state in _FAILURE_LIFECYCLE_STATES
        ]
        counts_by_category = {category: 0 for category in _CATEGORY_ORDER}
        for event in daily_events:
            counts_by_category[event.category] = counts_by_category.get(event.category, 0) + 1
        if startup_safety_report.primary_reason_code:
            counts_by_category["STARTUP_SAFETY"] = max(
                counts_by_category.get("STARTUP_SAFETY", 0),
                1,
            )

        highest_severity = "INFO"
        for event in daily_events:
            if _SEVERITY_RANK[event.severity] > _SEVERITY_RANK[highest_severity]:
                highest_severity = event.severity
        for state in active_states:
            if _SEVERITY_RANK[state.severity] > _SEVERITY_RANK[highest_severity]:
                highest_severity = state.severity
        startup_safety_severity = (
            "INFO" if startup_safety_report.startup_safety_passed else "CRITICAL"
        )
        if _SEVERITY_RANK[startup_safety_severity] > _SEVERITY_RANK[highest_severity]:
            highest_severity = startup_safety_severity

        summary = DailyOperationsSummary(
            schema_version=DAILY_SUMMARY_SCHEMA_VERSION,
            generated_at=to_rfc3339(evaluated_at),
            service_name=service_name,
            execution_mode=execution_mode,
            runtime_profile=runtime_profile,
            summary_date=summary_date.isoformat(),
            counts_by_category=counts_by_category,
            unresolved_count=len(active_states),
            highest_severity=highest_severity,
            startup_safety_status={
                "startup_safety_passed": startup_safety_report.startup_safety_passed,
                "primary_reason_code": startup_safety_report.primary_reason_code,
                "summary_text": startup_safety_report.summary_text,
                "startup_report_path": startup_safety_report.startup_validation.report_path,
            },
            order_failure_counts={
                "rejected": len(
                    [event for event in order_failure_events if event.lifecycle_state == "REJECTED"]
                ),
                "failed": len(
                    [event for event in order_failure_events if event.lifecycle_state == "FAILED"]
                ),
                "total_failures": len(order_failure_events),
                "window_minutes": self.config.order_failure_spike.window_minutes,
            },
            drawdown_state=self._daily_drawdown_state(
                service_risk_state=service_risk_state,
                max_drawdown_pct=max_drawdown_pct,
            ),
            actionable_signal_counts={
                "buy_count": len([trace for trace in actionable_traces if trace.signal == "BUY"]),
                "sell_count": len([trace for trace in actionable_traces if trace.signal == "SELL"]),
                "total_actionable": len(actionable_traces),
                "decision_trace_count": len(daily_decision_traces),
            },
            silence_flood_episodes={
                "signal_silence_events": len(
                    [
                        event
                        for event in daily_events
                        if event.category == "SIGNAL_SILENCE" and event.event_state == "OPEN"
                    ]
                ),
                "signal_flood_events": len(
                    [
                        event
                        for event in daily_events
                        if event.category == "SIGNAL_FLOOD" and event.event_state == "OPEN"
                    ]
                ),
            },
            live_mode_activation_count=len(
                [
                    event
                    for event in daily_events
                    if event.category == "LIVE_MODE_ACTIVATION"
                ]
            ),
        )
        output_path = (
            Path(self.config.artifacts.daily_summary_dir)
            / f"{summary_date.isoformat()}.json"
        )
        self._write_json(output_path, asdict(summary))
        return summary

    def compose_startup_safety_report(
        self,
        *,
        service_name: str,
        execution_mode: str,
        runtime_profile: str,
        startup_validation_passed: bool | None,
        startup_report_path: str,
        live_safety_state: LiveSafetyState | None,
        live_startup_checklist_path: str | None,
    ) -> StartupSafetyReport:
        """Compose the canonical startup-safety payload from existing M12 and M16 truth."""
        startup_validation_payload = self._load_json(Path(startup_report_path))
        checklist_payload = (
            None
            if not live_startup_checklist_path
            else self._load_json(Path(live_startup_checklist_path))
        )

        validation_passed = startup_validation_passed
        if validation_passed is None and startup_validation_payload is not None:
            raw_passed = startup_validation_payload.get("startup_validation_passed")
            if raw_passed is not None:
                validation_passed = bool(raw_passed)

        if validation_passed is True:
            validation_reason = STARTUP_SAFETY_PASSED
            validation_summary = "M16 startup validation passed."
        elif validation_passed is False:
            validation_reason = STARTUP_VALIDATION_FAILED
            validation_summary = "M16 startup validation failed."
        else:
            validation_reason = STARTUP_VALIDATION_MISSING
            validation_summary = "M16 startup validation report is missing."

        startup_validation = StartupSafetySection(
            report_path=startup_report_path,
            report_exists=startup_validation_payload is not None,
            startup_validation_passed=validation_passed,
            primary_reason_code=validation_reason,
            summary_text=validation_summary,
            detail=_coerce_errors(startup_validation_payload),
            payload={} if startup_validation_payload is None else startup_validation_payload,
        )

        if execution_mode != "live":
            live_section = StartupSafetySection(
                report_path=startup_report_path,
                report_exists=False,
                startup_validation_passed=True,
                primary_reason_code=STARTUP_SAFETY_PASSED,
                summary_text="M12 guarded-live startup checks are not required for this mode.",
            )
            startup_safety_passed = validation_passed is True
            primary_reason_code = (
                STARTUP_SAFETY_PASSED if startup_safety_passed else validation_reason
            )
            summary_text = (
                "Startup safety is clear for this non-live runtime."
                if startup_safety_passed
                else validation_summary
            )
            return StartupSafetyReport(
                schema_version=STARTUP_SAFETY_REPORT_SCHEMA_VERSION,
                generated_at=to_rfc3339(utc_now()),
                service_name=service_name,
                execution_mode=execution_mode,
                runtime_profile=runtime_profile,
                startup_safety_passed=startup_safety_passed,
                primary_reason_code=primary_reason_code,
                summary_text=summary_text,
                startup_validation=startup_validation,
                live_startup=live_section,
            )

        if live_safety_state is None:
            live_reason = STARTUP_SAFETY_LIVE_STATE_MISSING
            live_summary = "Live safety state is missing."
            live_detail = "No M12 live safety state was available during startup composition."
            live_passed = False
        elif not live_safety_state.startup_checks_passed:
            live_reason = (
                live_safety_state.primary_block_reason_code
                or STARTUP_SAFETY_LIVE_STATE_MISSING
            )
            live_summary = "Live startup safety checks did not pass."
            live_detail = live_safety_state.block_detail
            live_passed = False
        elif live_safety_state.reconciliation_status != "CLEAR":
            live_reason = (
                live_safety_state.reconciliation_reason_code
                or LIVE_RECONCILIATION_CLEAR
            )
            live_summary = "Live reconciliation is not clear."
            live_detail = live_safety_state.block_detail
            live_passed = False
        elif live_safety_state.health_gate_status != "CLEAR":
            live_reason = (
                live_safety_state.health_gate_reason_code or LIVE_HEALTH_GATE_CLEAR
            )
            live_summary = "Live health gate is not clear."
            live_detail = live_safety_state.health_gate_detail
            live_passed = False
        elif not live_safety_state.can_submit_live_now:
            live_reason = (
                live_safety_state.primary_block_reason_code
                or STARTUP_SAFETY_LIVE_STATE_MISSING
            )
            live_summary = "Live submit is blocked by the guarded-live gate."
            live_detail = live_safety_state.block_detail
            live_passed = False
        else:
            live_reason = STARTUP_SAFETY_PASSED
            live_summary = "Live startup safety checks passed."
            live_detail = None
            live_passed = True

        live_payload = {
            "account_validated": (
                None if live_safety_state is None else live_safety_state.account_validated
            ),
            "account_id": None if live_safety_state is None else live_safety_state.account_id,
            "environment_name": (
                None if live_safety_state is None else live_safety_state.environment_name
            ),
            "startup_checks_passed": (
                None
                if live_safety_state is None
                else live_safety_state.startup_checks_passed
            ),
            "reconciliation_status": (
                None
                if live_safety_state is None
                else live_safety_state.reconciliation_status
            ),
            "health_gate_status": (
                None if live_safety_state is None else live_safety_state.health_gate_status
            ),
            "can_submit_live_now": (
                None if live_safety_state is None else live_safety_state.can_submit_live_now
            ),
            "primary_block_reason_code": (
                None
                if live_safety_state is None
                else live_safety_state.primary_block_reason_code
            ),
        }
        if checklist_payload is not None:
            live_payload["startup_checklist"] = checklist_payload

        live_section = StartupSafetySection(
            report_path=startup_report_path,
            report_exists=live_safety_state is not None,
            checklist_path=live_startup_checklist_path,
            checklist_exists=checklist_payload is not None,
            checklist_passed=(
                None if checklist_payload is None else bool(checklist_payload.get("passed"))
            ),
            primary_reason_code=live_reason,
            summary_text=live_summary,
            detail=live_detail,
            payload=live_payload,
        )

        startup_safety_passed = validation_passed is True and live_passed
        primary_reason_code = (
            STARTUP_SAFETY_PASSED
            if startup_safety_passed
            else validation_reason
            if validation_passed is not True
            else live_reason
        )
        summary_text = (
            "Startup safety is clear."
            if startup_safety_passed
            else validation_summary
            if validation_passed is not True
            else live_summary
        )
        return StartupSafetyReport(
            schema_version=STARTUP_SAFETY_REPORT_SCHEMA_VERSION,
            generated_at=to_rfc3339(utc_now()),
            service_name=service_name,
            execution_mode=execution_mode,
            runtime_profile=runtime_profile,
            startup_safety_passed=startup_safety_passed,
            primary_reason_code=primary_reason_code,
            summary_text=summary_text,
            startup_validation=startup_validation,
            live_startup=live_section,
        )

    def _build_cycle_observations(
        self,
        *,
        service_name: str,
        execution_mode: str,
        interval_minutes: int,
        evaluated_at: datetime,
        system_reliability: CanonicalSystemReliability | None,
        service_risk_state: ServiceRiskState | None,
        order_events: Sequence[OrderLifecycleEvent],
        decision_traces: Sequence[DecisionTraceRecord],
        max_drawdown_pct: float,
    ) -> tuple[AlertObservation, ...]:
        observations: list[AlertObservation] = []
        observations.extend(
            self._feed_stale_observations(
                service_name=service_name,
                execution_mode=execution_mode,
                evaluated_at=evaluated_at,
                system_reliability=system_reliability,
            )
        )
        observations.extend(
            self._consumer_lag_observations(
                service_name=service_name,
                execution_mode=execution_mode,
                evaluated_at=evaluated_at,
                system_reliability=system_reliability,
            )
        )
        observations.append(
            self._order_failure_spike_observation(
                service_name=service_name,
                execution_mode=execution_mode,
                evaluated_at=evaluated_at,
                order_events=order_events,
            )
        )
        drawdown_observation = self._drawdown_breach_observation(
            service_name=service_name,
            execution_mode=execution_mode,
            evaluated_at=evaluated_at,
            service_risk_state=service_risk_state,
            max_drawdown_pct=max_drawdown_pct,
        )
        if drawdown_observation is not None:
            observations.append(drawdown_observation)
        observations.append(
            self._signal_silence_observation(
                service_name=service_name,
                execution_mode=execution_mode,
                evaluated_at=evaluated_at,
                interval_minutes=interval_minutes,
                system_reliability=system_reliability,
                decision_traces=decision_traces,
            )
        )
        observations.append(
            self._signal_flood_observation(
                service_name=service_name,
                execution_mode=execution_mode,
                evaluated_at=evaluated_at,
                interval_minutes=interval_minutes,
                system_reliability=system_reliability,
                decision_traces=decision_traces,
            )
        )
        return tuple(observations)

    def _signal_cadence_gate(
        self,
        *,
        system_reliability: CanonicalSystemReliability | None,
    ) -> tuple[bool, str | None, str]:
        if system_reliability is None:
            return (
                False,
                "SYSTEM_RELIABILITY_UNAVAILABLE",
                (
                    "Signal cadence evaluation is suppressed because "
                    "system reliability is unavailable."
                ),
            )
        producer_snapshot = next(
            (
                snapshot
                for snapshot in system_reliability.services
                if snapshot.component_name == "producer"
            ),
            None,
        )
        if producer_snapshot is not None and producer_snapshot.feed_reason_code == FEED_STALE:
            return (
                False,
                FEED_STALE,
                "Signal cadence evaluation is suppressed because the producer feed is stale.",
            )
        if system_reliability.lag_breach_active:
            return (
                False,
                FEATURE_LAG_BREACH,
                "Signal cadence evaluation is suppressed because a consumer lag breach is active.",
            )
        return True, None, "Signal cadence evaluation is clear."

    def _feed_stale_observations(
        self,
        *,
        service_name: str,
        execution_mode: str,
        evaluated_at: datetime,
        system_reliability: CanonicalSystemReliability | None,
    ) -> list[AlertObservation]:
        if system_reliability is None:
            return []
        producer_snapshot = next(
            (
                snapshot
                for snapshot in system_reliability.services
                if snapshot.component_name == "producer"
            ),
            None,
        )
        if producer_snapshot is None or producer_snapshot.feed_reason_code is None:
            return []
        return [
            AlertObservation(
                service_name=service_name,
                execution_mode=execution_mode,
                category="FEED_STALE",
                severity="CRITICAL" if execution_mode == "live" else "WARNING",
                reason_code=producer_snapshot.feed_reason_code,
                source_component="producer",
                fingerprint=build_alert_fingerprint(
                    service_name=service_name,
                    execution_mode=execution_mode,
                    category="FEED_STALE",
                    source_component="producer",
                ),
                summary_text=(
                    "Producer feed is stale."
                    if producer_snapshot.feed_reason_code == FEED_STALE
                    else "Producer feed freshness has recovered."
                ),
                detail=producer_snapshot.detail,
                event_time=evaluated_at,
                is_active=producer_snapshot.feed_reason_code == FEED_STALE,
                payload_json={
                    "feed_freshness_status": producer_snapshot.feed_freshness_status,
                    "feed_reason_code": producer_snapshot.feed_reason_code,
                    "feed_age_seconds": producer_snapshot.feed_age_seconds,
                    "system_health_status": system_reliability.health_overall_status,
                    "system_reason_codes": list(system_reliability.reason_codes),
                },
            )
        ]

    def _consumer_lag_observations(
        self,
        *,
        service_name: str,
        execution_mode: str,
        evaluated_at: datetime,
        system_reliability: CanonicalSystemReliability | None,
    ) -> list[AlertObservation]:
        if system_reliability is None:
            return []
        observations: list[AlertObservation] = []
        for lag_snapshot in system_reliability.lag_by_symbol:
            observations.append(
                AlertObservation(
                    service_name=service_name,
                    execution_mode=execution_mode,
                    category="CONSUMER_LAG",
                    severity="CRITICAL" if execution_mode == "live" else "WARNING",
                    reason_code=lag_snapshot.reason_code,
                    source_component=lag_snapshot.component_name,
                    symbol=lag_snapshot.symbol,
                    fingerprint=build_alert_fingerprint(
                        service_name=service_name,
                        execution_mode=execution_mode,
                        category="CONSUMER_LAG",
                        source_component=lag_snapshot.component_name,
                        symbol=lag_snapshot.symbol,
                    ),
                    summary_text=(
                        f"Feature consumer lag breach for {lag_snapshot.symbol}."
                        if lag_snapshot.lag_breach
                        else f"Feature consumer lag recovered for {lag_snapshot.symbol}."
                    ),
                    detail=lag_snapshot.detail,
                    event_time=evaluated_at,
                    is_active=lag_snapshot.lag_breach,
                    payload_json={
                        "time_lag_seconds": lag_snapshot.time_lag_seconds,
                        "processing_lag_seconds": lag_snapshot.processing_lag_seconds,
                        "time_lag_reason_code": lag_snapshot.time_lag_reason_code,
                        "processing_lag_reason_code": lag_snapshot.processing_lag_reason_code,
                    },
                )
            )
        return observations

    def _order_failure_spike_observation(
        self,
        *,
        service_name: str,
        execution_mode: str,
        evaluated_at: datetime,
        order_events: Sequence[OrderLifecycleEvent],
    ) -> AlertObservation:
        window_start = evaluated_at - timedelta(
            minutes=self.config.order_failure_spike.window_minutes
        )
        failures = [
            event
            for event in order_events
            if event.event_time >= window_start
            and event.lifecycle_state in _FAILURE_LIFECYCLE_STATES
        ]
        latest_failure = failures[-1] if failures else None
        failure_count = len(failures)
        state_counts = {
            "REJECTED": len(
                [event for event in failures if event.lifecycle_state == "REJECTED"]
            ),
            "FAILED": len(
                [event for event in failures if event.lifecycle_state == "FAILED"]
            ),
        }
        if failure_count >= self.config.order_failure_spike.critical_count:
            severity = "CRITICAL"
            reason_code = ORDER_FAILURE_SPIKE_CRITICAL
            active = True
            summary = "Order failure count has breached the critical spike threshold."
        elif failure_count >= self.config.order_failure_spike.warning_count:
            severity = "WARNING"
            reason_code = ORDER_FAILURE_SPIKE_WARNING
            active = True
            summary = "Order failure count has breached the warning spike threshold."
        else:
            severity = "WARNING"
            reason_code = ORDER_FAILURE_SPIKE_CLEARED
            active = False
            summary = "Order failure spike has cleared."
        return AlertObservation(
            service_name=service_name,
            execution_mode=execution_mode,
            category="ORDER_FAILURE_SPIKE",
            severity=severity,
            reason_code=reason_code,
            source_component="execution",
            fingerprint=build_alert_fingerprint(
                service_name=service_name,
                execution_mode=execution_mode,
                category="ORDER_FAILURE_SPIKE",
                source_component="execution",
            ),
            summary_text=summary,
            detail=(
                f"window_minutes={self.config.order_failure_spike.window_minutes} "
                f"failure_count={failure_count}"
            ),
            event_time=evaluated_at,
            is_active=active,
            related_order_request_id=(
                None if latest_failure is None else latest_failure.order_request_id
            ),
            related_decision_trace_id=(
                None if latest_failure is None else latest_failure.decision_trace_id
            ),
            payload_json={
                "window_minutes": self.config.order_failure_spike.window_minutes,
                "warning_count": self.config.order_failure_spike.warning_count,
                "critical_count": self.config.order_failure_spike.critical_count,
                "failure_count": failure_count,
                "state_counts": state_counts,
            },
        )

    def _drawdown_breach_observation(
        self,
        *,
        service_name: str,
        execution_mode: str,
        evaluated_at: datetime,
        service_risk_state: ServiceRiskState | None,
        max_drawdown_pct: float,
    ) -> AlertObservation | None:
        if service_risk_state is None or max_drawdown_pct <= 0.0:
            return None
        drawdown_pct = current_drawdown_pct(service_risk_state)
        is_breached = drawdown_pct >= max_drawdown_pct
        return AlertObservation(
            service_name=service_name,
            execution_mode=execution_mode,
            category="DRAWDOWN_BREACH",
            severity="CRITICAL",
            reason_code=(
                MAX_DRAWDOWN_BREACHED if is_breached else DRAWDOWN_BREACH_CLEARED
            ),
            source_component="risk_engine",
            fingerprint=build_alert_fingerprint(
                service_name=service_name,
                execution_mode=execution_mode,
                category="DRAWDOWN_BREACH",
                source_component="risk_engine",
            ),
            summary_text=(
                "Portfolio drawdown has breached the configured limit."
                if is_breached
                else "Portfolio drawdown is back within the configured limit."
            ),
            detail=(
                f"drawdown_pct={drawdown_pct:.6f} "
                f"threshold={max_drawdown_pct:.6f}"
            ),
            event_time=evaluated_at,
            is_active=is_breached,
            payload_json={
                "drawdown_pct": drawdown_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "current_equity": service_risk_state.current_equity,
                "equity_high_watermark": service_risk_state.equity_high_watermark,
                "trading_day": service_risk_state.trading_day.isoformat(),
            },
        )

    def _signal_silence_observation(
        self,
        *,
        service_name: str,
        execution_mode: str,
        evaluated_at: datetime,
        interval_minutes: int,
        system_reliability: CanonicalSystemReliability | None,
        decision_traces: Sequence[DecisionTraceRecord],
    ) -> AlertObservation:
        window_minutes = self.config.signals.silence_window_intervals * interval_minutes
        window_start = evaluated_at - timedelta(minutes=window_minutes)
        cadence_allowed, suppression_reason_code, gating_detail = self._signal_cadence_gate(
            system_reliability=system_reliability,
        )
        actionable_traces = [
            trace
            for trace in decision_traces
            if trace.signal in _ACTIONABLE_SIGNALS
        ]
        actionable_traces_in_window = [
            trace
            for trace in actionable_traces
            if trace.signal_as_of_time >= window_start
        ]
        latest_trace = max(
            actionable_traces,
            key=lambda trace: trace.signal_as_of_time,
            default=None,
        )
        is_active = cadence_allowed and len(actionable_traces_in_window) == 0
        return AlertObservation(
            service_name=service_name,
            execution_mode=execution_mode,
            category="SIGNAL_SILENCE",
            severity="CRITICAL" if execution_mode == "live" else "WARNING",
            reason_code=(
                SIGNAL_SILENCE_DETECTED if is_active else SIGNAL_SILENCE_CLEARED
            ),
            source_component="decision_trace",
            fingerprint=build_alert_fingerprint(
                service_name=service_name,
                execution_mode=execution_mode,
                category="SIGNAL_SILENCE",
                source_component="decision_trace",
            ),
            summary_text=(
                "No actionable decision traces were recorded inside the configured silence window."
                if is_active
                else "Actionable decision trace activity has resumed."
                if cadence_allowed
                else "Signal silence evaluation is suppressed until runtime health is clear."
            ),
            detail=(
                f"window_intervals={self.config.signals.silence_window_intervals} "
                f"window_minutes={window_minutes} "
                f"actionable_trace_count={len(actionable_traces_in_window)} "
                f"{gating_detail}"
            ),
            event_time=evaluated_at,
            is_active=is_active,
            related_decision_trace_id=(
                None if latest_trace is None else latest_trace.decision_trace_id
            ),
            payload_json={
                "cadence_evaluation_allowed": cadence_allowed,
                "suppression_reason_code": suppression_reason_code,
                "window_intervals": self.config.signals.silence_window_intervals,
                "window_minutes": window_minutes,
                "decision_trace_count": len(decision_traces),
                "actionable_trace_count": len(actionable_traces_in_window),
                "last_actionable_signal_as_of_time": (
                    None
                    if latest_trace is None
                    else to_rfc3339(latest_trace.signal_as_of_time)
                ),
            },
        )

    def _signal_flood_observation(
        self,
        *,
        service_name: str,
        execution_mode: str,
        evaluated_at: datetime,
        interval_minutes: int,
        system_reliability: CanonicalSystemReliability | None,
        decision_traces: Sequence[DecisionTraceRecord],
    ) -> AlertObservation:
        window_minutes = self.config.signals.flood_window_intervals * interval_minutes
        window_start = evaluated_at - timedelta(minutes=window_minutes)
        cadence_allowed, suppression_reason_code, gating_detail = self._signal_cadence_gate(
            system_reliability=system_reliability,
        )
        actionable_traces = [
            trace
            for trace in decision_traces
            if trace.signal_as_of_time >= window_start and trace.signal in _ACTIONABLE_SIGNALS
        ]
        latest_trace = actionable_traces[-1] if actionable_traces else None
        actionable_count = len(actionable_traces)
        by_signal = {
            "BUY": len([trace for trace in actionable_traces if trace.signal == "BUY"]),
            "SELL": len([trace for trace in actionable_traces if trace.signal == "SELL"]),
        }
        if not cadence_allowed:
            severity = "WARNING"
            reason_code = SIGNAL_FLOOD_CLEARED
            is_active = False
            summary = "Signal flood evaluation is suppressed until runtime health is clear."
        elif actionable_count >= self.config.signals.flood_critical_count:
            severity = "CRITICAL"
            reason_code = SIGNAL_FLOOD_CRITICAL
            is_active = True
            summary = "Actionable signal volume has breached the critical flood threshold."
        elif actionable_count >= self.config.signals.flood_warning_count:
            severity = "WARNING"
            reason_code = SIGNAL_FLOOD_WARNING
            is_active = True
            summary = "Actionable signal volume has breached the warning flood threshold."
        else:
            severity = "WARNING"
            reason_code = SIGNAL_FLOOD_CLEARED
            is_active = False
            summary = "Actionable signal flood has cleared."
        return AlertObservation(
            service_name=service_name,
            execution_mode=execution_mode,
            category="SIGNAL_FLOOD",
            severity=severity,
            reason_code=reason_code,
            source_component="decision_trace",
            fingerprint=build_alert_fingerprint(
                service_name=service_name,
                execution_mode=execution_mode,
                category="SIGNAL_FLOOD",
                source_component="decision_trace",
            ),
            summary_text=summary,
            detail=(
                f"window_intervals={self.config.signals.flood_window_intervals} "
                f"window_minutes={window_minutes} actionable_count={actionable_count} "
                f"{gating_detail}"
            ),
            event_time=evaluated_at,
            is_active=is_active,
            related_decision_trace_id=(
                None if latest_trace is None else latest_trace.decision_trace_id
            ),
            payload_json={
                "cadence_evaluation_allowed": cadence_allowed,
                "suppression_reason_code": suppression_reason_code,
                "window_intervals": self.config.signals.flood_window_intervals,
                "window_minutes": window_minutes,
                "warning_count": self.config.signals.flood_warning_count,
                "critical_count": self.config.signals.flood_critical_count,
                "actionable_count": actionable_count,
                "by_signal": by_signal,
            },
        )

    async def _persist_observations(
        self,
        observations: Sequence[AlertObservation],
    ) -> tuple[OperationalAlertEvent, ...]:
        if self.repository is None:
            return ()
        events: list[OperationalAlertEvent] = []
        for observation in observations:
            existing_state = await self.repository.load_state(
                fingerprint=observation.fingerprint
            )
            if observation.is_active:
                event = await self._persist_active_observation(
                    observation=observation,
                    existing_state=existing_state,
                )
                if event is not None:
                    events.append(event)
                continue
            cleared_event = await self._persist_clear_observation(
                observation=observation,
                existing_state=existing_state,
            )
            if cleared_event is not None:
                events.append(cleared_event)
        return tuple(events)

    async def _persist_active_observation(
        self,
        *,
        observation: AlertObservation,
        existing_state: OperationalAlertState | None,
    ) -> OperationalAlertEvent | None:
        if existing_state is None or not existing_state.is_active:
            event_state = "OPEN"
            occurrence_count = (
                1 if existing_state is None else existing_state.occurrence_count + 1
            )
            opened_at = observation.event_time
        elif (
            existing_state.severity != observation.severity
            or existing_state.reason_code != observation.reason_code
        ):
            event_state = "UPDATED"
            occurrence_count = existing_state.occurrence_count + 1
            opened_at = existing_state.opened_at
        else:
            await self.repository.save_state(
                OperationalAlertState(
                    fingerprint=observation.fingerprint,
                    service_name=observation.service_name,
                    execution_mode=observation.execution_mode,
                    category=observation.category,
                    symbol=observation.symbol,
                    source_component=observation.source_component,
                    is_active=True,
                    severity=existing_state.severity,
                    reason_code=existing_state.reason_code,
                    opened_at=existing_state.opened_at,
                    last_seen_at=observation.event_time,
                    last_event_id=existing_state.last_event_id,
                    occurrence_count=existing_state.occurrence_count,
                )
            )
            return None

        stored_event = await self.repository.insert_event(
            self._event_from_observation(observation=observation, event_state=event_state)
        )
        await self.repository.save_state(
            OperationalAlertState(
                fingerprint=observation.fingerprint,
                service_name=observation.service_name,
                execution_mode=observation.execution_mode,
                category=observation.category,
                symbol=observation.symbol,
                source_component=observation.source_component,
                is_active=True,
                severity=observation.severity,
                reason_code=observation.reason_code,
                opened_at=opened_at,
                last_seen_at=observation.event_time,
                last_event_id=stored_event.event_id,
                occurrence_count=occurrence_count,
            )
        )
        return stored_event

    async def _persist_clear_observation(
        self,
        *,
        observation: AlertObservation,
        existing_state: OperationalAlertState | None,
    ) -> OperationalAlertEvent | None:
        if existing_state is None or not existing_state.is_active:
            return None
        stored_event = await self.repository.insert_event(
            self._event_from_observation(
                observation=observation,
                event_state="CLEARED",
                severity=existing_state.severity,
            )
        )
        await self.repository.save_state(
            OperationalAlertState(
                fingerprint=existing_state.fingerprint,
                service_name=existing_state.service_name,
                execution_mode=existing_state.execution_mode,
                category=existing_state.category,
                symbol=existing_state.symbol,
                source_component=existing_state.source_component,
                is_active=False,
                severity=existing_state.severity,
                reason_code=observation.reason_code,
                opened_at=existing_state.opened_at,
                last_seen_at=observation.event_time,
                last_event_id=stored_event.event_id,
                occurrence_count=existing_state.occurrence_count + 1,
            )
        )
        return stored_event

    async def _persist_info_event(
        self,
        observation: AlertObservation,
    ) -> OperationalAlertEvent | None:
        if self.repository is None:
            return None
        existing_state = await self.repository.load_state(fingerprint=observation.fingerprint)
        stored_event = await self.repository.insert_event(
            self._event_from_observation(observation=observation, event_state="INFO")
        )
        await self.repository.save_state(
            OperationalAlertState(
                fingerprint=observation.fingerprint,
                service_name=observation.service_name,
                execution_mode=observation.execution_mode,
                category=observation.category,
                symbol=observation.symbol,
                source_component=observation.source_component,
                is_active=False,
                severity=observation.severity,
                reason_code=observation.reason_code,
                opened_at=(
                    observation.event_time
                    if existing_state is None
                    else existing_state.opened_at
                ),
                last_seen_at=observation.event_time,
                last_event_id=stored_event.event_id,
                occurrence_count=(
                    1 if existing_state is None else existing_state.occurrence_count + 1
                ),
            )
        )
        return stored_event

    async def _record_startup_safety_alert(
        self,
        *,
        report: StartupSafetyReport,
    ) -> tuple[OperationalAlertEvent, ...]:
        if self.repository is None:
            return ()
        fingerprint = build_alert_fingerprint(
            service_name=report.service_name,
            execution_mode=report.execution_mode,
            category="STARTUP_SAFETY",
            source_component="startup_safety",
        )
        existing_state = await self.repository.load_state(fingerprint=fingerprint)
        observation = AlertObservation(
            service_name=report.service_name,
            execution_mode=report.execution_mode,
            category="STARTUP_SAFETY",
            severity="INFO" if report.startup_safety_passed else "CRITICAL",
            reason_code=report.primary_reason_code,
            source_component="startup_safety",
            fingerprint=fingerprint,
            summary_text=report.summary_text,
            detail=(
                report.live_startup.detail
                or report.startup_validation.detail
                or report.summary_text
            ),
            event_time=utc_now(),
            is_active=not report.startup_safety_passed,
            payload_json=asdict(report),
        )
        if report.startup_safety_passed:
            cleared_event = await self._persist_clear_observation(
                observation=observation,
                existing_state=existing_state,
            )
            if cleared_event is not None:
                return (cleared_event,)
            info_event = await self._persist_info_event(observation)
            return () if info_event is None else (info_event,)
        active_event = await self._persist_active_observation(
            observation=observation,
            existing_state=existing_state,
        )
        return () if active_event is None else (active_event,)

    def _event_from_observation(
        self,
        *,
        observation: AlertObservation,
        event_state: str,
        severity: str | None = None,
    ) -> OperationalAlertEvent:
        return OperationalAlertEvent(
            service_name=observation.service_name,
            execution_mode=observation.execution_mode,
            category=observation.category,
            severity=severity or observation.severity,
            event_state=event_state,
            reason_code=observation.reason_code,
            source_component=observation.source_component,
            symbol=observation.symbol,
            fingerprint=observation.fingerprint,
            summary_text=observation.summary_text,
            detail=observation.detail,
            event_time=observation.event_time,
            related_order_request_id=observation.related_order_request_id,
            related_decision_trace_id=observation.related_decision_trace_id,
            payload_json=observation.payload_json or {},
        )

    def _daily_drawdown_state(
        self,
        *,
        service_risk_state: ServiceRiskState | None,
        max_drawdown_pct: float,
    ) -> dict[str, Any]:
        if service_risk_state is None:
            return {
                "available": False,
                "breached": False,
            }
        drawdown_pct = current_drawdown_pct(service_risk_state)
        return {
            "available": True,
            "breached": 0.0 < max_drawdown_pct <= drawdown_pct,
            "drawdown_pct": drawdown_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "current_equity": service_risk_state.current_equity,
            "equity_high_watermark": service_risk_state.equity_high_watermark,
            "trading_day": service_risk_state.trading_day.isoformat(),
        }

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(make_json_safe(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def _load_json(self, path: Path) -> dict[str, Any] | None:
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Artifact must deserialize into a mapping: {path}")
        return payload


def build_alert_fingerprint(
    *,
    service_name: str,
    execution_mode: str,
    category: str,
    source_component: str,
    symbol: str | None = None,
) -> str:
    """Build a stable, inspectable alert fingerprint."""
    symbol_part = "*" if symbol is None else symbol
    return "|".join(
        (
            service_name,
            execution_mode,
            category,
            source_component,
            symbol_part,
        )
    )


def _coerce_errors(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    errors = payload.get("errors")
    if not isinstance(errors, list) or not errors:
        return None
    return "; ".join(str(item) for item in errors)
