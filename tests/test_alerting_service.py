"""Focused M17 alerting backend tests."""

# pylint: disable=missing-class-docstring,missing-function-docstring

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from app.alerting.config import (
    AlertingArtifactConfig,
    AlertingConfig,
    OrderFailureSpikeConfig,
    SignalAlertConfig,
)
from app.alerting.service import (
    OperationalAlertService,
    ORDER_FAILURE_SPIKE_CRITICAL,
    ORDER_FAILURE_SPIKE_WARNING,
    SIGNAL_FLOOD_WARNING,
    SIGNAL_SILENCE_DETECTED,
    STARTUP_SAFETY_PASSED,
)
from app.common.time import to_rfc3339
from app.reliability.service import FEATURE_LAG_BREACH, FEATURE_LAG_OK, FEED_FRESH, FEED_STALE
from app.trading.schemas import (
    CanonicalFeatureLag,
    CanonicalSystemReliability,
    CanonicalServiceHealth,
    LiveSafetyState,
    OrderLifecycleEvent,
    ServiceRiskState,
)


class FakeAlertRepository:
    def __init__(self) -> None:
        self.events = []
        self.states = {}

    async def load_state(self, *, fingerprint: str):
        return self.states.get(fingerprint)

    async def save_state(self, state) -> None:
        self.states[state.fingerprint] = state

    async def insert_event(self, event):
        stored = replace(
            event,
            event_id=len(self.events) + 1,
            created_at=event.event_time,
        )
        self.events.append(stored)
        return stored

    async def load_active_states(self, *, service_name: str, execution_mode: str):
        return [
            state
            for state in self.states.values()
            if state.service_name == service_name
            and state.execution_mode == execution_mode
            and state.is_active
        ]

    async def load_events_for_day(
        self,
        *,
        service_name: str,
        execution_mode: str,
        summary_date: date,
    ):
        return [
            event
            for event in self.events
            if event.service_name == service_name
            and event.execution_mode == execution_mode
            and event.event_time.date() == summary_date
        ]


def _alerting_config(tmp_path: Path) -> AlertingConfig:
    return AlertingConfig(
        schema_version="m17_alerting_v1",
        order_failure_spike=OrderFailureSpikeConfig(
            window_minutes=60,
            warning_count=2,
            critical_count=4,
        ),
        signals=SignalAlertConfig(
            silence_window_intervals=3,
            flood_window_intervals=6,
            flood_warning_count=2,
            flood_critical_count=4,
        ),
        artifacts=AlertingArtifactConfig(
            daily_summary_dir=str(tmp_path / "daily"),
            startup_safety_path=str(tmp_path / "startup_safety.json"),
        ),
    )


def _now() -> datetime:
    return datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc)


def _system_reliability(
    *,
    feed_reason_code: str = FEED_FRESH,
    lag_breach: bool = False,
) -> CanonicalSystemReliability:
    observed_at = _now()
    return CanonicalSystemReliability(
        service_name="stream-alpha",
        checked_at=observed_at,
        health_overall_status="DEGRADED" if feed_reason_code == FEED_STALE else "HEALTHY",
        reason_codes=(feed_reason_code,),
        lag_breach_active=lag_breach,
        services=(
            CanonicalServiceHealth(
                service_name="stream-alpha",
                component_name="producer",
                checked_at=observed_at,
                heartbeat_at=observed_at,
                heartbeat_age_seconds=0.0,
                heartbeat_freshness_status="FRESH",
                health_overall_status="DEGRADED" if feed_reason_code == FEED_STALE else "HEALTHY",
                reason_code=feed_reason_code,
                detail="producer snapshot",
                feed_freshness_status="STALE" if feed_reason_code == FEED_STALE else "FRESH",
                feed_reason_code=feed_reason_code,
                feed_age_seconds=90.0 if feed_reason_code == FEED_STALE else 0.0,
            ),
        ),
        lag_by_symbol=(
            CanonicalFeatureLag(
                service_name="feature-builder",
                component_name="features",
                symbol="BTC/USD",
                evaluated_at=observed_at,
                latest_raw_event_received_at=observed_at,
                latest_feature_interval_begin=observed_at - timedelta(minutes=5),
                latest_feature_as_of_time=observed_at,
                time_lag_seconds=0.0 if not lag_breach else 600.0,
                processing_lag_seconds=0.0 if not lag_breach else 300.0,
                time_lag_reason_code="FEATURE_TIME_LAG_OK",
                processing_lag_reason_code="FEATURE_PROCESSING_LAG_OK",
                lag_breach=lag_breach,
                health_overall_status="DEGRADED" if lag_breach else "HEALTHY",
                reason_code=FEATURE_LAG_BREACH if lag_breach else FEATURE_LAG_OK,
                detail="lag snapshot",
            ),
        ),
    )


def _decision_trace(decision_trace_id: int, *, signal: str = "BUY", minutes_ago: int = 0):
    signal_time = _now() - timedelta(minutes=minutes_ago)
    return SimpleNamespace(
        decision_trace_id=decision_trace_id,
        signal=signal,
        signal_as_of_time=signal_time,
    )


def _order_event(
    lifecycle_state: str,
    *,
    minutes_ago: int = 0,
    order_request_id: int = 1,
    decision_trace_id: int | None = None,
) -> OrderLifecycleEvent:
    event_time = _now() - timedelta(minutes=minutes_ago)
    return OrderLifecycleEvent(
        order_request_id=order_request_id,
        service_name="paper-trader",
        execution_mode="paper",
        symbol="BTC/USD",
        action="BUY",
        lifecycle_state=lifecycle_state,
        event_time=event_time,
        decision_trace_id=decision_trace_id,
    )


def _service_risk_state(*, current_equity: float, high_watermark: float) -> ServiceRiskState:
    return ServiceRiskState(
        service_name="paper-trader",
        execution_mode="paper",
        trading_day=_now().date(),
        realized_pnl_today=0.0,
        equity_high_watermark=high_watermark,
        current_equity=current_equity,
        loss_streak_count=0,
    )


def _live_safety_state() -> LiveSafetyState:
    return LiveSafetyState(
        service_name="paper-trader",
        execution_mode="live",
        broker_name="alpaca",
        live_enabled=True,
        startup_checks_passed=True,
        startup_checks_passed_at=_now(),
        account_validated=True,
        account_id="PA12345",
        environment_name="paper",
        manual_disable_active=False,
        consecutive_live_failures=0,
        failure_hard_stop_active=False,
        system_health_status="HEALTHY",
        system_health_reason_code="SYSTEM_HEALTHY",
        system_health_checked_at=_now(),
        health_gate_status="CLEAR",
        health_gate_reason_code="LIVE_HEALTH_GATE_CLEAR",
        reconciliation_status="CLEAR",
        reconciliation_reason_code="LIVE_RECONCILIATION_CLEAR",
        reconciliation_checked_at=_now(),
        can_submit_live_now=True,
        primary_block_reason_code=None,
        block_detail=None,
    )


def test_stale_feed_open_and_clear(tmp_path: Path) -> None:
    repository = FakeAlertRepository()
    service = OperationalAlertService(
        config=_alerting_config(tmp_path),
        repository=repository,
    )

    opened = asyncio.run(
        service.evaluate_cycle(
            service_name="paper-trader",
            execution_mode="paper",
            interval_minutes=5,
            evaluated_at=_now(),
            system_reliability=_system_reliability(feed_reason_code=FEED_STALE),
            service_risk_state=_service_risk_state(current_equity=100.0, high_watermark=100.0),
            order_events=[],
            decision_traces=[_decision_trace(1)],
            max_drawdown_pct=0.15,
        )
    )
    cleared = asyncio.run(
        service.evaluate_cycle(
            service_name="paper-trader",
            execution_mode="paper",
            interval_minutes=5,
            evaluated_at=_now() + timedelta(minutes=5),
            system_reliability=_system_reliability(feed_reason_code=FEED_FRESH),
            service_risk_state=_service_risk_state(current_equity=100.0, high_watermark=100.0),
            order_events=[],
            decision_traces=[_decision_trace(1)],
            max_drawdown_pct=0.15,
        )
    )

    assert len(opened) == 1
    assert opened[0].category == "FEED_STALE"
    assert opened[0].event_state == "OPEN"
    assert len(cleared) == 1
    assert cleared[0].category == "FEED_STALE"
    assert cleared[0].event_state == "CLEARED"


def test_consumer_lag_open_and_clear(tmp_path: Path) -> None:
    repository = FakeAlertRepository()
    service = OperationalAlertService(
        config=_alerting_config(tmp_path),
        repository=repository,
    )

    opened = asyncio.run(
        service.evaluate_cycle(
            service_name="paper-trader",
            execution_mode="paper",
            interval_minutes=5,
            evaluated_at=_now(),
            system_reliability=_system_reliability(lag_breach=True),
            service_risk_state=_service_risk_state(current_equity=100.0, high_watermark=100.0),
            order_events=[],
            decision_traces=[_decision_trace(1)],
            max_drawdown_pct=0.15,
        )
    )
    cleared = asyncio.run(
        service.evaluate_cycle(
            service_name="paper-trader",
            execution_mode="paper",
            interval_minutes=5,
            evaluated_at=_now() + timedelta(minutes=5),
            system_reliability=_system_reliability(lag_breach=False),
            service_risk_state=_service_risk_state(current_equity=100.0, high_watermark=100.0),
            order_events=[],
            decision_traces=[_decision_trace(1)],
            max_drawdown_pct=0.15,
        )
    )

    assert len(opened) == 1
    assert opened[0].category == "CONSUMER_LAG"
    assert opened[0].event_state == "OPEN"
    assert opened[0].symbol == "BTC/USD"
    assert len(cleared) == 1
    assert cleared[0].category == "CONSUMER_LAG"
    assert cleared[0].event_state == "CLEARED"


def test_order_failure_spike_thresholding_dedupes_and_updates(tmp_path: Path) -> None:
    repository = FakeAlertRepository()
    service = OperationalAlertService(
        config=_alerting_config(tmp_path),
        repository=repository,
    )
    warning_events = [
        _order_event("REJECTED", order_request_id=1),
        _order_event("FAILED", order_request_id=2),
    ]
    critical_events = warning_events + [
        _order_event("FAILED", order_request_id=3),
        _order_event("REJECTED", order_request_id=4),
    ]

    opened = asyncio.run(
        service.evaluate_cycle(
            service_name="paper-trader",
            execution_mode="paper",
            interval_minutes=5,
            evaluated_at=_now(),
            system_reliability=_system_reliability(),
            service_risk_state=_service_risk_state(current_equity=100.0, high_watermark=100.0),
            order_events=warning_events,
            decision_traces=[_decision_trace(1)],
            max_drawdown_pct=0.15,
        )
    )
    duplicate = asyncio.run(
        service.evaluate_cycle(
            service_name="paper-trader",
            execution_mode="paper",
            interval_minutes=5,
            evaluated_at=_now() + timedelta(minutes=1),
            system_reliability=_system_reliability(),
            service_risk_state=_service_risk_state(current_equity=100.0, high_watermark=100.0),
            order_events=warning_events,
            decision_traces=[_decision_trace(1)],
            max_drawdown_pct=0.15,
        )
    )
    updated = asyncio.run(
        service.evaluate_cycle(
            service_name="paper-trader",
            execution_mode="paper",
            interval_minutes=5,
            evaluated_at=_now() + timedelta(minutes=2),
            system_reliability=_system_reliability(),
            service_risk_state=_service_risk_state(current_equity=100.0, high_watermark=100.0),
            order_events=critical_events,
            decision_traces=[_decision_trace(1)],
            max_drawdown_pct=0.15,
        )
    )

    assert len(opened) == 1
    assert opened[0].reason_code == ORDER_FAILURE_SPIKE_WARNING
    assert opened[0].event_state == "OPEN"
    assert duplicate == ()
    assert len(updated) == 1
    assert updated[0].reason_code == ORDER_FAILURE_SPIKE_CRITICAL
    assert updated[0].event_state == "UPDATED"


def test_drawdown_breach_alert_opens(tmp_path: Path) -> None:
    repository = FakeAlertRepository()
    service = OperationalAlertService(
        config=_alerting_config(tmp_path),
        repository=repository,
    )

    events = asyncio.run(
        service.evaluate_cycle(
            service_name="paper-trader",
            execution_mode="paper",
            interval_minutes=5,
            evaluated_at=_now(),
            system_reliability=_system_reliability(),
            service_risk_state=_service_risk_state(current_equity=80.0, high_watermark=100.0),
            order_events=[],
            decision_traces=[_decision_trace(1)],
            max_drawdown_pct=0.15,
        )
    )

    assert len(events) == 1
    assert events[0].category == "DRAWDOWN_BREACH"
    assert events[0].event_state == "OPEN"


def test_signal_silence_alert_opens(tmp_path: Path) -> None:
    repository = FakeAlertRepository()
    service = OperationalAlertService(
        config=_alerting_config(tmp_path),
        repository=repository,
    )

    events = asyncio.run(
        service.evaluate_cycle(
            service_name="paper-trader",
            execution_mode="paper",
            interval_minutes=5,
            evaluated_at=_now(),
            system_reliability=_system_reliability(),
            service_risk_state=_service_risk_state(current_equity=100.0, high_watermark=100.0),
            order_events=[],
            decision_traces=[],
            max_drawdown_pct=0.15,
        )
    )

    assert len(events) == 1
    assert events[0].category == "SIGNAL_SILENCE"
    assert events[0].reason_code == SIGNAL_SILENCE_DETECTED


def test_hold_only_decision_traces_still_trigger_signal_silence(tmp_path: Path) -> None:
    repository = FakeAlertRepository()
    service = OperationalAlertService(
        config=_alerting_config(tmp_path),
        repository=repository,
    )

    events = asyncio.run(
        service.evaluate_cycle(
            service_name="paper-trader",
            execution_mode="paper",
            interval_minutes=5,
            evaluated_at=_now(),
            system_reliability=_system_reliability(),
            service_risk_state=_service_risk_state(current_equity=100.0, high_watermark=100.0),
            order_events=[],
            decision_traces=[
                _decision_trace(1, signal="HOLD"),
                _decision_trace(2, signal="HOLD", minutes_ago=5),
            ],
            max_drawdown_pct=0.15,
        )
    )

    assert len(events) == 1
    assert events[0].category == "SIGNAL_SILENCE"
    assert events[0].reason_code == SIGNAL_SILENCE_DETECTED
    assert events[0].payload_json["decision_trace_count"] == 2
    assert events[0].payload_json["actionable_trace_count"] == 0
    assert events[0].payload_json["last_actionable_signal_as_of_time"] is None


def test_signal_flood_alert_opens(tmp_path: Path) -> None:
    repository = FakeAlertRepository()
    service = OperationalAlertService(
        config=_alerting_config(tmp_path),
        repository=repository,
    )
    traces = [
        _decision_trace(1, signal="BUY"),
        _decision_trace(2, signal="SELL", minutes_ago=5),
    ]

    events = asyncio.run(
        service.evaluate_cycle(
            service_name="paper-trader",
            execution_mode="paper",
            interval_minutes=5,
            evaluated_at=_now(),
            system_reliability=_system_reliability(),
            service_risk_state=_service_risk_state(current_equity=100.0, high_watermark=100.0),
            order_events=[],
            decision_traces=traces,
            max_drawdown_pct=0.15,
        )
    )

    assert len(events) == 1
    assert events[0].category == "SIGNAL_FLOOD"
    assert events[0].reason_code == SIGNAL_FLOOD_WARNING


def test_feed_stale_suppresses_signal_cadence_open_and_clears_silence(
    tmp_path: Path,
) -> None:
    repository = FakeAlertRepository()
    service = OperationalAlertService(
        config=_alerting_config(tmp_path),
        repository=repository,
    )

    opened = asyncio.run(
        service.evaluate_cycle(
            service_name="paper-trader",
            execution_mode="paper",
            interval_minutes=5,
            evaluated_at=_now(),
            system_reliability=_system_reliability(),
            service_risk_state=_service_risk_state(current_equity=100.0, high_watermark=100.0),
            order_events=[],
            decision_traces=[],
            max_drawdown_pct=0.15,
        )
    )
    suppressed = asyncio.run(
        service.evaluate_cycle(
            service_name="paper-trader",
            execution_mode="paper",
            interval_minutes=5,
            evaluated_at=_now() + timedelta(minutes=5),
            system_reliability=_system_reliability(feed_reason_code=FEED_STALE),
            service_risk_state=_service_risk_state(current_equity=100.0, high_watermark=100.0),
            order_events=[],
            decision_traces=[],
            max_drawdown_pct=0.15,
        )
    )

    assert len(opened) == 1
    assert opened[0].category == "SIGNAL_SILENCE"
    assert opened[0].event_state == "OPEN"
    assert [event.category for event in suppressed] == ["FEED_STALE", "SIGNAL_SILENCE"]
    assert suppressed[0].event_state == "OPEN"
    assert suppressed[1].event_state == "CLEARED"
    assert suppressed[1].payload_json["cadence_evaluation_allowed"] is False
    assert suppressed[1].payload_json["suppression_reason_code"] == FEED_STALE


def test_lag_breach_suppresses_signal_cadence_open_and_clears_flood(
    tmp_path: Path,
) -> None:
    repository = FakeAlertRepository()
    service = OperationalAlertService(
        config=_alerting_config(tmp_path),
        repository=repository,
    )
    traces = [
        _decision_trace(1, signal="BUY"),
        _decision_trace(2, signal="SELL", minutes_ago=5),
    ]

    opened = asyncio.run(
        service.evaluate_cycle(
            service_name="paper-trader",
            execution_mode="paper",
            interval_minutes=5,
            evaluated_at=_now(),
            system_reliability=_system_reliability(),
            service_risk_state=_service_risk_state(current_equity=100.0, high_watermark=100.0),
            order_events=[],
            decision_traces=traces,
            max_drawdown_pct=0.15,
        )
    )
    suppressed = asyncio.run(
        service.evaluate_cycle(
            service_name="paper-trader",
            execution_mode="paper",
            interval_minutes=5,
            evaluated_at=_now() + timedelta(minutes=5),
            system_reliability=_system_reliability(lag_breach=True),
            service_risk_state=_service_risk_state(current_equity=100.0, high_watermark=100.0),
            order_events=[],
            decision_traces=traces,
            max_drawdown_pct=0.15,
        )
    )

    assert len(opened) == 1
    assert opened[0].category == "SIGNAL_FLOOD"
    assert opened[0].event_state == "OPEN"
    assert [event.category for event in suppressed] == ["CONSUMER_LAG", "SIGNAL_FLOOD"]
    assert suppressed[0].event_state == "OPEN"
    assert suppressed[1].event_state == "CLEARED"
    assert suppressed[1].payload_json["cadence_evaluation_allowed"] is False
    assert suppressed[1].payload_json["suppression_reason_code"] == FEATURE_LAG_BREACH


def test_live_mode_activation_records_info_event(tmp_path: Path) -> None:
    repository = FakeAlertRepository()
    service = OperationalAlertService(
        config=_alerting_config(tmp_path),
        repository=repository,
    )

    events = asyncio.run(
        service.record_live_mode_activation(
            service_name="paper-trader",
            execution_mode="live",
            runtime_profile="live",
            live_safety_state=_live_safety_state(),
            event_time=_now(),
        )
    )

    assert len(events) == 1
    assert events[0].category == "LIVE_MODE_ACTIVATION"
    assert events[0].event_state == "INFO"


def test_startup_safety_report_composition_and_daily_summary(tmp_path: Path) -> None:
    startup_report_path = tmp_path / "startup_report.json"
    startup_report_path.write_text(
        '{"startup_validation_passed": true, "errors": []}',
        encoding="utf-8",
    )
    checklist_path = tmp_path / "startup_checklist.json"
    checklist_path.write_text('{"passed": true}', encoding="utf-8")
    repository = FakeAlertRepository()
    service = OperationalAlertService(
        config=_alerting_config(tmp_path),
        repository=repository,
    )

    report, startup_events = asyncio.run(
        service.write_startup_safety_artifact(
            service_name="paper-trader",
            execution_mode="live",
            runtime_profile="live",
            startup_validation_passed=True,
            startup_report_path=str(startup_report_path),
            live_safety_state=_live_safety_state(),
            live_startup_checklist_path=str(checklist_path),
        )
    )
    summary = asyncio.run(
        service.write_daily_summary(
            service_name="paper-trader",
            execution_mode="live",
            runtime_profile="live",
            evaluated_at=_now(),
            service_risk_state=_service_risk_state(current_equity=100.0, high_watermark=100.0),
            max_drawdown_pct=0.15,
            order_events=[],
            decision_traces=[_decision_trace(1)],
            startup_safety_report=report,
        )
    )

    assert report.startup_safety_passed is True
    assert report.primary_reason_code == STARTUP_SAFETY_PASSED
    assert startup_events[0].event_state == "INFO"
    assert summary.startup_safety_status["startup_safety_passed"] is True
    assert Path(_alerting_config(tmp_path).artifacts.startup_safety_path).is_file()
    assert Path(
        _alerting_config(tmp_path).artifacts.daily_summary_dir,
        f"{_now().date().isoformat()}.json",
    ).is_file()
    assert summary.actionable_signal_counts["total_actionable"] == 1
    assert summary.counts_by_category["STARTUP_SAFETY"] == 1
    assert summary.highest_severity == "INFO"
    assert startup_events[0].payload_json["startup_validation"]["report_path"] == str(
        startup_report_path
    )
    assert to_rfc3339(_now()) == "2026-03-22T12:00:00Z"
