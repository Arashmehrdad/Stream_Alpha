"""Focused Streamlit app helper tests for the operator console."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from datetime import datetime, timezone

from dashboards.data_sources import (
    ApiHealthSnapshot,
    ContinualLearningDriftCapItemSnapshot,
    ContinualLearningEventItemSnapshot,
    ContinualLearningProfileItemSnapshot,
    ContinualLearningPromotionItemSnapshot,
    ContinualLearningSnapshot,
    ContinualLearningSummarySnapshot,
    DashboardSnapshot,
    DatabaseSnapshot,
    SignalSnapshot,
)
from dashboards.streamlit_app import (
    _build_alert_severity_chart_rows,
    _build_continual_learning_breached_drift_rows,
    _build_continual_learning_event_rows,
    _build_continual_learning_freeze_rows,
    _build_continual_learning_guardrail_rows,
    _build_continual_learning_signal_rows,
    _build_continual_learning_workflow_rows,
    _build_drift_cap_status_chart_rows,
    _build_open_position_exposure_chart_rows,
    _build_regime_performance_chart_rows,
    _build_signal_distribution_chart_rows,
    resolve_display_runtime_profile,
)


def test_runtime_profile_prefers_loaded_api_snapshot() -> None:
    snapshot = DashboardSnapshot(
        api_health=ApiHealthSnapshot(
            available=True,
            checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
            status="ok",
            runtime_profile="shadow",
        ),
        signals=tuple(),
        freshness=tuple(),
        database=DatabaseSnapshot(
            available=True,
            checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
        ),
    )

    assert resolve_display_runtime_profile(snapshot=snapshot) == "SHADOW"


def test_runtime_profile_falls_back_to_env(monkeypatch) -> None:
    monkeypatch.setenv("STREAMALPHA_RUNTIME_PROFILE", "live")
    snapshot = DashboardSnapshot(
        api_health=ApiHealthSnapshot(
            available=False,
            checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
            status="unavailable",
            runtime_profile=None,
            error="api down",
        ),
        signals=tuple(),
        freshness=tuple(),
        database=DatabaseSnapshot(
            available=True,
            checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
        ),
    )

    assert resolve_display_runtime_profile(snapshot=snapshot) == "LIVE"


def test_signal_distribution_chart_rows_are_counted_from_visible_signals() -> None:
    rows = _build_signal_distribution_chart_rows(
        [
            {"symbol": "BTC/USD", "signal": "BUY"},
            {"symbol": "ETH/USD", "signal": "HOLD"},
            {"symbol": "SOL/USD", "signal": "BUY"},
            {"symbol": "DOGE/USD", "signal": "UNAVAILABLE"},
        ]
    )

    assert rows == [
        {"signal": "BUY", "count": 2},
        {"signal": "HOLD", "count": 1},
        {"signal": "UNAVAILABLE", "count": 1},
    ]


def test_chart_helpers_keep_existing_truth_aggregation_explicit() -> None:
    exposure_rows = _build_open_position_exposure_chart_rows(
        [
            {"symbol": "BTC/USD", "entry_notional": 1000.0},
            {"symbol": "BTC/USD", "entry_notional": 500.0},
            {"symbol": "ETH/USD", "entry_notional": 250.0},
        ]
    )
    regime_rows = _build_regime_performance_chart_rows(
        [
            {"regime_label": "RANGE", "total_pnl": 12.5},
            {"regime_label": "TREND_UP", "total_pnl": -4.0},
        ]
    )
    severity_rows = _build_alert_severity_chart_rows(
        [
            {"severity": "CRITICAL"},
            {"severity": "WARNING"},
            {"severity": "CRITICAL"},
        ]
    )

    assert exposure_rows == [
        {"symbol": "BTC/USD", "entry_notional": 1500.0},
        {"symbol": "ETH/USD", "entry_notional": 250.0},
    ]
    assert regime_rows == [
        {"regime_label": "RANGE", "total_pnl": 12.5},
        {"regime_label": "TREND_UP", "total_pnl": -4.0},
    ]
    assert severity_rows == [
        {"severity": "CRITICAL", "count": 2},
        {"severity": "WARNING", "count": 1},
    ]


def test_continual_learning_signal_rows_are_built_from_loaded_snapshot() -> None:
    snapshot = DashboardSnapshot(
        api_health=ApiHealthSnapshot(
            available=True,
            checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
            status="ok",
        ),
        signals=(
            SignalSnapshot(
                symbol="BTC/USD",
                checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
                available=True,
                continual_learning_status="ACTIVE",
                continual_learning_profile_id="cl-profile-1",
                continual_learning_frozen=True,
                continual_learning_candidate_type="CALIBRATION_OVERLAY",
                continual_learning_promotion_stage="LIVE_ELIGIBLE",
                continual_learning_baseline_target_type="MODEL_VERSION",
                continual_learning_baseline_target_id="m20-live",
                continual_learning_drift_cap_status="WATCH",
                continual_learning_reason_codes=(
                    "ACTIVE_PROFILE_PRESENT",
                    "CONTINUAL_LEARNING_FROZEN_BY_HEALTH_GATE",
                ),
            ),
        ),
        freshness=tuple(),
        database=DatabaseSnapshot(
            available=True,
            checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
        ),
    )

    rows = _build_continual_learning_signal_rows(snapshot=snapshot)
    freeze_rows = _build_continual_learning_freeze_rows(snapshot=snapshot)

    assert rows[0]["continual_learning_profile_id"] == "cl-profile-1"
    assert rows[0]["candidate_type"] == "CALIBRATION_OVERLAY"
    assert rows[0]["baseline_target_id"] == "m20-live"
    assert freeze_rows[0]["highlight"] == "FROZEN_BY_HEALTH_GATE"


def test_continual_learning_incident_rows_highlight_rollback_and_breached_drift() -> None:
    snapshot = DashboardSnapshot(
        api_health=ApiHealthSnapshot(
            available=True,
            checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
            status="ok",
        ),
        signals=tuple(),
        freshness=tuple(),
        database=DatabaseSnapshot(
            available=True,
            checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
        ),
        continual_learning=ContinualLearningSnapshot(
            summary=ContinualLearningSummarySnapshot(
                available=True,
                checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
                continual_learning_status="ACTIVE",
                aggregated_scope=True,
            ),
            profiles=(
                ContinualLearningProfileItemSnapshot(
                    profile_id="cl-profile-1",
                    status="ACTIVE",
                    candidate_type="CALIBRATION_OVERLAY",
                    execution_mode_scope="paper",
                    symbol_scope="BTC/USD",
                    regime_scope="TREND_UP",
                    baseline_target_type="MODEL_VERSION",
                    baseline_target_id="m20-live",
                    rollback_target_profile_id="cl-profile-prev-1",
                ),
            ),
            drift_caps=(
                ContinualLearningDriftCapItemSnapshot(
                    cap_id="cl-cap-1",
                    execution_mode_scope="paper",
                    symbol_scope="BTC/USD",
                    regime_scope="TREND_UP",
                    candidate_type="CALIBRATION_OVERLAY",
                    status="BREACHED",
                    observed_drift_score=0.25,
                    reason_code="DRIFT_BREACHED",
                ),
            ),
            promotions=(
                ContinualLearningPromotionItemSnapshot(
                    decision_id="cl-decision-1",
                    target_type="PROFILE",
                    target_id="cl-profile-1",
                    decision="ROLLBACK",
                    summary_text="rollback",
                    decided_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
                ),
            ),
            events=(
                ContinualLearningEventItemSnapshot(
                    event_id="cl-event-1",
                    event_type="ROLLBACK_APPLIED",
                    profile_id="cl-profile-1",
                    experiment_id="cl-exp-1",
                    decision_id="cl-decision-1",
                    reason_code="CONTINUAL_LEARNING_ROLLBACK_TARGET_ACTIVATED",
                    created_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
                ),
            ),
        ),
    )

    event_rows = _build_continual_learning_event_rows(snapshot=snapshot)
    breached_rows = _build_continual_learning_breached_drift_rows(snapshot=snapshot)

    assert event_rows[0]["highlight"] == "ROLLBACK_APPLIED"
    assert breached_rows[0]["highlight"] == "DRIFT_BREACHED"
    assert _build_drift_cap_status_chart_rows(snapshot=snapshot) == [
        {"status": "BREACHED", "count": 1}
    ]


def test_continual_learning_workflow_and_guardrail_rows_are_built_from_snapshot() -> None:
    snapshot = DashboardSnapshot(
        api_health=ApiHealthSnapshot(
            available=True,
            checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
            status="ok",
            runtime_profile="paper",
        ),
        signals=tuple(),
        freshness=tuple(),
        database=DatabaseSnapshot(
            available=True,
            checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
        ),
        continual_learning=ContinualLearningSnapshot(
            summary=ContinualLearningSummarySnapshot(
                available=True,
                checked_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
                continual_learning_status="ACTIVE",
                active_profile_count=1,
                active_profile_id="cl-profile-1",
                latest_drift_cap_status="BREACHED",
                latest_promotion_decision="HOLD",
                latest_event_type="PROMOTION_BLOCKED",
                aggregated_scope=True,
                reason_codes=("AGGREGATED_SCOPE_SUMMARY",),
            ),
            profiles=(
                ContinualLearningProfileItemSnapshot(
                    profile_id="cl-profile-1",
                    status="ACTIVE",
                    candidate_type="CALIBRATION_OVERLAY",
                    execution_mode_scope="paper",
                    symbol_scope="BTC/USD",
                    regime_scope="TREND_UP",
                    baseline_target_type="MODEL_VERSION",
                    baseline_target_id="m20-live",
                    promotion_stage="LIVE_ELIGIBLE",
                    live_eligible=True,
                    rollback_target_profile_id="cl-profile-prev-1",
                ),
                ContinualLearningProfileItemSnapshot(
                    profile_id="cl-profile-shadow-1",
                    status="APPROVED",
                    candidate_type="INCREMENTAL_SHADOW_CHALLENGER",
                    execution_mode_scope="paper",
                    symbol_scope="BTC/USD",
                    regime_scope="TREND_DOWN",
                    baseline_target_type="MODEL_VERSION",
                    baseline_target_id="m20-live",
                    promotion_stage="SHADOW_ONLY",
                    live_eligible=False,
                ),
            ),
            drift_caps=(
                ContinualLearningDriftCapItemSnapshot(
                    cap_id="cl-cap-1",
                    execution_mode_scope="paper",
                    symbol_scope="BTC/USD",
                    regime_scope="TREND_UP",
                    candidate_type="CALIBRATION_OVERLAY",
                    status="BREACHED",
                    observed_drift_score=0.23,
                    reason_code="DRIFT_BREACHED",
                ),
            ),
        ),
    )

    workflow_rows = _build_continual_learning_workflow_rows(snapshot=snapshot)
    guardrail_rows = _build_continual_learning_guardrail_rows(snapshot=snapshot)

    assert workflow_rows[0]["active_profile_id"] == "cl-profile-1"
    assert workflow_rows[0]["latest_event_type"] == "PROMOTION_BLOCKED"
    assert workflow_rows[0]["operator_note"] == "MANUAL_AND_GUARDED"
    assert guardrail_rows[0]["operator_note"] == "BLOCKED_BY_BREACHED_DRIFT"
    assert guardrail_rows[1]["operator_note"] == "SHADOW_ONLY_ONLY"
