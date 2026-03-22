"""Focused tests for additive M14 decision-trace schema fields."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

from app.adaptation.schemas import AdaptationContextPayload
from app.continual_learning.schemas import ContinualLearningContextPayload
from app.explainability.schemas import (
    DecisionTracePayload,
    DecisionTracePrediction,
    DecisionTraceSignal,
    PredictionExplanation,
)


def _prediction_explanation() -> PredictionExplanation:
    return PredictionExplanation(
        method="oat_ref_ablation",
        available=True,
        reason_code="EXPLAINABILITY_OK",
        summary_text="ok",
    )


def test_decision_trace_payload_accepts_continual_learning_context_additively() -> None:
    payload = DecisionTracePayload(
        schema_version="m14_decision_trace_v1",
        service_name="paper-trader",
        execution_mode="paper",
        symbol="BTC/USD",
        signal_row_id="BTC/USD|2026-03-21T12:00:00Z",
        signal_interval_begin="2026-03-21T12:00:00Z",
        signal_as_of_time="2026-03-21T12:05:00Z",
        model_name="dynamic_ensemble",
        model_version="ensemble_profile:ens-profile-1",
        prediction=DecisionTracePrediction(
            model_name="dynamic_ensemble",
            model_version="ensemble_profile:ens-profile-1",
            prob_up=0.67,
            prob_down=0.33,
            confidence=0.67,
            predicted_class="UP",
            prediction_explanation=_prediction_explanation(),
        ),
        signal=DecisionTraceSignal(
            signal="BUY",
            reason="prob_up >= threshold",
        ),
        adaptation=AdaptationContextPayload(
            adaptation_profile_id="adapt-1",
            adaptation_reason_codes=["ADAPTATION_PROFILE_ACTIVE"],
        ),
        continual_learning=ContinualLearningContextPayload(
            enabled=True,
            active_profile_id="cl-profile-1",
            candidate_type="CALIBRATION_OVERLAY",
            promotion_stage="LIVE_ELIGIBLE",
            live_eligible=True,
            baseline_target_type="MODEL_VERSION",
            baseline_target_id="m20-live",
            source_experiment_id="cl-exp-1",
            drift_cap_status="WATCH",
            latest_promotion_decision="HOLD",
            frozen_by_health_gate=True,
            reason_codes=[
                "ACTIVE_PROFILE_PRESENT",
                "CONTINUAL_LEARNING_FROZEN_BY_HEALTH_GATE",
            ],
        ),
    )

    assert payload.continual_learning is not None
    assert payload.continual_learning.frozen_by_health_gate is True
    assert payload.continual_learning.baseline_target_id == "m20-live"
