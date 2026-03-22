"""Focused unit tests for M21 read-only wiring inside inference service."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.continual_learning.schemas import ContinualLearningContextPayload
from app.inference.service import InferenceService, load_model_artifact
from tests.test_inference_api import (  # Reuse existing deterministic stubs/helpers.
    FakeAlertRepository,
    FakeDatabase,
    FakeReliabilityStore,
    NullAdaptationService,
    NullEnsembleService,
    _build_regime_runtime,
    _build_settings,
    _feature_row,
    _write_artifact,
)


class RecordingContinualLearningService:
    """Read-only continual-learning stub that records requested runtime contexts."""

    def __init__(self) -> None:
        self.config = type("Cfg", (), {"enabled": True})()
        self.calls: list[dict] = []

    async def startup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    async def resolve_runtime_context(self, **kwargs) -> ContinualLearningContextPayload:
        self.calls.append(dict(kwargs))
        frozen = (
            kwargs.get("health_overall_status") not in {None, "HEALTHY"}
            or kwargs.get("freshness_status") not in {None, "FRESH"}
        )
        reason_codes = ["ACTIVE_PROFILE_PRESENT"]
        if frozen:
            reason_codes.append("CONTINUAL_LEARNING_FROZEN_BY_HEALTH_GATE")
        return ContinualLearningContextPayload(
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
            frozen_by_health_gate=frozen,
            reason_codes=reason_codes,
        )

    async def summary(self, **_kwargs):
        raise AssertionError("Not used in this focused test file")

    async def experiments(self, **_kwargs):
        raise AssertionError("Not used in this focused test file")

    async def profiles(self, **_kwargs):
        raise AssertionError("Not used in this focused test file")

    async def drift_caps(self, **_kwargs):
        raise AssertionError("Not used in this focused test file")

    async def promotions(self, **_kwargs):
        raise AssertionError("Not used in this focused test file")

    async def events(self, **_kwargs):
        raise AssertionError("Not used in this focused test file")


def _build_service(
    tmp_path: Path,
    *,
    database: FakeDatabase,
    continual_learning_service: RecordingContinualLearningService,
    prob_up: float = 0.7,
) -> InferenceService:
    model_path = _write_artifact(tmp_path, prob_up=prob_up)
    artifact = load_model_artifact(str(model_path))
    return InferenceService(
        _build_settings(str(model_path)),
        database=database,
        model_artifact=artifact,
        regime_runtime=_build_regime_runtime(tmp_path),
        reliability_store=FakeReliabilityStore(),
        alert_repository=FakeAlertRepository(),
        adaptation_service=NullAdaptationService(),
        ensemble_service=NullEnsembleService(),
        continual_learning_service=continual_learning_service,
    )


def test_predict_from_row_adds_continual_learning_fields_without_score_change(
    tmp_path: Path,
) -> None:
    row = _feature_row()
    cl_service = RecordingContinualLearningService()
    service = _build_service(
        tmp_path,
        database=FakeDatabase(row=row),
        continual_learning_service=cl_service,
        prob_up=0.7,
    )

    freshness = asyncio.run(service.freshness_evaluation(symbol="BTC/USD"))
    prediction = asyncio.run(service.predict_from_row(row, freshness=freshness))

    assert prediction.prob_up == pytest.approx(0.7)
    assert prediction.prob_down == pytest.approx(0.3)
    assert prediction.continual_learning_profile_id == "cl-profile-1"
    assert prediction.continual_learning_status == "ACTIVE"
    assert prediction.continual_learning_frozen is False
    assert prediction.continual_learning is not None
    assert prediction.continual_learning.baseline_target_id == "m20-live"


def test_signal_for_request_sets_continual_learning_frozen_when_stale(
    tmp_path: Path,
) -> None:
    stale_base_time = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(
        minutes=20
    )
    row = _feature_row(base_time=stale_base_time)
    cl_service = RecordingContinualLearningService()
    service = _build_service(
        tmp_path,
        database=FakeDatabase(row=row),
        continual_learning_service=cl_service,
        prob_up=0.7,
    )

    signal = asyncio.run(service.signal_for_request(symbol="BTC/USD"))

    assert signal.signal == "HOLD"
    assert signal.decision_source == "reliability"
    assert signal.continual_learning_profile_id == "cl-profile-1"
    assert signal.continual_learning_status == "ACTIVE"
    assert signal.continual_learning_frozen is True
    assert signal.continual_learning is not None
    assert "CONTINUAL_LEARNING_FROZEN_BY_HEALTH_GATE" in signal.continual_learning.reason_codes
