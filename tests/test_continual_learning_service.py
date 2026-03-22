"""Focused tests for the Stream Alpha M21 continual-learning layer."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from app.continual_learning.config import ArtifactConfig, ContinualLearningConfig
from app.continual_learning.schemas import (
    CalibrationOverlayProfile,
    ContinualLearningDriftCapRecord,
    ContinualLearningProfileRecord,
)
from app.continual_learning.service import ContinualLearningService


class _FakeRepo:
    def __init__(self, *, active_profile, fallback_profile, drift_cap) -> None:
        self._active_profile = active_profile
        self._drift_cap = drift_cap
        self._profiles = {
            active_profile.profile_id: active_profile,
            fallback_profile.profile_id: fallback_profile,
        }
        self._promotion_decisions = []
        self._events = []

    async def connect(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def load_active_continual_learning_profile(self, **_kwargs):
        return self._active_profile

    async def load_continual_learning_profiles(self, *, limit: int):
        del limit
        return list(self._profiles.values())

    async def load_latest_continual_learning_drift_cap(self, **_kwargs):
        return self._drift_cap

    async def load_continual_learning_promotion_decisions(self, *, limit: int):
        del limit
        return list(self._promotion_decisions)

    async def load_continual_learning_events(self, *, limit: int):
        del limit
        return list(self._events)

    async def load_continual_learning_experiments(self, *, limit: int):
        del limit
        return []

    async def load_continual_learning_drift_caps(self, **_kwargs):
        return [self._drift_cap]

    async def load_continual_learning_profile(self, *, profile_id: str):
        return self._profiles.get(profile_id)

    async def rollback_continual_learning_profile(
        self,
        *,
        active_profile_id: str,
        rollback_target_profile_id: str,
        changed_at,
    ) -> None:
        active_profile = self._profiles[active_profile_id]
        rollback_target = self._profiles[rollback_target_profile_id]
        self._profiles[active_profile_id] = active_profile.model_copy(
            update={"status": "ROLLED_BACK", "superseded_at": changed_at}
        )
        self._profiles[rollback_target_profile_id] = rollback_target.model_copy(
            update={"status": "ACTIVE", "activated_at": changed_at, "superseded_at": None}
        )
        self._active_profile = self._profiles[rollback_target_profile_id]

    async def save_continual_learning_promotion_decision(self, decision) -> None:
        self._promotion_decisions.insert(0, decision)

    async def save_continual_learning_event(self, event) -> None:
        self._events.insert(0, event)


def _build_config(root_dir: str) -> ContinualLearningConfig:
    return ContinualLearningConfig(
        enabled=True,
        candidate_types=("CALIBRATION_OVERLAY", "INCREMENTAL_SHADOW_CHALLENGER"),
        live_eligible_candidate_types=("CALIBRATION_OVERLAY",),
        shadow_only_candidate_types=("INCREMENTAL_SHADOW_CHALLENGER",),
        artifacts=ArtifactConfig(
            root_dir=root_dir,
            summary_path=f"{root_dir}/summary.json",
            drift_caps_summary_path=f"{root_dir}/drift_caps_summary.json",
            current_profile_path=f"{root_dir}/current_profile.json",
            promotions_history_path=f"{root_dir}/promotions_history.jsonl",
            events_history_path=f"{root_dir}/events_history.jsonl",
            reports_dir=f"{root_dir}/reports",
            experiments_dir=f"{root_dir}/experiments",
        ),
    )


def _build_profiles() -> tuple[ContinualLearningProfileRecord, ContinualLearningProfileRecord]:
    fallback = ContinualLearningProfileRecord(
        profile_id="profile-fallback",
        candidate_type="CALIBRATION_OVERLAY",
        status="SUPERSEDED",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        baseline_target_type="MODEL_VERSION",
        baseline_target_id="m20-live",
        source_experiment_id="experiment-fallback",
        promotion_stage="LIVE_ELIGIBLE",
        calibration_overlay_json=CalibrationOverlayProfile(method="identity"),
        source_evidence_json={"source": "fallback"},
        live_eligible=True,
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        approved_at=datetime(2026, 4, 1, 1, tzinfo=timezone.utc),
        activated_at=datetime(2026, 4, 1, 2, tzinfo=timezone.utc),
        superseded_at=datetime(2026, 4, 2, 2, tzinfo=timezone.utc),
    )
    active = ContinualLearningProfileRecord(
        profile_id="profile-active",
        candidate_type="CALIBRATION_OVERLAY",
        status="ACTIVE",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        baseline_target_type="MODEL_VERSION",
        baseline_target_id="m20-live",
        source_experiment_id="experiment-active",
        promotion_stage="LIVE_ELIGIBLE",
        calibration_overlay_json=CalibrationOverlayProfile(
            method="isotonic",
            x_points=[0.2, 0.5],
            y_points=[0.25, 0.55],
            trained_sample_count=64,
        ),
        source_evidence_json={"source": "active"},
        live_eligible=True,
        rollback_target_profile_id="profile-fallback",
        created_at=datetime(2026, 4, 2, tzinfo=timezone.utc),
        approved_at=datetime(2026, 4, 2, 1, tzinfo=timezone.utc),
        activated_at=datetime(2026, 4, 2, 2, tzinfo=timezone.utc),
    )
    return active, fallback


def _read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_lines(path: str) -> list[str]:
    return Path(path).read_text(encoding="utf-8").strip().splitlines()


def test_continual_learning_service_summary_writes_runtime_artifacts() -> None:
    active, fallback = _build_profiles()
    drift_cap = ContinualLearningDriftCapRecord(
        cap_id="cap-1",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        candidate_type="CALIBRATION_OVERLAY",
        status="HEALTHY",
        observed_drift_score=0.03,
        warning_threshold=0.10,
        breach_threshold=0.20,
        reason_code="DRIFT_HEALTHY",
        created_at=datetime(2026, 4, 2, 3, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 2, 3, tzinfo=timezone.utc),
    )

    with TemporaryDirectory() as tmp_dir:
        service = ContinualLearningService(
            repository=_FakeRepo(
                active_profile=active,
                fallback_profile=fallback,
                drift_cap=drift_cap,
            ),
            config=_build_config(tmp_dir),
        )

        summary = asyncio.run(
            service.summary(
                execution_mode="paper",
                symbol="BTC/USD",
                regime_label="TREND_UP",
            )
        )
        drift_caps = asyncio.run(
            service.drift_caps(
                execution_mode="paper",
                symbol="BTC/USD",
                regime_label="TREND_UP",
            )
        )

        assert summary.continual_learning_status == "ACTIVE"
        assert summary.active_profile_id == "profile-active"
        assert summary.latest_drift_cap_status == "HEALTHY"
        assert drift_caps.items[0].cap_id == "cap-1"

        summary_payload = _read_json(service.config.artifacts.summary_path)
        drift_payload = _read_json(service.config.artifacts.drift_caps_summary_path)

        assert summary_payload["active_profile_id"] == "profile-active"
        assert summary_payload["active_candidate_type"] == "CALIBRATION_OVERLAY"
        assert drift_payload["item_count"] == 1
        assert drift_payload["items"][0]["status"] == "HEALTHY"


def test_continual_learning_service_rollback_restores_target_and_writes_history() -> None:
    active, fallback = _build_profiles()
    drift_cap = ContinualLearningDriftCapRecord(
        cap_id="cap-rollback",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        candidate_type="CALIBRATION_OVERLAY",
        status="WATCH",
        observed_drift_score=0.11,
        warning_threshold=0.10,
        breach_threshold=0.20,
        reason_code="DRIFT_WATCH",
        created_at=datetime(2026, 4, 2, 3, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 2, 3, tzinfo=timezone.utc),
    )

    with TemporaryDirectory() as tmp_dir:
        repository = _FakeRepo(
            active_profile=active,
            fallback_profile=fallback,
            drift_cap=drift_cap,
        )
        service = ContinualLearningService(
            repository=repository,
            config=_build_config(tmp_dir),
        )

        decision = asyncio.run(
            service.rollback_active_profile(
                execution_mode="paper",
                symbol="BTC/USD",
                regime_label="TREND_UP",
                decision_id="decision-rollback-1",
                summary_text="rollback for test",
            )
        )

        restored_profile = asyncio.run(
            repository.load_continual_learning_profile(profile_id="profile-fallback")
        )
        current_profile = _read_json(service.config.artifacts.current_profile_path)
        report_path = Path(service.config.artifacts.reports_dir) / "profile-fallback.json"
        report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        events_lines = _read_lines(service.config.artifacts.events_history_path)
        promotion_lines = _read_lines(service.config.artifacts.promotions_history_path)

        assert decision.decision == "ROLLBACK"
        assert decision.research_integrity_json["rollback_target_profile_id"] == "profile-fallback"
        assert restored_profile is not None
        assert restored_profile.status == "ACTIVE"
        assert current_profile["profile_id"] == "profile-fallback"
        assert current_profile["baseline_target_id"] == "m20-live"
        assert current_profile["promotion_stage"] == "LIVE_ELIGIBLE"
        assert report_payload["profile"]["profile_id"] == "profile-fallback"
        assert (
            report_payload["latest_promotion"]["research_integrity_json"][
                "source_experiment_id"
            ]
            == "experiment-fallback"
        )
        assert json.loads(events_lines[-1])["event_type"] == "ROLLBACK_APPLIED"
        assert json.loads(promotion_lines[-1])["decision"] == "ROLLBACK"


def test_live_eligible_profile_stage_is_valid_for_calibration_overlay() -> None:
    profile = ContinualLearningProfileRecord(
        profile_id="profile-valid",
        candidate_type="CALIBRATION_OVERLAY",
        status="APPROVED",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="ALL",
        baseline_target_type="MODEL_VERSION",
        baseline_target_id="m20-live",
        promotion_stage="LIVE_ELIGIBLE",
        live_eligible=True,
    )

    assert profile.promotion_stage == "LIVE_ELIGIBLE"
    assert profile.live_eligible is True


def test_shadow_challenger_live_eligible_stage_is_rejected() -> None:
    try:
        ContinualLearningProfileRecord(
            profile_id="profile-invalid",
            candidate_type="INCREMENTAL_SHADOW_CHALLENGER",
            status="DRAFT",
            execution_mode_scope="paper",
            symbol_scope="BTC/USD",
            regime_scope="ALL",
            baseline_target_type="MODEL_VERSION",
            baseline_target_id="m20-live",
            promotion_stage="LIVE_ELIGIBLE",
            live_eligible=True,
        )
    except ValueError as error:
        assert "INCREMENTAL_SHADOW_CHALLENGER" in str(error)
    else:
        raise AssertionError("Expected live-eligible shadow challenger profile to be rejected")
