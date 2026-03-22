"""Focused tests for the Stream Alpha M21 continual-learning layer."""

# pylint: disable=missing-function-docstring,too-many-arguments

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from app.continual_learning.config import ArtifactConfig, ContinualLearningConfig
from app.continual_learning.schemas import (
    CalibrationOverlayProfile,
    ContinualLearningDecisionType,
    ContinualLearningDriftCapRecord,
    ContinualLearningExperimentRecord,
    ContinualLearningProfileRecord,
    ContinualLearningPromoteProfileRequest,
    ContinualLearningPromotionDecisionRecord,
    ContinualLearningRollbackRequest,
)
from app.continual_learning.service import ContinualLearningService


class _FakeRepo:
    def __init__(
        self,
        *,
        active_profile,
        fallback_profile,
        drift_cap,
        experiments: list | None = None,
        extra_profiles: list | None = None,
        all_drift_caps: list | None = None,
        latest_decision: ContinualLearningDecisionType | None = None,
    ) -> None:
        self._active_profile = active_profile
        self._drift_cap = drift_cap
        self._all_drift_caps = [drift_cap] if all_drift_caps is None else list(all_drift_caps)
        self._profiles = {}
        if active_profile is not None:
            self._profiles[active_profile.profile_id] = active_profile
        if fallback_profile is not None:
            self._profiles[fallback_profile.profile_id] = fallback_profile
        if extra_profiles is not None:
            for profile in extra_profiles:
                self._profiles[profile.profile_id] = profile
        self._experiments = [] if experiments is None else list(experiments)
        self._promotion_decisions = []
        if latest_decision is not None:
            self._promotion_decisions.append(
                ContinualLearningPromotionDecisionRecord(
                    decision_id="promotion-latest",
                    target_type="PROFILE",
                    target_id=(
                        "profile-fallback"
                        if fallback_profile is None
                        else fallback_profile.profile_id
                    ),
                    candidate_type="CALIBRATION_OVERLAY",
                    decision=latest_decision,
                    summary_text="latest promotion",
                    decided_at=datetime(2026, 4, 2, 4, tzinfo=timezone.utc),
                )
            )
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
        return list(self._experiments)[:limit]

    async def load_continual_learning_drift_caps(self, **_kwargs):
        return [self._drift_cap]

    async def load_all_continual_learning_drift_caps(self, *, limit: int):
        return list(self._all_drift_caps)[:limit]

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

    async def promote_continual_learning_profile(
        self,
        *,
        target_profile_id: str,
        execution_mode_scope: str,
        symbol_scope: str,
        regime_scope: str,
        promotion_stage: str,
        live_eligible: bool,
        changed_at,
    ) -> str | None:
        incumbent_profile_id = None
        for profile in list(self._profiles.values()):
            if (
                profile.status == "ACTIVE"
                and profile.execution_mode_scope == execution_mode_scope
                and profile.symbol_scope == symbol_scope
                and profile.regime_scope == regime_scope
                and profile.profile_id != target_profile_id
            ):
                incumbent_profile_id = profile.profile_id
                self._profiles[profile.profile_id] = profile.model_copy(
                    update={"status": "SUPERSEDED", "superseded_at": changed_at}
                )
        target = self._profiles[target_profile_id]
        self._profiles[target_profile_id] = target.model_copy(
            update={
                "status": "ACTIVE",
                "promotion_stage": promotion_stage,
                "live_eligible": live_eligible,
                "approved_at": target.approved_at or changed_at,
                "activated_at": changed_at,
                "superseded_at": None,
            }
        )
        self._active_profile = self._profiles[target_profile_id]
        return incumbent_profile_id

    async def save_continual_learning_promotion_decision(self, decision) -> None:
        self._promotion_decisions.insert(0, decision)

    async def save_continual_learning_event(self, event) -> None:
        self._events.insert(0, event)

    @property
    def saved_promotions(self):
        return list(self._promotion_decisions)

    @property
    def saved_events(self):
        return list(self._events)


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


def test_continual_learning_service_promote_profile_applies_and_persists_truth() -> None:
    active, fallback = _build_profiles()
    drift_cap = ContinualLearningDriftCapRecord(
        cap_id="cap-promote",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        candidate_type="CALIBRATION_OVERLAY",
        status="WATCH",
        observed_drift_score=0.11,
        warning_threshold=0.10,
        breach_threshold=0.20,
        reason_code="DRIFT_WATCH",
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

        response = asyncio.run(
            service.promote_profile(
                ContinualLearningPromoteProfileRequest(
                    decision_id="decision-promote-1",
                    profile_id="profile-fallback",
                    requested_promotion_stage="LIVE_ELIGIBLE",
                    summary_text="promote the persisted fallback profile",
                    reason_codes=["OPERATOR_REVIEWED_EVIDENCE"],
                    operator_confirmed=True,
                ),
                health_overall_status="HEALTHY",
                freshness_status="FRESH",
            )
        )

        promoted = asyncio.run(
            repository.load_continual_learning_profile(profile_id="profile-fallback")
        )
        superseded = asyncio.run(
            repository.load_continual_learning_profile(profile_id="profile-active")
        )

        assert response.success is True
        assert response.blocked is False
        assert response.decision == "PROMOTE"
        assert response.target_profile_id == "profile-fallback"
        assert response.incumbent_profile_id == "profile-active"
        assert response.live_eligible_after is True
        assert promoted is not None
        assert promoted.status == "ACTIVE"
        assert superseded is not None
        assert superseded.status == "SUPERSEDED"
        assert repository.saved_promotions[0].decision == "PROMOTE"
        assert repository.saved_events[0].event_type == "PROMOTION_APPLIED"
        assert (
            _read_json(service.config.artifacts.current_profile_path)["profile_id"]
            == "profile-fallback"
        )


def test_continual_learning_service_promote_profile_blocks_on_breached_drift() -> None:
    active, fallback = _build_profiles()
    drift_cap = ContinualLearningDriftCapRecord(
        cap_id="cap-breached",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        candidate_type="CALIBRATION_OVERLAY",
        status="BREACHED",
        observed_drift_score=0.25,
        warning_threshold=0.10,
        breach_threshold=0.20,
        reason_code="DRIFT_BREACHED",
    )
    repository = _FakeRepo(
        active_profile=active,
        fallback_profile=fallback,
        drift_cap=drift_cap,
    )
    service = ContinualLearningService(
        repository=repository,
        config=_build_config("artifacts/tmp"),
    )

    response = asyncio.run(
        service.promote_profile(
            ContinualLearningPromoteProfileRequest(
                decision_id="decision-promote-block-drift",
                profile_id="profile-fallback",
                requested_promotion_stage="PAPER_APPROVED",
                summary_text="do not promote on breached drift",
                reason_codes=["OPERATOR_REVIEWED_EVIDENCE"],
                operator_confirmed=True,
            ),
            health_overall_status="HEALTHY",
            freshness_status="FRESH",
        )
    )

    active_after = asyncio.run(
        repository.load_continual_learning_profile(profile_id="profile-active")
    )

    assert response.success is False
    assert response.blocked is True
    assert response.decision == "HOLD"
    assert "CONTINUAL_LEARNING_DRIFT_CAP_BREACHED" in response.reason_codes
    assert active_after is not None
    assert active_after.status == "ACTIVE"
    assert repository.saved_promotions[0].decision == "HOLD"
    assert repository.saved_events[0].event_type == "PROMOTION_BLOCKED"


def test_continual_learning_service_promote_profile_blocks_on_degraded_health() -> None:
    active, fallback = _build_profiles()
    drift_cap = ContinualLearningDriftCapRecord(
        cap_id="cap-health",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        candidate_type="CALIBRATION_OVERLAY",
        status="WATCH",
        observed_drift_score=0.11,
        warning_threshold=0.10,
        breach_threshold=0.20,
        reason_code="DRIFT_WATCH",
    )
    repository = _FakeRepo(
        active_profile=active,
        fallback_profile=fallback,
        drift_cap=drift_cap,
    )
    service = ContinualLearningService(
        repository=repository,
        config=_build_config("artifacts/tmp"),
    )

    response = asyncio.run(
        service.promote_profile(
            ContinualLearningPromoteProfileRequest(
                decision_id="decision-promote-block-health",
                profile_id="profile-fallback",
                requested_promotion_stage="PAPER_APPROVED",
                summary_text="hold while health is degraded",
                reason_codes=["OPERATOR_REVIEWED_EVIDENCE"],
                operator_confirmed=True,
            ),
            health_overall_status="DEGRADED",
            freshness_status="FRESH",
        )
    )

    assert response.success is False
    assert response.blocked is True
    assert "CONTINUAL_LEARNING_BLOCKED_BY_HEALTH_STATUS" in response.reason_codes
    assert repository.saved_promotions[0].decision == "HOLD"
    assert repository.saved_events[0].event_type == "PROMOTION_BLOCKED"


def test_continual_learning_service_promote_profile_blocks_shadow_challenger_live() -> None:
    active, fallback = _build_profiles()
    shadow_profile = ContinualLearningProfileRecord(
        profile_id="profile-shadow-1",
        candidate_type="INCREMENTAL_SHADOW_CHALLENGER",
        status="APPROVED",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        baseline_target_type="MODEL_VERSION",
        baseline_target_id="m20-live",
        source_experiment_id="experiment-shadow-1",
        promotion_stage="SHADOW_ONLY",
        live_eligible=False,
    )
    drift_cap = ContinualLearningDriftCapRecord(
        cap_id="cap-shadow",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        candidate_type="INCREMENTAL_SHADOW_CHALLENGER",
        status="HEALTHY",
        observed_drift_score=0.02,
        warning_threshold=0.10,
        breach_threshold=0.20,
        reason_code="DRIFT_HEALTHY",
    )
    repository = _FakeRepo(
        active_profile=active,
        fallback_profile=fallback,
        drift_cap=drift_cap,
        extra_profiles=[shadow_profile],
    )
    service = ContinualLearningService(
        repository=repository,
        config=_build_config("artifacts/tmp"),
    )

    response = asyncio.run(
        service.promote_profile(
            ContinualLearningPromoteProfileRequest(
                decision_id="decision-promote-shadow-live",
                profile_id="profile-shadow-1",
                requested_promotion_stage="LIVE_ELIGIBLE",
                summary_text="shadow challenger must stay shadow only",
                reason_codes=["OPERATOR_REVIEWED_EVIDENCE"],
                operator_confirmed=True,
            ),
            health_overall_status="HEALTHY",
            freshness_status="FRESH",
        )
    )

    assert response.success is False
    assert response.blocked is True
    assert "CONTINUAL_LEARNING_SHADOW_CHALLENGER_LIVE_BLOCKED" in response.reason_codes
    assert repository.saved_promotions[0].decision == "HOLD"
    assert repository.saved_events[0].event_type == "PROMOTION_BLOCKED"


def test_continual_learning_service_rollback_profile_applies_and_persists_truth() -> None:
    active, fallback = _build_profiles()
    drift_cap = ContinualLearningDriftCapRecord(
        cap_id="cap-rollback-workflow",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        candidate_type="CALIBRATION_OVERLAY",
        status="WATCH",
        observed_drift_score=0.11,
        warning_threshold=0.10,
        breach_threshold=0.20,
        reason_code="DRIFT_WATCH",
    )
    repository = _FakeRepo(
        active_profile=active,
        fallback_profile=fallback,
        drift_cap=drift_cap,
    )
    service = ContinualLearningService(
        repository=repository,
        config=_build_config("artifacts/tmp"),
    )

    response = asyncio.run(
        service.rollback_profile(
            ContinualLearningRollbackRequest(
                decision_id="decision-rollback-workflow",
                execution_mode="paper",
                symbol="BTC/USD",
                regime_label="TREND_UP",
                summary_text="restore prior approved profile",
                operator_confirmed=True,
            ),
            health_overall_status="HEALTHY",
            freshness_status="FRESH",
        )
    )

    restored = asyncio.run(
        repository.load_continual_learning_profile(profile_id="profile-fallback")
    )

    assert response.success is True
    assert response.blocked is False
    assert response.decision == "ROLLBACK"
    assert response.target_profile_id == "profile-fallback"
    assert restored is not None
    assert restored.status == "ACTIVE"
    assert repository.saved_promotions[0].decision == "ROLLBACK"
    assert repository.saved_events[0].event_type == "ROLLBACK_APPLIED"


def test_continual_learning_service_rollback_profile_blocks_without_target() -> None:
    active, fallback = _build_profiles()
    del fallback
    active_without_target = active.model_copy(update={"rollback_target_profile_id": None})
    drift_cap = ContinualLearningDriftCapRecord(
        cap_id="cap-rollback-block",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        candidate_type="CALIBRATION_OVERLAY",
        status="WATCH",
        observed_drift_score=0.11,
        warning_threshold=0.10,
        breach_threshold=0.20,
        reason_code="DRIFT_WATCH",
    )
    repository = _FakeRepo(
        active_profile=active_without_target,
        fallback_profile=None,
        drift_cap=drift_cap,
    )
    service = ContinualLearningService(
        repository=repository,
        config=_build_config("artifacts/tmp"),
    )

    response = asyncio.run(
        service.rollback_profile(
            ContinualLearningRollbackRequest(
                decision_id="decision-rollback-blocked",
                execution_mode="paper",
                symbol="BTC/USD",
                regime_label="TREND_UP",
                summary_text="cannot rollback without explicit target",
                operator_confirmed=True,
            ),
            health_overall_status="HEALTHY",
            freshness_status="FRESH",
        )
    )

    assert response.success is False
    assert response.blocked is True
    assert response.decision == "HOLD"
    assert "CONTINUAL_LEARNING_NO_ROLLBACK_TARGET" in response.reason_codes
    assert repository.saved_promotions[0].decision == "HOLD"
    assert repository.saved_events[0].event_type == "ROLLBACK_BLOCKED"


def test_resolve_runtime_context_includes_active_profile_and_health_freeze() -> None:
    active, fallback = _build_profiles()
    drift_cap = ContinualLearningDriftCapRecord(
        cap_id="cap-runtime-1",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        candidate_type="CALIBRATION_OVERLAY",
        status="WATCH",
        observed_drift_score=0.11,
        warning_threshold=0.10,
        breach_threshold=0.20,
        reason_code="DRIFT_WATCH",
    )
    service = ContinualLearningService(
        repository=_FakeRepo(
            active_profile=active,
            fallback_profile=fallback,
            drift_cap=drift_cap,
            latest_decision="HOLD",
        ),
        config=_build_config("artifacts/tmp/m21-runtime-context"),
    )

    context = asyncio.run(
        service.resolve_runtime_context(
            execution_mode="paper",
            symbol="BTC/USD",
            regime_label="TREND_UP",
            health_overall_status="DEGRADED",
            freshness_status="FRESH",
        )
    )

    assert context.enabled is True
    assert context.active_profile_id == "profile-active"
    assert context.promotion_stage == "LIVE_ELIGIBLE"
    assert context.baseline_target_id == "m20-live"
    assert context.drift_cap_status == "WATCH"
    assert context.latest_promotion_decision == "HOLD"
    assert context.frozen_by_health_gate is True
    assert "CONTINUAL_LEARNING_FROZEN_BY_HEALTH_GATE" in context.reason_codes


def test_resolve_runtime_context_returns_idle_when_no_active_profile() -> None:
    _, fallback = _build_profiles()
    drift_cap = ContinualLearningDriftCapRecord(
        cap_id="cap-runtime-2",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        candidate_type="CALIBRATION_OVERLAY",
        status="HEALTHY",
        observed_drift_score=0.02,
        warning_threshold=0.10,
        breach_threshold=0.20,
        reason_code="DRIFT_HEALTHY",
    )
    service = ContinualLearningService(
        repository=_FakeRepo(
            active_profile=None,
            fallback_profile=fallback,
            drift_cap=drift_cap,
            latest_decision="REJECT",
        ),
        config=_build_config("artifacts/tmp/m21-runtime-context-no-active"),
    )

    context = asyncio.run(
        service.resolve_runtime_context(
            execution_mode="paper",
            symbol="BTC/USD",
            regime_label="TREND_UP",
        )
    )

    assert context.enabled is True
    assert context.active_profile_id is None
    assert context.live_eligible is False
    assert context.drift_cap_status == "HEALTHY"
    assert context.latest_promotion_decision == "REJECT"
    assert context.reason_codes == ["NO_ACTIVE_CONTINUAL_LEARNING_PROFILE"]


def test_resolve_runtime_context_marks_repository_unavailable() -> None:
    service = ContinualLearningService(
        repository=None,
        config=_build_config("artifacts/tmp/m21-runtime-context-unavailable"),
    )

    context = asyncio.run(
        service.resolve_runtime_context(
            execution_mode="paper",
            symbol="BTC/USD",
            regime_label="TREND_UP",
        )
    )

    assert context.enabled is True
    assert context.active_profile_id is None
    assert context.frozen_by_health_gate is False
    assert context.reason_codes == ["CONTINUAL_LEARNING_REPOSITORY_UNAVAILABLE"]


def test_summary_all_scope_aggregates_active_profiles_and_worst_drift() -> None:
    active, fallback = _build_profiles()
    second_active = fallback.model_copy(
        update={
            "profile_id": "profile-active-2",
            "status": "ACTIVE",
            "symbol_scope": "ETH/USD",
            "regime_scope": "RANGE",
            "activated_at": datetime(2026, 4, 2, 2, 30, tzinfo=timezone.utc),
            "superseded_at": None,
        }
    )
    drift_watch = ContinualLearningDriftCapRecord(
        cap_id="cap-watch",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        candidate_type="CALIBRATION_OVERLAY",
        status="WATCH",
        observed_drift_score=0.12,
        warning_threshold=0.10,
        breach_threshold=0.20,
        reason_code="DRIFT_WATCH",
    )
    drift_breached = ContinualLearningDriftCapRecord(
        cap_id="cap-breached",
        execution_mode_scope="paper",
        symbol_scope="ETH/USD",
        regime_scope="RANGE",
        candidate_type="CALIBRATION_OVERLAY",
        status="BREACHED",
        observed_drift_score=0.24,
        warning_threshold=0.10,
        breach_threshold=0.20,
        reason_code="DRIFT_BREACHED",
    )
    service = ContinualLearningService(
        repository=_FakeRepo(
            active_profile=active,
            fallback_profile=fallback,
            drift_cap=drift_watch,
            extra_profiles=[second_active],
            all_drift_caps=[drift_watch, drift_breached],
        ),
        config=_build_config("artifacts/tmp/m21-summary-all"),
    )

    summary = asyncio.run(
        service.summary(
            execution_mode="ALL",
            symbol="ALL",
            regime_label="ALL",
        )
    )

    assert summary.active_profile_count == 2
    assert summary.active_profile_id is None
    assert summary.active_candidate_type is None
    assert summary.latest_drift_cap_status == "BREACHED"
    assert "AGGREGATED_SCOPE_SUMMARY" in summary.reason_codes


def test_profiles_and_experiments_apply_scope_filtering() -> None:
    active, fallback = _build_profiles()
    experiment_all = ContinualLearningExperimentRecord(
        experiment_id="exp-all",
        candidate_type="CALIBRATION_OVERLAY",
        status="EVALUATED",
        execution_mode_scope="ALL",
        symbol_scope="ALL",
        regime_scope="ALL",
        baseline_target_type="MODEL_VERSION",
        baseline_target_id="m20-live",
    )
    experiment_other_symbol = ContinualLearningExperimentRecord(
        experiment_id="exp-eth",
        candidate_type="CALIBRATION_OVERLAY",
        status="EVALUATED",
        execution_mode_scope="paper",
        symbol_scope="ETH/USD",
        regime_scope="TREND_UP",
        baseline_target_type="MODEL_VERSION",
        baseline_target_id="m20-live",
    )
    service = ContinualLearningService(
        repository=_FakeRepo(
            active_profile=active,
            fallback_profile=fallback,
            drift_cap=ContinualLearningDriftCapRecord(
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
            ),
            experiments=[
                experiment_all,
                experiment_other_symbol,
            ],
        ),
        config=_build_config("artifacts/tmp/m21-scope-filter"),
    )

    profiles = asyncio.run(
        service.profiles(
            execution_mode="paper",
            symbol="BTC/USD",
            regime_label="TREND_UP",
            limit=50,
        )
    )
    experiments = asyncio.run(
        service.experiments(
            execution_mode="paper",
            symbol="BTC/USD",
            regime_label="TREND_UP",
            limit=50,
        )
    )

    assert all(item.symbol_scope in {"BTC/USD", "ALL"} for item in profiles.items)
    assert [item.experiment_id for item in experiments.items] == ["exp-all"]
