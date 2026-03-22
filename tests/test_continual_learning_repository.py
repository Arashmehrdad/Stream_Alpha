"""Focused PostgreSQL round-trip tests for the Stream Alpha M21 continual-learning tables."""

# pylint: disable=missing-function-docstring
# pylint: disable=too-many-locals,too-many-statements

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from uuid import uuid4

from app.continual_learning.schemas import (
    CalibrationOverlayProfile,
    ContinualLearningDriftCapRecord,
    ContinualLearningEventRecord,
    ContinualLearningExperimentRecord,
    ContinualLearningProfileRecord,
    ContinualLearningPromotionDecisionRecord,
)
from app.trading.repository import TradingRepository


def _postgres_dsn() -> str:
    host = os.getenv("POSTGRES_HOST", "127.0.0.1").strip() or "127.0.0.1"
    if host == "postgres":
        host = "127.0.0.1"
    port = int(os.getenv("POSTGRES_PORT", "5432").strip())
    database = os.getenv("POSTGRES_DB", "streamalpha").strip() or "streamalpha"
    user = os.getenv("POSTGRES_USER", "streamalpha").strip() or "streamalpha"
    password = os.getenv("POSTGRES_PASSWORD", "change-me-local-only").strip()
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def test_continual_learning_repository_round_trip_supports_scope_and_rollback() -> None:
    asyncio.run(_run_round_trip())


async def _run_round_trip() -> None:
    suffix = uuid4().hex[:10]
    experiment_id = f"experiment-{suffix}"
    global_profile_id = f"profile-global-{suffix}"
    scoped_profile_id = f"profile-scoped-{suffix}"
    rollback_profile_id = f"profile-rollback-{suffix}"
    cap_id = f"cap-{suffix}"
    decision_id = f"decision-{suffix}"
    event_id = f"event-{suffix}"
    rollback_decision_id = f"rollback-{suffix}"
    repository = TradingRepository(_postgres_dsn(), "feature_ohlc")
    await repository.connect()

    experiment = ContinualLearningExperimentRecord(
        experiment_id=experiment_id,
        candidate_type="CALIBRATION_OVERLAY",
        status="APPROVED",
        execution_mode_scope="shadow",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        baseline_target_type="MODEL_VERSION",
        baseline_target_id="m20-live",
        base_model_version="m20-live",
        candidate_model_version="m21-overlay-1",
        reference_window_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
        reference_window_end=datetime(2026, 3, 10, tzinfo=timezone.utc),
        update_window_start=datetime(2026, 3, 11, tzinfo=timezone.utc),
        update_window_end=datetime(2026, 3, 20, tzinfo=timezone.utc),
        shadow_window_start=datetime(2026, 3, 21, tzinfo=timezone.utc),
        shadow_window_end=datetime(2026, 4, 1, tzinfo=timezone.utc),
        config_json={"source": "unit-test"},
        metrics_before_json={"ece": 0.07, "trade_count": 20},
        metrics_after_json={"ece": 0.04, "trade_count": 20},
        shadow_summary_json={"shadow_precision": 0.61},
        research_integrity_json={"slice_count": 3, "leakage_detected": False},
        artifact_paths_json={"report": "artifacts/continual_learning/experiments/x.json"},
        reason_codes=["EXPERIMENT_APPROVED"],
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 1, 1, tzinfo=timezone.utc),
    )
    global_profile = ContinualLearningProfileRecord(
        profile_id=global_profile_id,
        candidate_type="CALIBRATION_OVERLAY",
        status="ACTIVE",
        execution_mode_scope="ALL",
        symbol_scope="ALL",
        regime_scope="ALL",
        baseline_target_type="MODEL_VERSION",
        baseline_target_id="m20-live",
        promotion_stage="LIVE_ELIGIBLE",
        calibration_overlay_json=CalibrationOverlayProfile(method="identity"),
        source_evidence_json={"source": "global"},
        live_eligible=True,
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        activated_at=datetime(2026, 4, 1, 2, tzinfo=timezone.utc),
    )
    scoped_profile = ContinualLearningProfileRecord(
        profile_id=scoped_profile_id,
        candidate_type="CALIBRATION_OVERLAY",
        status="ACTIVE",
        execution_mode_scope="shadow",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        baseline_target_type="MODEL_VERSION",
        baseline_target_id="m20-live",
        source_experiment_id=experiment_id,
        promotion_stage="LIVE_ELIGIBLE",
        calibration_overlay_json=CalibrationOverlayProfile(
            method="isotonic",
            x_points=[0.2, 0.5, 0.8],
            y_points=[0.25, 0.55, 0.78],
            trained_sample_count=128,
            source_window="shadow:last_128",
        ),
        source_evidence_json={"source": "scoped"},
        live_eligible=True,
        rollback_target_profile_id=rollback_profile_id,
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        approved_at=datetime(2026, 4, 1, 2, tzinfo=timezone.utc),
        activated_at=datetime(2026, 4, 1, 3, tzinfo=timezone.utc),
    )
    rollback_profile = ContinualLearningProfileRecord(
        profile_id=rollback_profile_id,
        candidate_type="CALIBRATION_OVERLAY",
        status="SUPERSEDED",
        execution_mode_scope="shadow",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        baseline_target_type="MODEL_VERSION",
        baseline_target_id="m20-live",
        source_experiment_id=experiment_id,
        promotion_stage="LIVE_ELIGIBLE",
        calibration_overlay_json=CalibrationOverlayProfile(method="identity"),
        source_evidence_json={"source": "rollback-target"},
        live_eligible=True,
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        approved_at=datetime(2026, 4, 1, 1, tzinfo=timezone.utc),
        activated_at=datetime(2026, 4, 1, 1, 30, tzinfo=timezone.utc),
        superseded_at=datetime(2026, 4, 1, 3, tzinfo=timezone.utc),
    )
    drift_cap = ContinualLearningDriftCapRecord(
        cap_id=cap_id,
        execution_mode_scope="shadow",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        candidate_type="CALIBRATION_OVERLAY",
        status="WATCH",
        observed_drift_score=0.12,
        warning_threshold=0.10,
        breach_threshold=0.20,
        reason_code="DRIFT_WATCH",
        detail="unit-test",
        created_at=datetime(2026, 4, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 4, 1, 4, tzinfo=timezone.utc),
    )
    promotion = ContinualLearningPromotionDecisionRecord(
        decision_id=decision_id,
        target_type="PROFILE",
        target_id=scoped_profile_id,
        incumbent_id=global_profile_id,
        candidate_type="CALIBRATION_OVERLAY",
        decision="PROMOTE",
        live_eligible_after_decision=True,
        metrics_delta_json={"ece_delta": -0.03},
        safety_checks_json={"reliability_healthy": True},
        research_integrity_json={"slice_count": 3, "leakage_detected": False},
        reason_codes=["PROMOTION_CRITERIA_PASSED"],
        summary_text="unit-test promotion",
        decided_at=datetime(2026, 4, 1, 4, 5, tzinfo=timezone.utc),
    )
    event = ContinualLearningEventRecord(
        event_id=event_id,
        event_type="PROMOTION_APPLIED",
        profile_id=scoped_profile_id,
        experiment_id=experiment_id,
        decision_id=decision_id,
        reason_code="PROMOTION_CRITERIA_PASSED",
        payload_json={"source": "unit-test"},
        created_at=datetime(2026, 4, 1, 4, 6, tzinfo=timezone.utc),
    )

    try:
        await repository.save_continual_learning_experiment(experiment)
        await repository.save_continual_learning_profile(global_profile)
        await repository.save_continual_learning_profile(scoped_profile)
        await repository.save_continual_learning_profile(rollback_profile)
        await repository.save_continual_learning_drift_cap(drift_cap)
        await repository.save_continual_learning_promotion_decision(promotion)
        await repository.save_continual_learning_event(event)

        loaded_active = await repository.load_active_continual_learning_profile(
            execution_mode="shadow",
            symbol="BTC/USD",
            regime_label="TREND_UP",
        )
        loaded_experiments = await repository.load_continual_learning_experiments(limit=5)
        loaded_profiles = await repository.load_continual_learning_profiles(limit=10)
        loaded_drift = await repository.load_latest_continual_learning_drift_cap(
            execution_mode="shadow",
            symbol="BTC/USD",
            regime_label="TREND_UP",
        )
        loaded_promotions = await repository.load_continual_learning_promotion_decisions(
            limit=5
        )
        loaded_events = await repository.load_continual_learning_events(limit=5)

        assert loaded_active is not None
        assert loaded_active.profile_id == scoped_profile_id
        loaded_experiment = next(
            item for item in loaded_experiments if item.experiment_id == experiment_id
        )
        loaded_profile = next(
            item for item in loaded_profiles if item.profile_id == scoped_profile_id
        )
        assert loaded_experiment.baseline_target_type == "MODEL_VERSION"
        assert loaded_experiment.baseline_target_id == "m20-live"
        assert loaded_experiment.reference_window_end == datetime(
            2026, 3, 10, tzinfo=timezone.utc
        )
        assert loaded_experiment.update_window_end == datetime(
            2026, 3, 20, tzinfo=timezone.utc
        )
        assert loaded_experiment.shadow_window_end == datetime(
            2026, 4, 1, tzinfo=timezone.utc
        )
        assert loaded_experiment.metrics_before_json["ece"] == 0.07
        assert loaded_experiment.metrics_after_json["ece"] == 0.04
        assert loaded_experiment.research_integrity_json["slice_count"] == 3
        assert loaded_profile.baseline_target_id == "m20-live"
        assert loaded_profile.source_experiment_id == experiment_id
        assert loaded_profile.promotion_stage == "LIVE_ELIGIBLE"
        assert loaded_drift is not None
        assert loaded_drift.status == "WATCH"
        loaded_promotion = next(
            item for item in loaded_promotions if item.decision_id == decision_id
        )
        assert loaded_promotion.research_integrity_json["slice_count"] == 3
        assert any(item.event_id == event_id for item in loaded_events)

        changed_at = datetime(2026, 4, 1, 4, 10, tzinfo=timezone.utc)
        await repository.rollback_continual_learning_profile(
            active_profile_id=scoped_profile_id,
            rollback_target_profile_id=rollback_profile_id,
            changed_at=changed_at,
        )
        rollback_decision = ContinualLearningPromotionDecisionRecord(
            decision_id=rollback_decision_id,
            target_type="PROFILE",
            target_id=rollback_profile_id,
            incumbent_id=scoped_profile_id,
            candidate_type="CALIBRATION_OVERLAY",
            decision="ROLLBACK",
            live_eligible_after_decision=True,
            metrics_delta_json={"rolled_back_profile_id": scoped_profile_id},
            safety_checks_json={"runtime_rollback": True},
            research_integrity_json={"rollback_verified": True},
            reason_codes=["CONTINUAL_LEARNING_ROLLBACK_TARGET_ACTIVATED"],
            summary_text="unit-test rollback",
            decided_at=changed_at,
        )
        await repository.save_continual_learning_promotion_decision(rollback_decision)

        reloaded_active = await repository.load_active_continual_learning_profile(
            execution_mode="shadow",
            symbol="BTC/USD",
            regime_label="TREND_UP",
        )
        reloaded_previous = await repository.load_continual_learning_profile(
            profile_id=scoped_profile_id
        )
        reloaded_target = await repository.load_continual_learning_profile(
            profile_id=rollback_profile_id
        )

        assert reloaded_active is not None
        assert reloaded_active.profile_id == rollback_profile_id
        assert reloaded_previous is not None
        assert reloaded_previous.status == "ROLLED_BACK"
        assert reloaded_target is not None
        assert reloaded_target.status == "ACTIVE"
    finally:
        pool = repository._require_pool()  # pylint: disable=protected-access
        await pool.execute(
            "DELETE FROM continual_learning_events WHERE event_id = $1",
            event_id,
        )
        await pool.execute(
            "DELETE FROM continual_learning_promotion_decisions WHERE decision_id = $1",
            rollback_decision_id,
        )
        await pool.execute(
            "DELETE FROM continual_learning_promotion_decisions WHERE decision_id = $1",
            decision_id,
        )
        await pool.execute(
            "DELETE FROM continual_learning_drift_caps WHERE cap_id = $1",
            cap_id,
        )
        await pool.execute(
            "DELETE FROM continual_learning_profiles WHERE profile_id = $1",
            scoped_profile_id,
        )
        await pool.execute(
            "DELETE FROM continual_learning_profiles WHERE profile_id = $1",
            rollback_profile_id,
        )
        await pool.execute(
            "DELETE FROM continual_learning_profiles WHERE profile_id = $1",
            global_profile_id,
        )
        await pool.execute(
            "DELETE FROM continual_learning_experiments WHERE experiment_id = $1",
            experiment_id,
        )
        await repository.close()


def test_continual_learning_experiment_rejects_reversed_windows() -> None:
    try:
        ContinualLearningExperimentRecord(
            experiment_id="experiment-invalid",
            candidate_type="CALIBRATION_OVERLAY",
            status="DRAFT",
            execution_mode_scope="paper",
            symbol_scope="BTC/USD",
            regime_scope="ALL",
            baseline_target_type="MODEL_VERSION",
            baseline_target_id="m20-live",
            reference_window_start=datetime(2026, 4, 2, tzinfo=timezone.utc),
            reference_window_end=datetime(2026, 4, 1, tzinfo=timezone.utc),
        )
    except ValueError as error:
        assert "reference_window_start" in str(error)
    else:
        raise AssertionError("Expected reversed reference windows to be rejected")
