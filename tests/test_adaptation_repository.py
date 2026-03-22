"""Focused PostgreSQL round-trip tests for the Stream Alpha M19 adaptation tables."""

# pylint: disable=missing-function-docstring
# pylint: disable=too-many-locals

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg
import pytest

from app.adaptation.schemas import (
    AdaptiveChallengerRunRecord,
    AdaptiveDriftRecord,
    AdaptivePerformanceWindow,
    AdaptiveProfileRecord,
    AdaptivePromotionDecisionRecord,
    CalibrationProfile,
    SizingPolicy,
    ThresholdPolicy,
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


def test_adaptation_repository_round_trip_supports_profile_drift_and_promotion() -> None:
    asyncio.run(_run_round_trip())


async def _run_round_trip() -> None:
    suffix = uuid4().hex[:10]
    profile_id = f"profile-{suffix}"
    challenger_run_id = f"challenger-{suffix}"
    decision_id = f"decision-{suffix}"
    repository = TradingRepository(_postgres_dsn(), "feature_ohlc")
    try:
        await repository.connect()
    except (OSError, asyncpg.CannotConnectNowError) as error:
        pytest.skip(f"PostgreSQL not reachable for adaptation repository test: {error}")
        return

    drift = AdaptiveDriftRecord(
        symbol="BTC/USD",
        regime_label="ALL",
        detector_name="psi",
        window_id=f"window-{suffix}",
        reference_window_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
        reference_window_end=datetime(2026, 3, 10, tzinfo=timezone.utc),
        live_window_start=datetime(2026, 3, 11, tzinfo=timezone.utc),
        live_window_end=datetime(2026, 3, 22, tzinfo=timezone.utc),
        drift_score=0.11,
        warning_threshold=0.10,
        breach_threshold=0.20,
        status="WATCH",
        reason_code="DRIFT_WATCH",
        detail="unit-test",
    )
    performance = AdaptivePerformanceWindow(
        execution_mode="shadow",
        symbol="BTC/USD",
        regime_label="ALL",
        window_id=f"perf-{suffix}",
        window_type="trade_count",
        window_start=datetime(2026, 3, 10, tzinfo=timezone.utc),
        window_end=datetime(2026, 3, 22, tzinfo=timezone.utc),
        trade_count=20,
        net_pnl_after_costs=0.02,
        max_drawdown=0.01,
        profit_factor=1.2,
        expectancy=0.001,
        win_rate=0.55,
        precision=0.57,
        avg_slippage_bps=3.0,
        blocked_trade_rate=0.10,
        shadow_divergence_rate=0.05,
        health_context={"health_overall_status": "HEALTHY"},
    )
    profile = AdaptiveProfileRecord(
        profile_id=profile_id,
        status="ACTIVE",
        execution_mode_scope="shadow",
        symbol_scope="BTC/USD",
        regime_scope="ALL",
        threshold_policy_json=ThresholdPolicy(buy_threshold_delta=0.02),
        sizing_policy_json=SizingPolicy(size_multiplier=1.1),
        calibration_profile_json=CalibrationProfile(method="identity"),
        source_evidence_json={"evidence": "unit-test"},
    )
    challenger = AdaptiveChallengerRunRecord(
        challenger_run_id=challenger_run_id,
        status="EVALUATED",
        train_window_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        train_window_end=datetime(2026, 2, 29, tzinfo=timezone.utc),
        validation_window_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
        validation_window_end=datetime(2026, 3, 10, tzinfo=timezone.utc),
        shadow_window_start=datetime(2026, 3, 11, tzinfo=timezone.utc),
        shadow_window_end=datetime(2026, 3, 22, tzinfo=timezone.utc),
        candidate_model_version="m19-shadow-1",
        config_json={"source": "unit-test"},
        metrics_json={"net_pnl_after_costs": 0.02},
        shadow_summary_json={"shadow_divergence_rate": 0.05},
        artifact_paths_json={"report": "artifacts/adaptation/challengers/x/report.json"},
        reason_codes=["EVIDENCE_READY"],
    )
    promotion = AdaptivePromotionDecisionRecord(
        decision_id=decision_id,
        target_type="PROFILE",
        target_id=profile_id,
        incumbent_id=None,
        decision="PROMOTE",
        metrics_delta_json={"net_pnl_after_costs": 0.02},
        safety_checks_json={"reliability_healthy": True},
        research_integrity_json={"trade_count": 20},
        reason_codes=["PROMOTION_CRITERIA_PASSED"],
        summary_text="unit-test promotion",
        decided_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
    )

    try:
        await repository.save_adaptive_drift_state(drift)
        await repository.save_adaptive_performance_window(performance)
        await repository.save_adaptive_profile(profile)
        await repository.save_adaptive_challenger_run(challenger)
        await repository.save_adaptive_promotion_decision(promotion)

        loaded_profile = await repository.load_active_adaptive_profile(
            execution_mode="shadow",
            symbol="BTC/USD",
            regime_label="ALL",
        )
        loaded_drift = await repository.load_latest_adaptive_drift_state(
            symbol="BTC/USD",
            regime_label="ALL",
        )
        loaded_performance = await repository.load_latest_adaptive_performance_window(
            execution_mode="shadow",
            symbol="BTC/USD",
            regime_label="ALL",
        )
        loaded_challengers = await repository.load_adaptive_challenger_runs(limit=5)
        loaded_promotions = await repository.load_adaptive_promotion_decisions(limit=5)

        assert loaded_profile is not None
        assert loaded_profile.profile_id == profile_id
        assert loaded_drift is not None
        assert loaded_drift.status == "WATCH"
        assert loaded_performance is not None
        assert loaded_performance.trade_count == 20
        assert any(item.challenger_run_id == challenger_run_id for item in loaded_challengers)
        assert any(item.decision_id == decision_id for item in loaded_promotions)
    finally:
        pool = repository._require_pool()  # pylint: disable=protected-access
        await pool.execute(
            "DELETE FROM adaptive_promotion_decisions WHERE decision_id = $1",
            decision_id,
        )
        await pool.execute(
            "DELETE FROM adaptive_challenger_runs WHERE challenger_run_id = $1",
            challenger_run_id,
        )
        await pool.execute(
            "DELETE FROM adaptive_profiles WHERE profile_id = $1",
            profile_id,
        )
        await pool.execute(
            (
                "DELETE FROM adaptive_performance_windows "
                "WHERE execution_mode = $1 AND symbol = $2 AND window_id = $3"
            ),
            performance.execution_mode,
            performance.symbol,
            performance.window_id,
        )
        await pool.execute(
            (
                "DELETE FROM adaptive_drift_state "
                "WHERE symbol = $1 AND detector_name = $2 AND window_id = $3"
            ),
            drift.symbol,
            drift.detector_name,
            drift.window_id,
        )
        await repository.close()
