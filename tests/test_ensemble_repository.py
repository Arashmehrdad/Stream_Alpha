"""Focused PostgreSQL round-trip tests for the Stream Alpha M20 ensemble tables."""

# pylint: disable=missing-function-docstring

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from uuid import uuid4

import asyncpg
import pytest

from app.ensemble.schemas import (
    EnsembleChallengerRunRecord,
    EnsembleProfileRecord,
    EnsemblePromotionDecisionRecord,
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


def test_ensemble_repository_round_trip_supports_scope_matching_and_reads() -> None:
    asyncio.run(_run_round_trip())


async def _run_round_trip() -> None:  # pylint: disable=too-many-locals
    suffix = uuid4().hex[:10]
    exact_profile_id = f"ensemble-exact-{suffix}"
    broad_profile_id = f"ensemble-broad-{suffix}"
    challenger_run_id = f"ensemble-challenger-{suffix}"
    decision_id = f"ensemble-decision-{suffix}"
    repository = TradingRepository(_postgres_dsn(), "feature_ohlc")

    try:
        await repository.connect()
    except (OSError, asyncpg.CannotConnectNowError) as error:
        pytest.skip(f"PostgreSQL not reachable for ensemble repository test: {error}")
        return

    exact_profile = EnsembleProfileRecord(
        profile_id=exact_profile_id,
        status="ACTIVE",
        approval_stage="ACTIVATED",
        execution_mode_scope="paper",
        symbol_scope="BTC/USD",
        regime_scope="TREND_UP",
        candidate_roster_json=[
            {
                "candidate_id": "generalist-1",
                "candidate_role": "GENERALIST",
                "model_version": "ensemble-generalist-v1",
                "scope_regimes": ["TREND_UP"],
                "enabled": True,
            }
        ],
        created_at=datetime(2026, 3, 21, 12, 0, tzinfo=timezone.utc),
        activated_at=datetime(2026, 3, 21, 12, 30, tzinfo=timezone.utc),
    )
    broad_profile = EnsembleProfileRecord(
        profile_id=broad_profile_id,
        status="ACTIVE",
        approval_stage="ACTIVATED",
        execution_mode_scope="ALL",
        symbol_scope="ALL",
        regime_scope="ALL",
        candidate_roster_json=[
            {
                "candidate_id": "generalist-all",
                "candidate_role": "GENERALIST",
                "model_version": "ensemble-generalist-v1",
                "scope_regimes": [],
                "enabled": True,
            }
        ],
        created_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
        activated_at=datetime(2026, 3, 22, 12, 30, tzinfo=timezone.utc),
    )
    challenger = EnsembleChallengerRunRecord(
        challenger_run_id=challenger_run_id,
        status="EVALUATED",
        config_json={"source": "unit-test"},
        metrics_json={"net_pnl_after_costs": 0.03},
        reason_codes=["EVIDENCE_READY"],
        created_at=datetime(2026, 3, 22, 12, 0, tzinfo=timezone.utc),
        updated_at=datetime(2026, 3, 22, 12, 5, tzinfo=timezone.utc),
    )
    promotion = EnsemblePromotionDecisionRecord(
        decision_id=decision_id,
        target_type="PROFILE",
        target_id=exact_profile_id,
        incumbent_id=broad_profile_id,
        decision="PROMOTE",
        metrics_delta_json={"profit_factor": 0.1},
        safety_checks_json={"reliability_healthy": True},
        reason_codes=["PROMOTION_CRITERIA_PASSED"],
        summary_text="unit-test promotion",
        decided_at=datetime(2026, 3, 22, 12, 10, tzinfo=timezone.utc),
    )

    try:
        await repository.save_ensemble_profile(broad_profile)
        await repository.save_ensemble_profile(exact_profile)
        await repository.save_ensemble_challenger_run(challenger)
        await repository.save_ensemble_promotion_decision(promotion)

        loaded_exact = await repository.load_active_ensemble_profile(
            execution_mode="paper",
            symbol="BTC/USD",
            regime_label="TREND_UP",
        )
        loaded_broad = await repository.load_active_ensemble_profile(
            execution_mode="paper",
            symbol="ETH/USD",
            regime_label="RANGE",
        )
        loaded_profile = await repository.load_ensemble_profile(profile_id=exact_profile_id)
        loaded_profiles = await repository.load_ensemble_profiles(limit=10)
        loaded_challengers = await repository.load_ensemble_challenger_runs(limit=10)
        loaded_promotions = await repository.load_ensemble_promotion_decisions(limit=10)

        assert loaded_exact is not None
        assert loaded_exact.profile_id == exact_profile_id
        assert loaded_exact.execution_mode_scope == "paper"
        assert loaded_broad is not None
        assert loaded_broad.profile_id == broad_profile_id
        assert loaded_profile is not None
        assert loaded_profile.symbol_scope == "BTC/USD"
        assert any(item.profile_id == exact_profile_id for item in loaded_profiles)
        assert any(item.profile_id == broad_profile_id for item in loaded_profiles)
        assert any(item.challenger_run_id == challenger_run_id for item in loaded_challengers)
        assert any(item.decision_id == decision_id for item in loaded_promotions)
    finally:
        pool = repository._require_pool()  # pylint: disable=protected-access
        await pool.execute(
            "DELETE FROM ensemble_promotion_decisions WHERE decision_id = $1",
            decision_id,
        )
        await pool.execute(
            "DELETE FROM ensemble_challenger_runs WHERE challenger_run_id = $1",
            challenger_run_id,
        )
        await pool.execute(
            "DELETE FROM ensemble_profiles WHERE profile_id = $1",
            exact_profile_id,
        )
        await pool.execute(
            "DELETE FROM ensemble_profiles WHERE profile_id = $1",
            broad_profile_id,
        )
        await repository.close()
