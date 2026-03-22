"""Packet 2 research, profile activation, and rollback tests for M20."""

# pylint: disable=duplicate-code,too-many-lines,missing-function-docstring,too-few-public-methods

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from app.common.time import to_rfc3339
from app.ensemble.promote import (
    activate_draft_profile,
    build_draft_profile,
    save_challenger_evidence,
    save_draft_profile,
)
from app.ensemble.research import (
    GENERALIST,
    RANGE_SPECIALIST,
    TREND_SPECIALIST,
    evaluate_registry_candidates,
    load_registry_research_candidates,
    select_runtime_roster,
)
from app.ensemble.rollback import rollback_active_profile
from app.ensemble.schemas import (
    EnsembleEvaluationSliceMetrics,
    EnsembleProfileRecord,
    EnsembleResearchCandidate,
    EnsembleResearchResult,
)
from app.training.dataset import (
    DatasetSample,
    ModelHyperparameters,
    TrainingConfig,
    TrainingDataset,
)
from app.training.registry import (
    export_external_model_to_registry,
    write_current_registry_entry,
)


class LookupProbabilityModel:
    """Serializable candidate model that looks up one deterministic prob_up per row."""

    def __init__(self, mapping: dict[float, float]) -> None:
        self._mapping = {float(key): float(value) for key, value in mapping.items()}

    def predict_proba(self, rows: list[dict[str, Any]]) -> list[list[float]]:
        payload: list[list[float]] = []
        for row in rows:
            prob_up = self._mapping[float(row["close_price"])]
            payload.append([1.0 - prob_up, prob_up])
        return payload


class _FakeEnsembleRepo:
    """Minimal in-memory repository for Packet 2 profile lifecycle tests."""

    def __init__(self, *, incumbent: EnsembleProfileRecord | None = None) -> None:
        self.profiles: dict[str, EnsembleProfileRecord] = {}
        self.promotion_decisions = []
        self.challenger_runs = []
        if incumbent is not None:
            self.profiles[incumbent.profile_id] = incumbent

    async def save_ensemble_profile(self, record: EnsembleProfileRecord) -> None:
        self.profiles[record.profile_id] = record

    async def load_active_ensemble_profile(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
    ) -> EnsembleProfileRecord | None:
        del execution_mode, symbol, regime_label
        active_profiles = [
            profile
            for profile in self.profiles.values()
            if profile.status == "ACTIVE"
        ]
        return None if not active_profiles else active_profiles[0]

    async def save_ensemble_promotion_decision(self, record) -> None:
        self.promotion_decisions.insert(0, record)

    async def save_ensemble_challenger_run(self, record) -> None:
        self.challenger_runs.insert(0, record)

    async def load_ensemble_profile(
        self,
        *,
        profile_id: str,
    ) -> EnsembleProfileRecord | None:
        return self.profiles.get(profile_id)


def test_packet2_research_selects_the_canonical_three_role_roster(tmp_path: Path) -> None:
    registry_root = tmp_path / "registry"
    _export_candidate(
        tmp_path,
        registry_root=registry_root,
        model_version="m3-incumbent-generalist-v1",
        model_name="hist_gradient_boosting",
        model_family="AUTOGLUON",
        candidate_role=GENERALIST,
        mapping={1.0: 0.80, 2.0: 0.20, 3.0: 0.55, 4.0: 0.45, 5.0: 0.65},
        set_current=True,
    )
    _export_candidate(
        tmp_path,
        registry_root=registry_root,
        model_version="m20-autogluon-generalist-v2",
        model_name="autogluon_generalist",
        model_family="AUTOGLUON",
        candidate_role=GENERALIST,
        mapping={1.0: 0.85, 2.0: 0.20, 3.0: 0.75, 4.0: 0.20, 5.0: 0.10},
    )
    _export_candidate(
        tmp_path,
        registry_root=registry_root,
        model_version="m20-nhits-trend-v1",
        model_name="nhits_trend_specialist",
        model_family="NEURALFORECAST_NHITS",
        candidate_role=TREND_SPECIALIST,
        mapping={1.0: 0.90, 2.0: 0.20, 3.0: 0.20, 4.0: 0.20, 5.0: 0.20},
    )
    _export_candidate(
        tmp_path,
        registry_root=registry_root,
        model_version="m20-tft-trend-v1",
        model_name="tft_trend_specialist",
        model_family="NEURALFORECAST_TFT",
        candidate_role=TREND_SPECIALIST,
        mapping={1.0: 0.60, 2.0: 0.55, 3.0: 0.20, 4.0: 0.20, 5.0: 0.20},
    )
    _export_candidate(
        tmp_path,
        registry_root=registry_root,
        model_version="m20-patchtst-range-v1",
        model_name="patchtst_range_specialist",
        model_family="NEURALFORECAST_PATCHTST",
        candidate_role=RANGE_SPECIALIST,
        mapping={1.0: 0.20, 2.0: 0.20, 3.0: 0.85, 4.0: 0.20, 5.0: 0.20},
    )
    _export_candidate(
        tmp_path,
        registry_root=registry_root,
        model_version="m20-nbeatsx-range-v1",
        model_name="nbeatsx_range_specialist",
        model_family="NEURALFORECAST_NBEATSX",
        candidate_role=RANGE_SPECIALIST,
        mapping={1.0: 0.20, 2.0: 0.20, 3.0: 0.60, 4.0: 0.55, 5.0: 0.20},
    )

    candidates = load_registry_research_candidates(registry_root=registry_root)
    results = evaluate_registry_candidates(
        candidates=candidates,
        dataset=_dataset(),
        regime_labels_by_row_id=_regime_labels_by_row_id(),
        training_config=_training_config(),
        slippage_bps=5.0,
    )

    selection = select_runtime_roster(results)

    assert selection.generalist.candidate.model_version == "m20-autogluon-generalist-v2"
    assert selection.trend_specialist.candidate.model_version == "m20-nhits-trend-v1"
    assert selection.range_specialist.candidate.model_version == "m20-patchtst-range-v1"
    assert selection.evidence_summary_json["top_level_model_identity"]["model_name"] == (
        "dynamic_ensemble"
    )
    assert set(selection.evidence_summary_json["selected_roster"]) == {
        GENERALIST,
        TREND_SPECIALIST,
        RANGE_SPECIALIST,
    }


def test_packet2_profile_lifecycle_persists_draft_activation_and_rollback() -> None:
    selection = _selection_for_profile_tests()
    incumbent = EnsembleProfileRecord(
        profile_id="ens-prev",
        status="ACTIVE",
        approval_stage="ACTIVATED",
        execution_mode_scope="paper",
        symbol_scope="ALL",
        regime_scope="ALL",
        candidate_roster_json=[
            {
                "candidate_id": "generalist:ens-prev",
                "candidate_role": GENERALIST,
                "model_version": "m3-incumbent-generalist-v1",
                "scope_regimes": ["TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL"],
                "enabled": True,
                "expected_model_name": "hist_gradient_boosting",
            }
        ],
        created_at=datetime(2026, 3, 20, tzinfo=timezone.utc),
        approved_at=datetime(2026, 3, 20, tzinfo=timezone.utc),
        activated_at=datetime(2026, 3, 20, tzinfo=timezone.utc),
    )
    repository = _FakeEnsembleRepo(incumbent=incumbent)
    draft = build_draft_profile(
        selection=selection,
        profile_id="ens-p2-active",
        execution_mode_scope="paper",
        symbol_scope="ALL",
        regime_scope="ALL",
    )

    asyncio.run(save_draft_profile(repository, draft))
    asyncio.run(
        save_challenger_evidence(
            repository,
            challenger_run_id="ens-challenger-1",
            selection=selection,
            report_path="artifacts/ensemble/research/20260322T000000Z/research_report.json",
        )
    )
    active_profile, promotion = asyncio.run(
        activate_draft_profile(
            repository,
            draft_profile=draft,
            decision_id="ens-promote-1",
            summary_text="Packet 2 roster promoted after honest regime-conditioned evaluation.",
            metrics_delta_json={"net_pnl_after_fees_slippage": 0.12},
            safety_checks_json={"reliability_healthy": True},
            reason_codes=["ENSEMBLE_PACKET2_PROMOTION_APPROVED"],
        )
    )
    rollback = asyncio.run(
        rollback_active_profile(
            repository,
            active_profile_id="ens-p2-active",
            rollback_target_profile_id="ens-prev",
            decision_id="ens-rollback-1",
            summary_text="Rollback to previous stable Packet 1 profile.",
        )
    )

    assert draft.status == "DRAFT"
    assert len(draft.candidate_roster_json) == 3
    assert {item["candidate_role"] for item in draft.candidate_roster_json} == {
        GENERALIST,
        TREND_SPECIALIST,
        RANGE_SPECIALIST,
    }
    assert active_profile.status == "ACTIVE"
    assert active_profile.rollback_target_profile_id == "ens-prev"
    assert active_profile.evidence_summary_json["top_level_model_identity"]["model_name"] == (
        "dynamic_ensemble"
    )
    assert promotion.decision == "PROMOTE"
    assert repository.profiles["ens-prev"].status == "ACTIVE"
    assert repository.profiles["ens-p2-active"].status == "ROLLED_BACK"
    assert repository.challenger_runs[0].challenger_run_id == "ens-challenger-1"
    assert rollback.decision == "ROLLBACK"


def _selection_for_profile_tests() -> Any:
    candidates = [
        _candidate_result(
            model_version="m20-autogluon-generalist-v2",
            model_name="autogluon_generalist",
            model_family="AUTOGLUON",
            candidate_role=GENERALIST,
            net_all=0.12,
            net_primary=0.12,
        ),
        _candidate_result(
            model_version="m20-nhits-trend-v1",
            model_name="nhits_trend_specialist",
            model_family="NEURALFORECAST_NHITS",
            candidate_role=TREND_SPECIALIST,
            net_all=0.05,
            net_primary=0.09,
        ),
        _candidate_result(
            model_version="m20-patchtst-range-v1",
            model_name="patchtst_range_specialist",
            model_family="NEURALFORECAST_PATCHTST",
            candidate_role=RANGE_SPECIALIST,
            net_all=0.04,
            net_primary=0.08,
        ),
    ]
    return select_runtime_roster(candidates)


def _candidate_result(  # pylint: disable=too-many-arguments
    *,
    model_version: str,
    model_name: str,
    model_family: str,
    candidate_role: str,
    net_all: float,
    net_primary: float,
) -> EnsembleResearchResult:
    primary_slice = "ALL" if candidate_role == GENERALIST else (
        "TREND_COMBINED" if candidate_role == TREND_SPECIALIST else "RANGE"
    )
    return EnsembleResearchResult(
        candidate=EnsembleResearchCandidate(
            model_version=model_version,
            model_name=model_name,
            model_family=model_family,
            candidate_role=candidate_role,
            artifact_path=f"artifacts/registry/models/{model_version}/model.joblib",
            trained_at="2026-03-22T00:00:00Z",
            scope_regimes=["TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL"],
            entry_metadata={},
        ),
        metrics_by_slice={
            "ALL": EnsembleEvaluationSliceMetrics(
                slice_label="ALL",
                net_pnl_after_fees_slippage=net_all,
                max_drawdown=0.02,
                calmar_ratio=6.0,
                profit_factor=2.0,
                signal_precision=0.75,
                trade_count=4,
                blocked_trade_rate=0.0,
            ),
            "TREND_COMBINED": EnsembleEvaluationSliceMetrics(
                slice_label="TREND_COMBINED",
                net_pnl_after_fees_slippage=net_primary,
                max_drawdown=0.02,
                calmar_ratio=4.5,
                profit_factor=2.0,
                signal_precision=0.75,
                trade_count=3,
                blocked_trade_rate=0.0,
            ),
            "RANGE": EnsembleEvaluationSliceMetrics(
                slice_label="RANGE",
                net_pnl_after_fees_slippage=net_primary,
                max_drawdown=0.01,
                calmar_ratio=5.0,
                profit_factor=2.0,
                signal_precision=0.80,
                trade_count=2,
                blocked_trade_rate=0.0,
            ),
            "HIGH_VOL": EnsembleEvaluationSliceMetrics(
                slice_label="HIGH_VOL",
                net_pnl_after_fees_slippage=0.0,
                max_drawdown=0.0,
                calmar_ratio=None,
                profit_factor=None,
                signal_precision=0.0,
                trade_count=0,
                blocked_trade_rate=0.0,
            ),
        },
        primary_slice=primary_slice,
        primary_metric_value=net_primary,
    )


def _export_candidate(  # pylint: disable=too-many-arguments
    tmp_path: Path,
    *,
    registry_root: Path,
    model_version: str,
    model_name: str,
    model_family: str,
    candidate_role: str,
    mapping: dict[float, float],
    set_current: bool = False,
) -> None:
    artifact_path = tmp_path / f"{model_version}.joblib"
    joblib.dump(
        {
            "model_name": model_name,
            "trained_at": "2026-03-22T00:00:00Z",
            "feature_columns": [
                "symbol",
                "close_price",
                "realized_vol_12",
                "momentum_3",
                "macd_line_12_26",
            ],
            "expanded_feature_names": [
                "symbol=BTC/USD",
                "close_price",
                "realized_vol_12",
                "momentum_3",
                "macd_line_12_26",
            ],
            "model": LookupProbabilityModel(mapping),
        },
        artifact_path,
    )
    entry = export_external_model_to_registry(
        model_artifact_path=artifact_path,
        model_version=model_version,
        registry_root=registry_root,
        metadata={
            "model_family": model_family,
            "candidate_role": candidate_role,
            "scope_regimes": _scope_regimes_for_role(candidate_role),
        },
    )
    if set_current:
        write_current_registry_entry(
            {
                **entry,
                "activated_at": to_rfc3339(datetime(2026, 3, 22, tzinfo=timezone.utc)),
                "activation_reason": "PROMOTE",
                "metadata": {
                    "candidate_role": candidate_role,
                    "scope_regimes": _scope_regimes_for_role(candidate_role),
                },
            },
            registry_root=registry_root,
        )


def _scope_regimes_for_role(candidate_role: str) -> list[str]:
    if candidate_role == TREND_SPECIALIST:
        return ["TREND_UP", "TREND_DOWN"]
    if candidate_role == RANGE_SPECIALIST:
        return ["RANGE"]
    return ["TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL"]


def _training_config() -> TrainingConfig:
    return TrainingConfig(
        source_table="feature_ohlc",
        symbols=("BTC/USD",),
        time_column="as_of_time",
        interval_column="interval_begin",
        close_column="close_price",
        categorical_feature_columns=("symbol",),
        numeric_feature_columns=(
            "close_price",
            "realized_vol_12",
            "momentum_3",
            "macd_line_12_26",
        ),
        label_horizon_candles=3,
        purge_gap_candles=3,
        test_folds=2,
        first_train_fraction=0.5,
        test_fraction=0.2,
        round_trip_fee_bps=20.0,
        artifact_root="artifacts/training/m7",
        models=ModelHyperparameters(
            logistic_regression={"max_iter": 100},
            hist_gradient_boosting={"max_iter": 10},
        ),
    )


def _dataset() -> TrainingDataset:
    samples = (
        _sample(
            row_id="BTC/USD|2026-03-21T00:00:00Z",
            interval_begin=datetime(2026, 3, 21, 0, 0, tzinfo=timezone.utc),
            close_price=1.0,
            future_return_3=0.05,
            label=1,
            realized_vol_12=0.01,
            momentum_3=0.08,
            macd_line_12_26=0.05,
        ),
        _sample(
            row_id="BTC/USD|2026-03-21T00:05:00Z",
            interval_begin=datetime(2026, 3, 21, 0, 5, tzinfo=timezone.utc),
            close_price=2.0,
            future_return_3=-0.04,
            label=0,
            realized_vol_12=0.01,
            momentum_3=-0.08,
            macd_line_12_26=-0.05,
        ),
        _sample(
            row_id="BTC/USD|2026-03-21T00:10:00Z",
            interval_begin=datetime(2026, 3, 21, 0, 10, tzinfo=timezone.utc),
            close_price=3.0,
            future_return_3=0.03,
            label=1,
            realized_vol_12=0.01,
            momentum_3=0.00,
            macd_line_12_26=0.00,
        ),
        _sample(
            row_id="BTC/USD|2026-03-21T00:15:00Z",
            interval_begin=datetime(2026, 3, 21, 0, 15, tzinfo=timezone.utc),
            close_price=4.0,
            future_return_3=-0.03,
            label=0,
            realized_vol_12=0.01,
            momentum_3=0.00,
            macd_line_12_26=0.00,
        ),
        _sample(
            row_id="BTC/USD|2026-03-21T00:20:00Z",
            interval_begin=datetime(2026, 3, 21, 0, 20, tzinfo=timezone.utc),
            close_price=5.0,
            future_return_3=-0.05,
            label=0,
            realized_vol_12=0.50,
            momentum_3=0.01,
            macd_line_12_26=0.00,
        ),
    )
    return TrainingDataset(
        samples=samples,
        source_schema=(
            "symbol",
            "interval_begin",
            "as_of_time",
            "close_price",
            "realized_vol_12",
            "momentum_3",
            "macd_line_12_26",
        ),
        manifest={"loaded_rows": len(samples)},
        feature_columns=(
            "symbol",
            "close_price",
            "realized_vol_12",
            "momentum_3",
            "macd_line_12_26",
        ),
        categorical_feature_columns=("symbol",),
        numeric_feature_columns=(
            "close_price",
            "realized_vol_12",
            "momentum_3",
            "macd_line_12_26",
        ),
    )


def _sample(  # pylint: disable=too-many-arguments
    *,
    row_id: str,
    interval_begin: datetime,
    close_price: float,
    future_return_3: float,
    label: int,
    realized_vol_12: float,
    momentum_3: float,
    macd_line_12_26: float,
) -> DatasetSample:
    return DatasetSample(
        row_id=row_id,
        symbol="BTC/USD",
        interval_begin=interval_begin,
        as_of_time=interval_begin,
        close_price=close_price,
        future_close_price=close_price * (1.0 + future_return_3),
        future_return_3=future_return_3,
        label=label,
        persistence_prediction=0,
        features={
            "symbol": "BTC/USD",
            "close_price": close_price,
            "realized_vol_12": realized_vol_12,
            "momentum_3": momentum_3,
            "macd_line_12_26": macd_line_12_26,
        },
    )


def _regime_labels_by_row_id() -> dict[str, str]:
    return {
        "BTC/USD|2026-03-21T00:00:00Z": "TREND_UP",
        "BTC/USD|2026-03-21T00:05:00Z": "TREND_DOWN",
        "BTC/USD|2026-03-21T00:10:00Z": "RANGE",
        "BTC/USD|2026-03-21T00:15:00Z": "RANGE",
        "BTC/USD|2026-03-21T00:20:00Z": "HIGH_VOL",
    }
