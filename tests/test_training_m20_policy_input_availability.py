"""Tests for M20 policy input availability audit."""

from __future__ import annotations

import csv
from pathlib import Path

from app.training.m20_policy_input_availability import audit_m20_policy_inputs

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixtures(base: Path) -> tuple[Path, Path]:
    source = base / "source"
    prediction = base / "prediction"
    research = source / "research_labels" / "vol_scaled"
    _write_csv(
        prediction / "oof_predictions.csv",
        [
            {
                "model_name": "m",
                "fold_index": 1,
                "symbol": "BTC",
                "interval_begin": "2024-01-01T00:00:00Z",
                "prob_up": 0.7,
                "confidence": 0.2,
                "regime_label": "TREND_UP",
                "long_trade_taken": 1,
                "y_true": 1,
            }
        ],
    )
    _write_csv(
        research / "economic_outcome_artifacts" / "economic_outcomes.csv",
        [
            {
                "fold_index": 1,
                "symbol": "BTC",
                "interval_begin": "2024-01-01T00:00:00Z",
                "net_value_proxy": 0.01,
            }
        ],
    )
    _write_csv(
        research / "strategy_candidate_v2_refined_factory" / "strategy_candidates_v2.csv",
        [
            {
                "fold_index": 1,
                "symbol": "BTC",
                "interval_begin": "2024-01-01T00:00:00Z",
                "candidate_name": "candidate",
            }
        ],
    )
    _write_csv(
        research / "research_feature_enrichment" / "research_features.csv",
        [
            {
                "fold_index": 1,
                "symbol": "BTC",
                "interval_begin": "2024-01-01T00:00:00Z",
                "regime_label": "TREND_UP",
            }
        ],
    )
    _write_csv(
        research / "fee_exceedance_labels_vol_scaled.csv",
        [
            {
                "fold_index": 1,
                "symbol": "BTC",
                "interval_begin": "2024-01-01T00:00:00Z",
                "label": 1,
            }
        ],
    )
    _write_csv(
        research / "triple_barrier_labels_vol_scaled.csv",
        [
            {
                "fold_index": 1,
                "symbol": "BTC",
                "interval_begin": "2024-01-01T00:00:00Z",
                "label": 1,
            }
        ],
    )
    return source, prediction


def test_policy_input_audit_detects_ready_policy_families(tmp_path: Path) -> None:
    source, prediction = _fixtures(tmp_path)

    result = audit_m20_policy_inputs(
        source_run_dir=source,
        prediction_run_dir=prediction,
    )
    statuses = {
        row["policy_family"]: row["ready"]
        for row in result["policy_family_readiness"]
    }

    assert statuses["OOF_PROBABILITY_THRESHOLD"] == "True"
    assert statuses["CANDIDATE_EVENT_POLICY"] == "True"
    assert result["recommendation"] == "BUILD_RESEARCH_ONLY_DECISION_POLICY_EVALUATOR"


def test_policy_input_audit_blocks_missing_required_inputs(tmp_path: Path) -> None:
    source, prediction = _fixtures(tmp_path)
    (
        source
        / "research_labels"
        / "vol_scaled"
        / "economic_outcome_artifacts"
        / "economic_outcomes.csv"
    ).unlink()

    result = audit_m20_policy_inputs(
        source_run_dir=source,
        prediction_run_dir=prediction,
    )

    assert result["recommendation"] == "BLOCKED_MISSING_POLICY_INPUTS"
    assert result["missing_inputs"][0]["input_name"] == "economic_outcomes"


def test_policy_input_audit_marks_outcomes_as_non_selection_inputs(tmp_path: Path) -> None:
    source, prediction = _fixtures(tmp_path)

    result = audit_m20_policy_inputs(
        source_run_dir=source,
        prediction_run_dir=prediction,
    )
    blocked_columns = {row["column_name"] for row in result["leakage_audit"]}

    assert "net_value_proxy" in blocked_columns
    assert "y_true" in blocked_columns
    assert "NO_RUNTIME_EFFECT" in result["overall_status"]
