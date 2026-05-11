"""Focused tests for M20 specialist confirmation adjudication."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.training.m20_specialist_confirmation_adjudication import (
    write_m20_specialist_confirmation_adjudication,
)

# pylint: disable=missing-function-docstring


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_confirmation_sources(run_dir: Path) -> None:
    base = (
        run_dir
        / "research_labels"
        / "vol_scaled"
        / "specialist_conditional_usefulness"
    )
    _write_json(
        base / "report.json",
        {
            "joined_rows": 624970,
            "best_candidate": "neuralforecast_patchtst",
            "recommendation": "RUN_SPECIALIST_CONFIRMATION_EXPORT",
            "prediction_source": "score_only_confirmation",
        },
    )
    _write_json(
        base / "recommendation.json",
        {
            "recommendation": "RUN_SPECIALIST_CONFIRMATION_EXPORT",
            "best_candidate": "neuralforecast_patchtst",
        },
    )
    _write_csv(
        base / "comparison.csv",
        [
            {
                "model_name": "neuralforecast_nhits",
                "rows": 312485,
                "base_positive_rate": 0.1354593020,
                "pr_auc": 0.1375560011,
                "roc_auc": 0.4835002977,
                "top5_precision": 0.1609703021,
                "top5_lift": 1.1883296287,
                "enable_slice_count": 3,
                "best_slice": "month=2025-02",
                "recommendation_basis": "baseline",
            },
            {
                "model_name": "neuralforecast_patchtst",
                "rows": 312485,
                "base_positive_rate": 0.1354593020,
                "pr_auc": 0.1645954481,
                "roc_auc": 0.4890033256,
                "top5_precision": 0.2668330773,
                "top5_lift": 1.9698394520,
                "enable_slice_count": 18,
                "best_slice": "month=2024-06",
                "recommendation_basis": "strong topk",
            },
        ],
    )
    _write_csv(
        base / "topk_metrics.csv",
        [
            {
                "model_name": "neuralforecast_nhits",
                "top_k_fraction": 0.01,
                "selected_rows": 3124,
                "coverage": 0.01,
                "precision": 0.1235595391,
                "base_positive_rate": 0.1354593020,
                "lift": 0.9121524855,
                "recall": 0.0091190437,
                "false_positives": 2738,
                "avg_probability": 0.51,
            },
            {
                "model_name": "neuralforecast_nhits",
                "top_k_fraction": 0.02,
                "selected_rows": 6249,
                "coverage": 0.02,
                "precision": 0.1488238118,
                "base_positive_rate": 0.1354593020,
                "lift": 1.0986607015,
                "recall": 0.0219707529,
                "false_positives": 5319,
                "avg_probability": 0.51,
            },
            {
                "model_name": "neuralforecast_nhits",
                "top_k_fraction": 0.05,
                "selected_rows": 15624,
                "coverage": 0.05,
                "precision": 0.1609703021,
                "base_positive_rate": 0.1354593020,
                "lift": 1.1883296287,
                "recall": 0.0594155307,
                "false_positives": 13109,
                "avg_probability": 0.51,
            },
            {
                "model_name": "neuralforecast_nhits",
                "top_k_fraction": 0.1,
                "selected_rows": 31248,
                "coverage": 0.1,
                "precision": 0.1625064004,
                "base_positive_rate": 0.1354593020,
                "lift": 1.1996695535,
                "recall": 0.1199650358,
                "false_positives": 26170,
                "avg_probability": 0.51,
            },
            {
                "model_name": "neuralforecast_patchtst",
                "top_k_fraction": 0.01,
                "selected_rows": 3124,
                "coverage": 0.01,
                "precision": 0.3517925736,
                "base_positive_rate": 0.1354593020,
                "lift": 2.5970351855,
                "recall": 0.0259632876,
                "false_positives": 2025,
                "avg_probability": 0.51,
            },
            {
                "model_name": "neuralforecast_patchtst",
                "top_k_fraction": 0.02,
                "selected_rows": 6249,
                "coverage": 0.02,
                "precision": 0.3198911826,
                "base_positive_rate": 0.1354593020,
                "lift": 2.3615298304,
                "recall": 0.0472253065,
                "false_positives": 4250,
                "avg_probability": 0.51,
            },
            {
                "model_name": "neuralforecast_patchtst",
                "top_k_fraction": 0.05,
                "selected_rows": 15624,
                "coverage": 0.05,
                "precision": 0.2668330773,
                "base_positive_rate": 0.1354593020,
                "lift": 1.9698394520,
                "recall": 0.0984903967,
                "false_positives": 11455,
                "avg_probability": 0.51,
            },
            {
                "model_name": "neuralforecast_patchtst",
                "top_k_fraction": 0.1,
                "selected_rows": 31248,
                "coverage": 0.1,
                "precision": 0.2215821813,
                "base_positive_rate": 0.1354593020,
                "lift": 1.6357841648,
                "recall": 0.1635757991,
                "false_positives": 24324,
                "avg_probability": 0.51,
            },
        ],
    )
    _write_csv(
        base / "model_metrics.csv",
        [
            {
                "model_name": "neuralforecast_nhits",
                "row_count": 312485,
                "positive_count": 42329,
                "positive_rate": 0.1354593020,
                "accuracy_diagnostic": 0.2938509048,
                "balanced_accuracy": 0.4775359137,
                "precision": 0.1286111759,
                "recall": 0.7294762456,
                "f1": 0.2186695560,
                "false_positives": 209210,
                "true_positives": 30878,
                "roc_auc": 0.4835002977,
                "pr_auc": 0.1375560011,
            },
            {
                "model_name": "neuralforecast_patchtst",
                "row_count": 312485,
                "positive_count": 42329,
                "positive_rate": 0.1354593020,
                "accuracy_diagnostic": 0.2938509048,
                "balanced_accuracy": 0.4775359137,
                "precision": 0.1286111759,
                "recall": 0.7294762456,
                "f1": 0.2186695560,
                "false_positives": 209210,
                "true_positives": 30878,
                "roc_auc": 0.4890033256,
                "pr_auc": 0.1645954481,
            },
        ],
    )


def test_adjudication_writes_expected_outputs(tmp_path: Path) -> None:
    confirmation = tmp_path / "confirm"
    original = tmp_path / "original"
    _write_confirmation_sources(confirmation)
    _write_confirmation_sources(original)

    result = write_m20_specialist_confirmation_adjudication(
        confirmation_run_dir=confirmation,
        original_run_dir=original,
    )

    output_files = result["output_files"]
    assert Path(output_files["manifest_json"]).exists()
    assert Path(output_files["specialist_confirmation_adjudication_json"]).exists()
    assert Path(output_files["specialist_confirmation_adjudication_md"]).exists()
    assert Path(output_files["candidate_decisions_csv"]).exists()
    assert Path(output_files["evidence_metrics_csv"]).exists()
    assert Path(output_files["next_actions_csv"]).exists()


def test_patchtst_becomes_confirmed_selective_research_candidate(tmp_path: Path) -> None:
    confirmation = tmp_path / "confirm"
    _write_confirmation_sources(confirmation)

    result = write_m20_specialist_confirmation_adjudication(
        confirmation_run_dir=confirmation
    )

    assert (
        result["patchtst_decision"]
        == "CONFIRMED_SELECTIVE_RANK_SLICE_RESEARCH_CANDIDATE"
    )
    assert result["runtime_status"] == "NO_RUNTIME_EFFECT"
    assert result["promotion_status"] == "NOT_PROMOTABLE"
    assert result["profitability_status"] == "NO_PROFIT_CLAIM"
    assert result["required_next_action"] == (
        "DESIGN_COST_AWARE_SPECIALIST_POLICY_EVALUATOR"
    )


def test_nhits_remains_secondary_watchlist_candidate(tmp_path: Path) -> None:
    confirmation = tmp_path / "confirm"
    _write_confirmation_sources(confirmation)

    result = write_m20_specialist_confirmation_adjudication(
        confirmation_run_dir=confirmation
    )
    decision_rows = _read_csv(Path(result["output_files"]["candidate_decisions_csv"]))
    decision_by_model = {row["model_name"]: row["candidate_decision"] for row in decision_rows}

    assert (
        decision_by_model["neuralforecast_nhits"]
        == "SECONDARY_WATCHLIST_OR_WEAKER_CANDIDATE"
    )


def test_missing_confirmation_artifacts_raise_clear_error(tmp_path: Path) -> None:
    confirmation = tmp_path / "confirm"
    with pytest.raises(
        ValueError,
        match="Missing confirmation specialist conditional usefulness artifacts",
    ):
        write_m20_specialist_confirmation_adjudication(
            confirmation_run_dir=confirmation
        )
