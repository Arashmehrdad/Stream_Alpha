"""Focused tests for generic M20 specialist edge evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.training.m20_specialist_edge_evaluator import analyze_m20_specialist_edge

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _prediction_rows(model_name: str) -> list[dict[str, object]]:
    rows = []
    for index in range(100):
        is_strong = model_name == "model_strong"
        probability = 0.99 - index * 0.001 if is_strong else 0.40 + index * 0.001
        rows.append(
            {
                "symbol": "BTC/USD" if index < 50 else "ETH/USD",
                "interval_begin": f"2025-01-{(index // 24) + 1:02d}T{index % 24:02d}:00:00Z",
                "model_name": model_name,
                "y_pred": "0",
                "prob_up": probability,
            }
        )
    return rows


def _label_rows(model_name: str) -> list[dict[str, object]]:
    rows = []
    for index in range(100):
        is_top_five = index < 5
        rows.append(
            {
                "model_name": model_name,
                "symbol": "BTC/USD" if index < 50 else "ETH/USD",
                "interval_begin": f"2025-01-{(index // 24) + 1:02d}T{index % 24:02d}:00:00Z",
                "label": "1" if is_top_five else "0",
            }
        )
    return rows


def _write_sources(base: Path, labels: Path, *, with_net_proxy: bool = False) -> None:
    prediction_dir = base / "research_labels" / "vol_scaled" / "specialist_predictions"
    for model_name in ("model_strong", "model_weak"):
        _write_csv(
            prediction_dir / f"predictions_{model_name}_score_only_confirmation.csv",
            _prediction_rows(model_name),
        )
    label_rows = _label_rows("model_strong") + _label_rows("model_weak")
    if with_net_proxy:
        for row in label_rows:
            row["long_only_net_value_proxy"] = 0.01 if row["label"] == "1" else -0.01
    _write_csv(
        labels
        / "research_labels"
        / "vol_scaled"
        / "fee_exceedance_labels_vol_scaled.csv",
        label_rows,
    )


def test_evaluator_supports_multiple_models_in_one_run(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)

    result = analyze_m20_specialist_edge(
        prediction_run_dir=prediction_run,
        label_source_run_dir=label_run,
        prediction_source="score_only_confirmation",
    )

    assert result["manifest"]["models"] == ["model_strong", "model_weak"]
    assert result["joined_rows"] == 200
    assert Path(result["output_files"]["specialist_edge_report_json"]).exists()


def test_topk_metrics_are_deterministic_and_join_labels(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)

    result = analyze_m20_specialist_edge(
        prediction_run_dir=prediction_run,
        label_source_run_dir=label_run,
        prediction_source="score_only_confirmation",
        models=["model_strong"],
    )
    topk = _read_csv(Path(result["output_files"]["topk_policy_metrics_csv"]))
    top5 = next(row for row in topk if row["top_k_fraction"] == "0.05")

    assert int(top5["selected_rows"]) == 5
    assert float(top5["precision"]) == 1.0
    assert float(top5["lift_vs_base"]) == 20.0
    assert result["manifest"]["joined_rows_by_model"]["model_strong"] == 100


def test_missing_prediction_file_fails_clearly(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)

    with pytest.raises(ValueError, match="Missing specialist prediction file"):
        analyze_m20_specialist_edge(
            prediction_run_dir=prediction_run,
            label_source_run_dir=label_run,
            prediction_source="score_only_confirmation",
            models=["missing_model"],
        )


def test_missing_label_file_fails_clearly(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)
    label_path = (
        label_run
        / "research_labels"
        / "vol_scaled"
        / "fee_exceedance_labels_vol_scaled.csv"
    )
    label_path.unlink()

    with pytest.raises(ValueError, match="Missing label file"):
        analyze_m20_specialist_edge(
            prediction_run_dir=prediction_run,
            label_source_run_dir=label_run,
            prediction_source="score_only_confirmation",
        )


def test_no_net_proxy_emits_status(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)

    result = analyze_m20_specialist_edge(
        prediction_run_dir=prediction_run,
        label_source_run_dir=label_run,
        prediction_source="score_only_confirmation",
        models=["model_strong"],
    )
    model_metrics = _read_csv(Path(result["output_files"]["model_edge_metrics_csv"]))
    recommendation = json.loads(
        Path(result["output_files"]["recommendation_json"]).read_text(encoding="utf-8")
    )

    assert model_metrics[0]["net_proxy_status"] == "NET_PROXY_NOT_AVAILABLE"
    assert "NET_PROXY_NOT_AVAILABLE" in recommendation["evidence_blockers"]
    assert "ECONOMIC_POLICY_EVALUATION_REQUIRED" in recommendation["evidence_blockers"]
    assert recommendation["next_required_action"] == (
        "DESIGN_COST_AWARE_SPECIALIST_POLICY_EVALUATOR"
    )


def test_strong_topk_weak_global_is_research_only_candidate(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)

    result = analyze_m20_specialist_edge(
        prediction_run_dir=prediction_run,
        label_source_run_dir=label_run,
        prediction_source="score_only_confirmation",
        models=["model_strong"],
    )
    decisions = _read_csv(Path(result["output_files"]["candidate_decisions_csv"]))

    assert decisions[0]["candidate_decision"] == (
        "CONFIRMED_SELECTIVE_EDGE_RESEARCH_CANDIDATE"
    )
    assert decisions[0]["runtime_status"] == "NO_RUNTIME_EFFECT"
    assert decisions[0]["promotion_status"] == "NOT_PROMOTABLE"
    assert decisions[0]["profitability_status"] == "NO_PROFIT_CLAIM"


def test_output_includes_research_only_statuses(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)

    result = analyze_m20_specialist_edge(
        prediction_run_dir=prediction_run,
        label_source_run_dir=label_run,
        prediction_source="score_only_confirmation",
    )
    report = json.loads(
        Path(result["output_files"]["specialist_edge_report_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert "NO_RUNTIME_EFFECT" in report["overall_status"]
    assert "NOT_PROMOTABLE" in report["overall_status"]
    assert "NO_PROFIT_CLAIM" in report["overall_status"]
    assert "NOT_BACKTEST" in report["evidence_blockers"]
    assert "NOT_RUNTIME_READY" in report["evidence_blockers"]
