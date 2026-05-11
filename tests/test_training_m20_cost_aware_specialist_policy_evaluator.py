"""Focused tests for M20 cost-aware specialist policy evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.training.m20_cost_aware_specialist_policy_evaluator import (
    analyze_m20_cost_aware_specialist_policy,
)

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
        strong = model_name == "model_strong"
        probability = 0.99 - index * 0.001 if strong else 0.40 + index * 0.001
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


def _label_rows(model_name: str, *, with_net_proxy: bool) -> list[dict[str, object]]:
    rows = []
    for index in range(100):
        label = "1" if index < 5 else "0"
        row: dict[str, object] = {
            "model_name": model_name,
            "symbol": "BTC/USD" if index < 50 else "ETH/USD",
            "interval_begin": f"2025-01-{(index // 24) + 1:02d}T{index % 24:02d}:00:00Z",
            "label": label,
        }
        if with_net_proxy:
            row["long_only_net_value_proxy"] = 0.10 if label == "1" else -0.02
        rows.append(row)
    return rows


def _write_sources(base: Path, labels: Path, *, with_net_proxy: bool = False) -> None:
    prediction_dir = base / "research_labels" / "vol_scaled" / "specialist_predictions"
    for model_name in ("model_strong", "model_weak"):
        _write_csv(
            prediction_dir / f"predictions_{model_name}_score_only_confirmation.csv",
            _prediction_rows(model_name),
        )
    label_rows = _label_rows(
        "model_strong",
        with_net_proxy=with_net_proxy,
    ) + _label_rows("model_weak", with_net_proxy=with_net_proxy)
    _write_csv(
        labels
        / "research_labels"
        / "vol_scaled"
        / "fee_exceedance_labels_vol_scaled.csv",
        label_rows,
    )


def _write_economic_outcomes(base: Path) -> Path:
    outcome_dir = base / "research_labels" / "vol_scaled" / "economic_outcome_artifacts"
    rows = []
    for index in range(100):
        label = index < 5
        rows.append(
            {
                "symbol": "BTC/USD" if index < 50 else "ETH/USD",
                "interval_begin": f"2025-01-{(index // 24) + 1:02d}T{index % 24:02d}:00:00Z",
                "fold_index": "",
                "gross_value_proxy": 0.11 if label else -0.01,
                "net_value_proxy": 0.10 if label else -0.02,
            }
        )
    _write_csv(outcome_dir / "economic_outcomes.csv", rows)
    return outcome_dir


def test_evaluator_supports_multiple_models_in_one_run(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)

    result = analyze_m20_cost_aware_specialist_policy(
        prediction_run_dir=prediction_run,
        label_source_run_dir=label_run,
        prediction_source="score_only_confirmation",
    )

    assert result["manifest"]["models"] == ["model_strong", "model_weak"]
    assert result["manifest"]["joined_rows"] == 200
    assert Path(result["output_files"]["policy_candidates_csv"]).exists()


def test_topk_policy_metrics_are_deterministic_and_join_labels(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)

    result = analyze_m20_cost_aware_specialist_policy(
        prediction_run_dir=prediction_run,
        label_source_run_dir=label_run,
        prediction_source="score_only_confirmation",
        models=["model_strong"],
    )
    topk = _read_csv(Path(result["output_files"]["topk_policy_metrics_csv"]))
    top5 = next(row for row in topk if row["policy_name"] == "TOP_5_PERCENT")

    assert int(top5["selected_rows"]) == 5
    assert float(top5["precision"]) == 1.0
    assert float(top5["lift_vs_base"]) == 20.0
    assert result["manifest"]["joined_rows_by_model"]["model_strong"] == 100


def test_missing_prediction_files_fail_clearly(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)

    with pytest.raises(ValueError, match="Missing specialist prediction file"):
        analyze_m20_cost_aware_specialist_policy(
            prediction_run_dir=prediction_run,
            label_source_run_dir=label_run,
            prediction_source="score_only_confirmation",
            models=["missing_model"],
        )


def test_missing_label_file_fails_clearly(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)
    (
        label_run
        / "research_labels"
        / "vol_scaled"
        / "fee_exceedance_labels_vol_scaled.csv"
    ).unlink()

    with pytest.raises(ValueError, match="Missing label file"):
        analyze_m20_cost_aware_specialist_policy(
            prediction_run_dir=prediction_run,
            label_source_run_dir=label_run,
            prediction_source="score_only_confirmation",
        )


def test_missing_net_proxy_emits_blockers(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)

    result = analyze_m20_cost_aware_specialist_policy(
        prediction_run_dir=prediction_run,
        label_source_run_dir=label_run,
        prediction_source="score_only_confirmation",
    )
    economics = json.loads(
        Path(result["output_files"]["economics_availability_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert economics["economics_available"] is False
    assert "NET_PROXY_NOT_AVAILABLE" in economics["evidence_blockers"]
    assert "ECONOMIC_POLICY_EVALUATION_REQUIRED" in economics["evidence_blockers"]


def test_missing_economic_outcome_dir_keeps_blocker_behavior(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)

    result = analyze_m20_cost_aware_specialist_policy(
        prediction_run_dir=prediction_run,
        label_source_run_dir=label_run,
        prediction_source="score_only_confirmation",
        economic_outcome_dir=tmp_path / "missing_outcomes",
    )

    assert result["economics_available"] is False
    assert result["recommendation"] == "ADD_SAFE_NET_PROXY_OR_ECONOMIC_OUTCOME_ARTIFACTS"


def test_binary_label_only_does_not_claim_profit(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)

    result = analyze_m20_cost_aware_specialist_policy(
        prediction_run_dir=prediction_run,
        label_source_run_dir=label_run,
        prediction_source="score_only_confirmation",
        models=["model_strong"],
    )

    assert result["profitability_status"] == "NO_PROFIT_CLAIM"
    assert result["recommendation"] == "ADD_SAFE_NET_PROXY_OR_ECONOMIC_OUTCOME_ARTIFACTS"


def test_safe_net_proxy_computes_economic_metrics(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run, with_net_proxy=True)

    result = analyze_m20_cost_aware_specialist_policy(
        prediction_run_dir=prediction_run,
        label_source_run_dir=label_run,
        prediction_source="score_only_confirmation",
        models=["model_strong"],
    )
    topk = _read_csv(Path(result["output_files"]["topk_policy_metrics_csv"]))
    top5 = next(row for row in topk if row["policy_name"] == "TOP_5_PERCENT")

    assert float(top5["mean_net_proxy"]) == pytest.approx(0.1)
    assert float(top5["cumulative_net_proxy"]) == pytest.approx(0.5)
    assert float(top5["max_drawdown_proxy"]) == pytest.approx(0.0)


def test_economic_outcome_dir_is_accepted_and_joined(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)
    outcome_dir = _write_economic_outcomes(label_run)

    result = analyze_m20_cost_aware_specialist_policy(
        prediction_run_dir=prediction_run,
        label_source_run_dir=label_run,
        prediction_source="score_only_confirmation",
        economic_outcome_dir=outcome_dir,
        models=["model_strong"],
    )
    economics = json.loads(
        Path(result["output_files"]["economics_availability_json"]).read_text(
            encoding="utf-8"
        )
    )
    topk = _read_csv(Path(result["output_files"]["topk_policy_metrics_csv"]))
    top5 = next(row for row in topk if row["policy_name"] == "TOP_5_PERCENT")

    assert economics["economics_available"] is True
    assert economics["safe_source"] == "economic_outcome_artifacts"
    assert float(top5["mean_net_proxy"]) == pytest.approx(0.1)
    assert float(top5["mean_gross_proxy"]) == pytest.approx(0.11)
    assert float(top5["best_5_net_proxy"]) == pytest.approx(0.1)
    assert float(top5["worst_5_net_proxy"]) == pytest.approx(0.1)


def test_default_economic_outcome_dir_is_discovered(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)
    _write_economic_outcomes(label_run)

    result = analyze_m20_cost_aware_specialist_policy(
        prediction_run_dir=prediction_run,
        label_source_run_dir=label_run,
        prediction_source="score_only_confirmation",
        models=["model_strong"],
    )

    assert result["economics_available"] is True
    assert result["recommendation"] == "PLAN_STRICT_OUT_OF_SAMPLE_POLICY_CONFIRMATION"


def test_strong_signal_without_economics_is_unknown_not_promotable(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)

    result = analyze_m20_cost_aware_specialist_policy(
        prediction_run_dir=prediction_run,
        label_source_run_dir=label_run,
        prediction_source="score_only_confirmation",
        models=["model_strong"],
    )
    decisions = _read_csv(Path(result["output_files"]["candidate_decisions_csv"]))

    assert decisions[0]["candidate_decision"] == "SIGNAL_CONFIRMED_ECONOMICS_UNKNOWN"
    assert decisions[0]["promotion_status"] == "NOT_PROMOTABLE"


def test_outputs_include_research_only_statuses(tmp_path: Path) -> None:
    prediction_run = tmp_path / "prediction"
    label_run = tmp_path / "labels"
    _write_sources(prediction_run, label_run)

    result = analyze_m20_cost_aware_specialist_policy(
        prediction_run_dir=prediction_run,
        label_source_run_dir=label_run,
        prediction_source="score_only_confirmation",
    )
    report = json.loads(
        Path(result["output_files"]["cost_aware_policy_report_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert "NO_RUNTIME_EFFECT" in report["overall_status"]
    assert "NOT_PROMOTABLE" in report["overall_status"]
    assert "NO_PROFIT_CLAIM" in report["overall_status"]
