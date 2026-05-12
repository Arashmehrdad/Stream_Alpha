"""Tests for M20 input-redesign recovery tools."""

from __future__ import annotations

import csv
from pathlib import Path

from app.training.m20_decision_policy_evaluator import evaluate_m20_decision_policies
from app.training.m20_input_failure_analysis import analyze_m20_input_failures
from app.training.m20_input_redesign_decision import write_m20_input_redesign_decision
from app.training.m20_input_redesign_plan import plan_m20_input_redesign
from app.training.m20_redesigned_research_inputs import build_m20_redesigned_research_inputs
from app.training.m20_research_input_catalogue import build_m20_research_input_catalogue

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source(base: Path) -> tuple[Path, Path]:
    source = base / "source"
    prediction = base / "prediction"
    research = source / "research_labels" / "vol_scaled"
    features = []
    outcomes = []
    oof = []
    for index in range(20):
        timestamp = f"2024-01-01T00:{index:02d}:00Z"
        close = 100.0 + index
        features.append(
            {
                "fold_index": 1,
                "symbol": "BTC",
                "interval_begin": timestamp,
                "close_price": close,
                "high_price": close + 1.0,
                "low_price": close - 1.0,
                "realized_vol_12": 0.01,
                "regime_label": "TREND_UP",
                "adx_14": 25,
                "volume_zscore_12": 0.1,
                "close_zscore_12": 0.1,
                "momentum_3": 0.02,
            }
        )
        outcomes.append(
            {
                "fold_index": 1,
                "symbol": "BTC",
                "interval_begin": timestamp,
                "fee_bps": 20,
                "slippage_bps": 0,
                "net_value_proxy": -0.01,
            }
        )
        oof.append(
            {
                "model_name": "model",
                "fold_index": 1,
                "symbol": "BTC",
                "interval_begin": timestamp,
                "prob_up": 0.7,
                "confidence": 0.7,
                "regime_label": "TREND_UP",
                "long_trade_taken": 1,
                "y_true": 1,
            }
        )
    _write_csv(research / "research_feature_enrichment" / "research_features.csv", features)
    _write_csv(research / "economic_outcome_artifacts" / "economic_outcomes.csv", outcomes)
    _write_csv(prediction / "oof_predictions.csv", oof)
    _write_csv(
        research / "strategy_candidate_v2_refined_factory" / "candidate_metrics.csv",
        [{"classification": "V2_STRATEGY_CANDIDATE_ECONOMICS_NEGATIVE"}],
    )
    _write_csv(
        research / "decision_policy_eval" / "policy_metrics.csv",
        [{"classification": "POLICY_ECONOMICS_NEGATIVE"}],
    )
    _write_csv(
        research / "trading_aware_policy_eval" / "policy_metrics.csv",
        [{"policy_name": "p", "classification": "POLICY_ECONOMICS_NEGATIVE"}],
    )
    _write_csv(
        research / "trading_aware_labels" / "blocked_labels.csv",
        [
            {
                "label_name": "forward_return_6_candles",
                "blocker": "MISSING_SAFE_FORWARD_RETURN_6_SOURCE",
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
                "candidate_name": "c",
            }
        ],
    )
    (research / "trading_aware_policy_validation_audit").mkdir(parents=True)
    (research / "trading_aware_policy_validation_audit" / "recommendation.json").write_text(
        '{"recommendation":"DESIGN_TRADING_AWARE_RESEARCH_LABELS"}',
        encoding="utf-8",
    )
    (research / "shadow_adaptation_observer_plan").mkdir(parents=True)
    (research / "shadow_adaptation_observer_plan" / "recommendation.json").write_text(
        '{"recommendation":"PAUSE_M20_POLICY_ROUTE_AND_REDESIGN_INPUTS"}',
        encoding="utf-8",
    )
    return source, prediction


def test_input_failure_analyzer_classifies_failed_routes(tmp_path: Path) -> None:
    source, _prediction = _source(tmp_path)

    result = analyze_m20_input_failures(source_run_dir=source)
    families = {row["failure_family"] for row in result["failure_attribution"]}

    assert "CANDIDATE_DEFINITIONS" in families
    assert result["recommendation"] == "BUILD_RESEARCH_INPUT_CATALOGUE_AND_REDESIGN_PLAN"


def test_catalogue_marks_blocked_horizons_safe_computable(tmp_path: Path) -> None:
    source, prediction = _source(tmp_path)

    result = build_m20_research_input_catalogue(
        source_run_dir=source,
        prediction_run_dir=prediction,
    )

    assert result["safe_computable_blocked_labels"] == 2
    assert result["recommendation"] == "BUILD_SAFE_INPUT_REDESIGN_PLAN"


def test_redesign_plan_and_builder_compute_multi_horizon_labels(tmp_path: Path) -> None:
    source, prediction = _source(tmp_path)
    build_m20_research_input_catalogue(source_run_dir=source, prediction_run_dir=prediction)
    plan = plan_m20_input_redesign(source_run_dir=source)
    result = build_m20_redesigned_research_inputs(source_run_dir=source)

    assert plan["ready_spec_count"] >= 1
    assert result["row_count"] == 20
    assert result["blocked_input_count"] == 0
    labels_path = Path(result["output_files"]["multi_horizon_labels_csv"])
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        labels = list(csv.DictReader(handle))
    assert labels[0]["future_return_6"] != ""
    assert labels[-1]["future_return_6"] == ""
    assert result["recommendation"] == "RE_RUN_DECISION_POLICY_EVALUATOR_WITH_REDESIGNED_INPUTS"


def test_policy_evaluator_joins_redesigned_label_for_calibration(tmp_path: Path) -> None:
    source, prediction = _source(tmp_path)
    build_m20_redesigned_research_inputs(source_run_dir=source)

    result = evaluate_m20_decision_policies(
        source_run_dir=source,
        prediction_run_dir=prediction,
        research_input_dir=(
            source
            / "research_labels"
            / "vol_scaled"
            / "m20_redesigned_research_inputs"
        ),
        label_column="fee_plus_slippage_exceedance_6",
        output_name="m20_redesigned_policy_eval",
    )
    label_columns = {row["label_column"] for row in result["calibration_metrics"]}

    assert "redesigned_label" in label_columns
    assert "NO_RUNTIME_EFFECT" in result["overall_status"]


def test_input_redesign_decision_pauses_when_policy_still_negative(tmp_path: Path) -> None:
    source, prediction = _source(tmp_path)
    build_m20_redesigned_research_inputs(source_run_dir=source)
    evaluate_m20_decision_policies(
        source_run_dir=source,
        prediction_run_dir=prediction,
        research_input_dir=(
            source
            / "research_labels"
            / "vol_scaled"
            / "m20_redesigned_research_inputs"
        ),
        label_column="fee_plus_slippage_exceedance_6",
        output_name="m20_redesigned_policy_eval",
    )

    result = write_m20_input_redesign_decision(source_run_dir=source)

    assert result["final_decision"] in {
        "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY",
        "M20_READY_FOR_CONSERVATIVE_VALIDATION_AUDIT",
    }
    assert "NOT_PROMOTABLE" in result["overall_status"]
