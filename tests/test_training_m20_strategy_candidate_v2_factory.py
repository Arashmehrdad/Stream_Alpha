"""Focused tests for M20 v2 strategy candidate factory."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_strategy_candidate_v2_factory import (
    build_m20_strategy_candidates_v2,
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


def _write_sources(base: Path) -> None:
    rows = [_feature_row(index) for index in range(6)]
    _write_csv(base / "training_frame" / "m20_training_frame_features.csv", rows)
    feature_columns = [
        column for column in rows[0]
        if column not in ("symbol", "interval_begin", "fold_index", "row_id")
    ]
    (base / "training_frame" / "m20_training_frame_feature_columns.json").write_text(
        json.dumps({"feature_columns": feature_columns}),
        encoding="utf-8",
    )
    _write_csv(
        base
        / "research_labels"
        / "vol_scaled"
        / "strategy_candidate_redesign_plan"
        / "candidate_definition_specs.csv",
        _definitions(),
    )
    _write_csv(
        base / "research_labels" / "vol_scaled" / "fee_exceedance_labels_vol_scaled.csv",
        [
            {
                "symbol": row["symbol"],
                "interval_begin": row["interval_begin"],
                "fold_index": row["fold_index"],
                "scenario_name": "current_fee",
                "label": "1" if index in (0, 2, 4) else "0",
            }
            for index, row in enumerate(rows)
        ],
    )
    _write_csv(
        base
        / "research_labels"
        / "vol_scaled"
        / "economic_outcome_artifacts"
        / "economic_outcomes.csv",
        [
            {
                "symbol": row["symbol"],
                "interval_begin": row["interval_begin"],
                "fold_index": row["fold_index"],
                "fee_exceedance_label": "1" if index in (0, 2, 4) else "0",
                "triple_barrier_label": "1" if index in (0, 2, 4) else "0",
                "gross_value_proxy": 0.01 if index in (0, 2, 4) else -0.01,
                "net_value_proxy": 0.008 if index in (0, 2, 4) else -0.012,
            }
            for index, row in enumerate(rows)
        ],
    )


def _feature_row(index: int) -> dict[str, object]:
    return {
        "symbol": "BTC/USD" if index < 3 else "ETH/USD",
        "interval_begin": f"2025-01-01T0{index}:00:00Z",
        "fold_index": "4",
        "row_id": f"row-{index}",
        "open_price": 100 + index,
        "high_price": 102 + index,
        "low_price": 99,
        "close_price": 100 + index,
        "vwap": 100 + index,
        "volume": 100 + index,
        "volume_zscore_12": [1.0, -1.0, 0.8, 0.2, 2.5, -2.5][index],
        "log_return_1": [0.01, -0.01, 0.02, -0.02, 0.01, -0.01][index],
        "momentum_3": [0.02, -0.02, 0.03, 0.01, 0.04, -0.04][index],
        "return_std_12": [0.002, 0.003, 0.002, 0.02, 0.02, 0.02][index],
        "realized_vol_12": [0.01, 0.02, 0.015, 0.03, 0.035, 0.04][index],
        "rsi_14": [55, 30, 60, 50, 65, 80][index],
        "macd_line_12_26": [2.0, -1.0, 3.0, 0.5, 4.0, -2.0][index],
        "close_zscore_12": [0.5, 1.0, 1.4, 2.5, 0.5, -2.2][index],
    }


def _definitions() -> list[dict[str, object]]:
    return [
        _definition(
            "multi_condition",
            "momentum_volume_confirmed",
            "momentum_3|volume_zscore_12|realized_vol_12",
            "",
            "READY_FOR_V2_FACTORY",
        ),
        _definition(
            "lower_turnover",
            "high_agreement_low_turnover_setup",
            "macd_line_12_26|momentum_3|rsi_14|volume_zscore_12",
            "",
            "READY_FOR_V2_FACTORY",
        ),
        _definition(
            "regime_conditioned",
            "regime_conditioned_momentum",
            "regime_label|momentum_3|realized_vol_12",
            "regime_label",
            "BLOCKED_MISSING_FEATURES",
        ),
    ]


def _definition(
    family: str,
    name: str,
    required: str,
    missing: str,
    status: str,
) -> dict[str, object]:
    return {
        "redesign_family": family,
        "candidate_name": name,
        "candidate_version": "v2_design",
        "required_features": required,
        "rationale": "test definition",
        "uses_economic_outcome_as_feature": False,
        "evaluates_candidate_now": False,
        "missing_features": missing,
        "definition_status": status,
    }


def test_ready_v2_definitions_generate_candidate_rows(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = build_m20_strategy_candidates_v2(source_run_dir=tmp_path)
    candidates = _read_csv(Path(result["output_files"]["strategy_candidates_v2_csv"]))
    names = {row["candidate_name"] for row in candidates}

    assert "momentum_volume_confirmed" in names
    assert "high_agreement_low_turnover_setup" in names
    assert result["candidate_event_rows"] > 0


def test_blocked_definitions_remain_blocked(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = build_m20_strategy_candidates_v2(source_run_dir=tmp_path)
    blocked = _read_csv(Path(result["output_files"]["blocked_definitions_csv"]))

    assert blocked[0]["candidate_name"] == "regime_conditioned_momentum"
    assert blocked[0]["missing_features"] == "regime_label"


def test_label_and_economic_joins_are_correct(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = build_m20_strategy_candidates_v2(source_run_dir=tmp_path)
    metrics = _read_csv(Path(result["output_files"]["candidate_metrics_csv"]))
    candidate = next(
        row for row in metrics
        if row["candidate_name"] == "momentum_volume_confirmed"
    )

    assert float(candidate["selected_positive_rate"]) > 0.0
    assert candidate["mean_net_proxy"] != ""


def test_low_turnover_definition_is_narrower_than_broad_fixture(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = build_m20_strategy_candidates_v2(source_run_dir=tmp_path)
    metrics = _read_csv(Path(result["output_files"]["candidate_metrics_csv"]))
    low_turnover = next(
        row for row in metrics
        if row["candidate_name"] == "high_agreement_low_turnover_setup"
    )

    assert float(low_turnover["coverage"]) < 0.6


def test_outputs_include_research_only_statuses(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = build_m20_strategy_candidates_v2(source_run_dir=tmp_path)
    report = json.loads(
        Path(result["output_files"]["strategy_candidate_v2_report_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert "NO_RUNTIME_EFFECT" in report["overall_status"]
    assert "NOT_PROMOTABLE" in report["overall_status"]
    assert "NO_PROFIT_CLAIM" in report["overall_status"]


def test_no_economic_outcomes_used_as_predicate_inputs() -> None:
    source = Path("app/training/m20_strategy_candidate_v2_factory.py").read_text(
        encoding="utf-8"
    )
    predicate_source = source.split("def _candidate_row", maxsplit=1)[0]

    assert "net_value_proxy" not in predicate_source
    assert "gross_value_proxy" not in predicate_source
    assert "future_return" not in predicate_source


def test_no_runtime_imports_or_registry_writes() -> None:
    source = Path("app/training/m20_strategy_candidate_v2_factory.py").read_text(
        encoding="utf-8"
    )

    assert "app.inference" not in source
    assert "app.trading" not in source
    assert "app.training.registry" not in source
