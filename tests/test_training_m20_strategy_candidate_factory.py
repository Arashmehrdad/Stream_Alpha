"""Focused tests for M20 strategy candidate factory."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_strategy_candidate_factory import build_m20_strategy_candidates

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


def _feature_rows() -> list[dict[str, object]]:
    rows = []
    for index in range(8):
        rows.append(
            {
                "symbol": "BTC/USD" if index < 4 else "ETH/USD",
                "interval_begin": f"2025-01-01T0{index}:00:00Z",
                "fold_index": "4",
                "row_id": f"row-{index}",
                "high_price": 102 + index,
                "low_price": 99,
                "close_price": 100 + index,
                "volume": 10 + index,
                "log_return_1": -0.01 if index % 2 else 0.01,
                "momentum_3": 0.02 if index % 2 else -0.02,
                "realized_vol_12": 0.01 + index * 0.001,
                "rsi_14": [25, 35, 45, 55, 65, 75, 50, 20][index],
                "macd_line_12_26": [-2, -1, 0.2, 1, 2, -0.5, 0.0, 3][index],
            }
        )
    return rows


def _write_sources(base: Path, *, omit_macd: bool = False, economics: bool = True) -> None:
    rows = _feature_rows()
    if omit_macd:
        for row in rows:
            row.pop("macd_line_12_26")
    feature_path = base / "training_frame" / "m20_training_frame_features.csv"
    _write_csv(feature_path, rows)
    feature_columns = [
        column for column in rows[0]
        if column not in ("symbol", "interval_begin", "fold_index", "row_id")
    ]
    (base / "training_frame" / "m20_training_frame_feature_columns.json").write_text(
        json.dumps({"feature_columns": feature_columns}),
        encoding="utf-8",
    )
    label_rows = [
        {
            "symbol": row["symbol"],
            "interval_begin": row["interval_begin"],
            "fold_index": row["fold_index"],
            "scenario_name": "current_fee",
            "label": "1" if index in (0, 2, 5) else "0",
        }
        for index, row in enumerate(rows)
    ]
    _write_csv(
        base / "research_labels" / "vol_scaled" / "fee_exceedance_labels_vol_scaled.csv",
        label_rows,
    )
    if economics:
        outcome_rows = [
            {
                "symbol": row["symbol"],
                "interval_begin": row["interval_begin"],
                "fold_index": row["fold_index"],
                "gross_value_proxy": 0.01 if index in (0, 2, 5) else -0.01,
                "net_value_proxy": 0.008 if index in (0, 2, 5) else -0.012,
                "fee_exceedance_label": "1" if index in (0, 2, 5) else "0",
                "triple_barrier_label": "1" if index in (0, 2, 5) else "0",
            }
            for index, row in enumerate(rows)
        ]
        _write_csv(
            base
            / "research_labels"
            / "vol_scaled"
            / "economic_outcome_artifacts"
            / "economic_outcomes.csv",
            outcome_rows,
        )


def test_multiple_strategy_families_are_evaluated(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = build_m20_strategy_candidates(source_run_dir=tmp_path)

    assert "macd_momentum" in result["strategy_families_evaluated"]
    assert "rsi_mean_reversion" in result["strategy_families_evaluated"]
    assert "volume_context" in result["strategy_families_evaluated"]
    assert result["candidate_count"] >= 10


def test_missing_required_features_are_blocked(tmp_path: Path) -> None:
    _write_sources(tmp_path, omit_macd=True)

    result = build_m20_strategy_candidates(source_run_dir=tmp_path)
    audit = _read_csv(Path(result["output_files"]["feature_family_audit_csv"]))
    macd = next(row for row in audit if row["strategy_family"] == "macd_momentum")

    assert macd["status"] == "BLOCKED_REQUIRED_FEATURES_MISSING"


def test_candidate_metrics_are_deterministic_and_labels_join(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = build_m20_strategy_candidates(source_run_dir=tmp_path)
    metrics = _read_csv(Path(result["output_files"]["candidate_metrics_csv"]))
    candidate = next(row for row in metrics if row["candidate_name"] == "return_positive")

    assert int(candidate["selected_rows"]) == 4
    assert float(candidate["selected_positive_rate"]) == 0.5


def test_economic_outcomes_join_correctly(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = build_m20_strategy_candidates(source_run_dir=tmp_path)
    metrics = _read_csv(Path(result["output_files"]["candidate_metrics_csv"]))
    candidate = next(row for row in metrics if row["candidate_name"] == "return_positive")

    assert float(candidate["mean_net_proxy"]) == -0.002
    assert float(candidate["cumulative_net_proxy"]) == -0.008


def test_negative_economics_do_not_promote_or_claim_profit(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = build_m20_strategy_candidates(source_run_dir=tmp_path)
    decisions = _read_csv(Path(result["output_files"]["candidate_decisions_csv"]))

    assert any(
        row["candidate_decision"] == "STRATEGY_CANDIDATE_ECONOMICS_NEGATIVE"
        for row in decisions
    )
    assert all(row["promotion_status"] == "NOT_PROMOTABLE" for row in decisions)
    assert all(row["profitability_status"] == "NO_PROFIT_CLAIM" for row in decisions)


def test_missing_economics_emits_blocker(tmp_path: Path) -> None:
    _write_sources(tmp_path, economics=False)

    result = build_m20_strategy_candidates(source_run_dir=tmp_path)

    assert result["economics_available"] is False
    assert "ECONOMIC_OUTCOMES_NOT_AVAILABLE" in result["blockers"]


def test_outputs_include_research_only_statuses(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = build_m20_strategy_candidates(source_run_dir=tmp_path)
    report = json.loads(
        Path(result["output_files"]["strategy_candidate_report_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert "NO_RUNTIME_EFFECT" in report["overall_status"]
    assert "NOT_PROMOTABLE" in report["overall_status"]
    assert "NO_PROFIT_CLAIM" in report["overall_status"]


def test_no_runtime_imports_or_registry_writes() -> None:
    source = Path("app/training/m20_strategy_candidate_factory.py").read_text(
        encoding="utf-8"
    )

    assert "app.inference" not in source
    assert "app.trading" not in source
    assert "app.training.registry" not in source
