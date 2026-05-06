"""Focused tests for M20 research strategy selector design artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

from app.training.m20_strategy_selector_design import design_m20_strategy_selector

# pylint: disable=missing-function-docstring,too-many-arguments,too-many-positional-arguments


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(column)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_inputs(original: Path) -> None:
    comparison = (
        original
        / "research_labels"
        / "vol_scaled"
        / "confirmation_plan"
        / "confirmation_comparison"
    )
    audit = original / "research_labels" / "vol_scaled" / "model_member_audit"
    _write_csv(
        comparison / "confirmation_slice_comparison.csv",
        [
            _row("momentum", "flat", "STRONGLY_CONFIRMED", 2.0, 0.12, 1500, 150),
            _row("range", "low", "STRONGLY_CONFIRMED", 1.9, 0.10, 2000, 200),
            _row("symbol", "BTC/USD", "STRONGLY_CONFIRMED", 2.3, 0.13, 2500, 250),
            _row("macd", "positive", "STRONGLY_CONFIRMED", 1.8, 0.11, 1800, 180),
            _row("volume", "low", "STRONGLY_CONFIRMED", 2.1, 0.09, 1900, 190),
            _row("volume", "high", "CONFIRMED", 1.3, 0.03, 2000, 120),
            _row("month", "2026-04", "MISSING_IN_CONFIRMATION", 0.0, 0.0, 0, 0),
            _row("quarter", "2026Q2", "MISSING_IN_CONFIRMATION", 0.0, 0.0, 0, 0),
        ],
    )
    _write_csv(
        audit / "strategy_ensemble_candidate_ledger.csv",
        [
            {"model_name": "neuralforecast_patchtst"},
            {"model_name": "neuralforecast_nhits"},
        ],
    )


def _row(
    family: str,
    value: str,
    status: str,
    lift: float,
    pr_lift: float,
    rows: int,
    positives: int,
) -> dict[str, object]:
    return {
        "slice_family": family,
        "slice_value": value,
        "slice_id": f"{family}={value}",
        "original_classification": "ENABLE_CANDIDATE",
        "confirmation_status": status,
        "original_row_count": rows,
        "confirmation_row_count": rows,
        "original_positive_count": positives,
        "confirmation_positive_count": positives,
        "original_top_5_lift": lift,
        "confirmation_top_5_lift": lift,
        "original_pr_auc_lift_over_base": pr_lift,
        "confirmation_pr_auc_lift_over_base": pr_lift,
    }


def test_selector_design_writes_spec_and_evidence(tmp_path: Path) -> None:
    original = tmp_path / "original"
    confirmation = tmp_path / "confirmation"
    confirmation.mkdir()
    _write_inputs(original)

    result = design_m20_strategy_selector(
        original_run_dir=original,
        confirmation_run_dir=confirmation,
    )

    spec = result["selector_spec"]
    assert spec["selector_id"] == "fee_exceedance_gate_v0_research"
    assert spec["runtime_enabled"] is False
    assert spec["promotable"] is False
    assert spec["registry_write"] is False
    assert "momentum=flat" in spec["confirmed_conditions"]
    assert "month=2026-04" in spec["disable_gap_conditions"]
    assert (
        "neuralforecast_patchtst:CONDITIONAL_USEFULNESS_UNKNOWN"
        in spec["unknown_conditions"]
    )
    assert "RESEARCH_ONLY_STRATEGY_SELECTOR_DESIGN" in result["honesty_flags"]

    output_files = result["output_files"]
    assert Path(output_files["strategy_selector_evidence_table_csv"]).exists()
    assert Path(output_files["strategy_selector_unknowns_and_gaps_csv"]).exists()
    assert Path(output_files["strategy_selector_manual_next_steps_md"]).exists()


def test_weights_and_gap_actions_are_deterministic(tmp_path: Path) -> None:
    original = tmp_path / "original"
    confirmation = tmp_path / "confirmation"
    confirmation.mkdir()
    _write_inputs(original)

    result = design_m20_strategy_selector(
        original_run_dir=original,
        confirmation_run_dir=confirmation,
    )

    rows = {row["slice_id"]: row for row in result["evidence_rows"]}
    assert rows["momentum=flat"]["evidence_weight"] > rows["volume=high"]["evidence_weight"]
    assert rows["month=2026-04"]["proposed_selector_action"] == "DISABLE_GAP_UNTESTED"
    assert rows["quarter=2026Q2"]["proposed_selector_action"] == "DISABLE_GAP_UNTESTED"


def test_missing_comparison_file_is_handled_gracefully(tmp_path: Path) -> None:
    original = tmp_path / "original"
    confirmation = tmp_path / "confirmation"
    confirmation.mkdir()
    _write_csv(
        original
        / "research_labels"
        / "vol_scaled"
        / "conditional_usefulness_full_test"
        / "conditional_usefulness_by_slice.csv",
        [
            {
                "slice_family": "symbol",
                "slice_value": "BTC/USD",
                "classification": "ENABLE_CANDIDATE",
            }
        ],
    )

    result = design_m20_strategy_selector(
        original_run_dir=original,
        confirmation_run_dir=confirmation,
    )

    assert result["selector_id"] == "fee_exceedance_gate_v0_research"
    assert Path(result["output_files"]["strategy_selector_candidate_spec_json"]).exists()
