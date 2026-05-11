"""Focused tests for M20 strategy candidate redesign planning."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_strategy_candidate_redesign_plan import (
    plan_m20_strategy_candidate_redesign,
)

# pylint: disable=missing-function-docstring


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_sources(base: Path, *, include_regime: bool = False) -> None:
    columns = [
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "vwap",
        "volume",
        "volume_zscore_12",
        "log_return_1",
        "momentum_3",
        "return_std_12",
        "realized_vol_12",
        "rsi_14",
        "macd_line_12_26",
        "close_zscore_12",
    ]
    if include_regime:
        columns.append("regime_label")
    _write_json(
        base / "training_frame" / "m20_training_frame_feature_columns.json",
        {"feature_columns": columns},
    )
    vol = base / "research_labels" / "vol_scaled"
    _write_json(
        vol / "research_candidate_comparator" / "recommendation.json",
        {"evidence_blockers": ["NO_POSITIVE_PROXY_RESEARCH_CANDIDATE"]},
    )
    _write_json(
        vol / "m20_research_dashboard" / "recommendation.json",
        {"evidence_blockers": ["NO_RUNTIME_OR_PROMOTION_DECISION"]},
    )


def test_plan_writes_v2_candidate_definitions_without_evaluation(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = plan_m20_strategy_candidate_redesign(source_run_dir=tmp_path)
    specs = _read_csv(Path(result["output_files"]["candidate_definition_specs_csv"]))

    assert result["ready_definition_count"] > 0
    assert any(row["redesign_family"] == "multi_condition" for row in specs)
    assert all(row["evaluates_candidate_now"] == "False" for row in specs)
    assert result["recommendation"] == "BUILD_GENERIC_V2_STRATEGY_CANDIDATES"


def test_missing_feature_families_are_reported(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = plan_m20_strategy_candidate_redesign(source_run_dir=tmp_path)
    missing = _read_csv(Path(result["output_files"]["missing_feature_families_csv"]))
    blocked = _read_csv(Path(result["output_files"]["blocked_definitions_csv"]))

    assert any(row["feature_family"] == "regime" for row in missing)
    assert any(row["candidate_name"] == "regime_conditioned_momentum" for row in blocked)


def test_regime_definition_unblocks_when_feature_exists(tmp_path: Path) -> None:
    _write_sources(tmp_path, include_regime=True)

    result = plan_m20_strategy_candidate_redesign(source_run_dir=tmp_path)
    specs = _read_csv(Path(result["output_files"]["candidate_definition_specs_csv"]))
    regime = next(row for row in specs if row["candidate_name"] == "regime_conditioned_momentum")

    assert regime["definition_status"] == "READY_FOR_V2_FACTORY"


def test_contract_forbids_economic_outcomes_as_features(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = plan_m20_strategy_candidate_redesign(source_run_dir=tmp_path)
    contract = json.loads(
        Path(result["output_files"]["candidate_contract_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert "net_value_proxy" in contract["forbidden_feature_inputs"]
    assert "future_return" in contract["forbidden_feature_inputs"]


def test_outputs_include_research_only_statuses(tmp_path: Path) -> None:
    _write_sources(tmp_path)

    result = plan_m20_strategy_candidate_redesign(source_run_dir=tmp_path)
    plan = json.loads(
        Path(result["output_files"]["strategy_candidate_redesign_plan_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert "NO_RUNTIME_EFFECT" in plan["honesty_flags"]
    assert "NOT_PROMOTABLE" in plan["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in plan["honesty_flags"]
    assert plan["promotion_status"] == "NOT_PROMOTABLE"


def test_no_runtime_imports_or_registry_writes() -> None:
    source = Path("app/training/m20_strategy_candidate_redesign_plan.py").read_text(
        encoding="utf-8"
    )

    assert "app.inference" not in source
    assert "app.trading" not in source
    assert "app.training.registry" not in source
