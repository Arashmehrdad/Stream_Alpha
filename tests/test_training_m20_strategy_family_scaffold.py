"""Focused tests for M20 strategy-family scaffold."""

from __future__ import annotations

import json
from pathlib import Path

from app.training.m20_strategy_family_scaffold import design_m20_strategy_families

# pylint: disable=missing-function-docstring


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_strategy_family_scaffold_writes_expected_families(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_json(
        base / "research_labels" / "vol_scaled" / "m20_decision_memo" / "decision_memo.json",
        {
            "decision": "PAUSE_RANK_GATE_AS_STANDALONE_PATH",
            "decision_statuses": ["RESEARCH_SIGNAL_CONFIRMED", "NOT_PROMOTABLE"],
        },
    )
    _write_json(
        base / "training_frame" / "m20_training_frame_feature_columns.json",
        ["log_return_1", "macd_line_12_26", "realized_vol_12", "volume", "rsi_14"],
    )

    result = design_m20_strategy_families(base_run_dir=base)

    assert result["family_count"] == 4
    assert result["rank_gate_usage"] == "OPTIONAL_FILTER_ONLY"
    assert result["promotion_status"] == "NOT_PROMOTABLE"
    assert Path(result["output_files"]["strategy_families_csv"]).exists()
    assert Path(result["output_files"]["feature_requirements_csv"]).exists()


def test_strategy_family_scaffold_preserves_no_runtime_flag(tmp_path: Path) -> None:
    result = design_m20_strategy_families(base_run_dir=tmp_path / "run")

    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NO_TRADING_LOGIC" in result["honesty_flags"]
    assert Path(result["output_files"]["next_experiments_csv"]).exists()
