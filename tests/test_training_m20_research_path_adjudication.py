"""Focused tests for M20 research path adjudication."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_research_path_adjudication import (
    write_m20_research_path_adjudication,
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


def _write_sources(base: Path) -> None:
    vol = base / "research_labels" / "vol_scaled"
    _write_json(
        vol / "m20_decision_memo" / "decision_memo.json",
        {"decision": "PAUSE_RANK_GATE_AS_STANDALONE_PATH"},
    )
    _write_json(
        vol / "strategy_family_adjudication" / "recommendation.json",
        {"recommendation": "TEST_VOLATILITY_EXPANSION_NEXT"},
    )
    _write_json(
        vol / "volatility_expansion_deep_dive" / "recommendation.json",
        {"recommendation": "TEST_VOLATILITY_EXPANSION_COMBO_NEXT"},
    )
    _write_json(
        vol / "volatility_combo_economics" / "recommendation.json",
        {"recommendation": "TRY_VOLATILITY_AS_OPTIONAL_GATE_FILTER"},
    )
    _write_json(
        vol / "abstention_hold_research" / "recommendation.json",
        {
            "recommendation": "KEEP_ABSTENTION_AS_RESEARCH_FILTER",
            "oracle_diagnostic_rules": ["HOLD_SELECTED_NEGATIVE_NET_PROXY"],
        },
    )
    _write_csv(
        vol / "model_member_audit" / "candidate_next_actions.csv",
        [
            {
                "candidate_id": "x",
                "model_name": "neuralforecast_patchtst",
                "next_required_action": "export row-level probabilities before conditional claims",
            }
        ],
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_research_path_adjudication_writes_outputs(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_sources(base)

    result = write_m20_research_path_adjudication(base_run_dir=base)

    assert result["recommended_next_action"] == "PLAN_ROW_LEVEL_SPECIALIST_PREDICTION_EXPORT"
    assert "ORACLE_HOLD_RULES_NOT_IMPLEMENTABLE" in result["decision_statuses"]
    assert Path(result["output_files"]["adjudication_md"]).exists()


def test_research_path_adjudication_preserves_path_decisions(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_sources(base)

    result = write_m20_research_path_adjudication(base_run_dir=base)
    decisions = _read_csv(Path(result["output_files"]["path_decisions_csv"]))
    decision_by_path = {row["path"]: row["decision"] for row in decisions}

    assert decision_by_path["rank_gate_standalone"] == "PAUSE"
    assert decision_by_path["row_level_specialist_predictions"] == "PLAN_NEXT"


def test_research_path_adjudication_handles_missing_sources(tmp_path: Path) -> None:
    result = write_m20_research_path_adjudication(base_run_dir=tmp_path / "run")

    assert result["promotion_status"] == "NOT_PROMOTABLE"
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]
    assert Path(result["output_files"]["evidence_rollup_csv"]).exists()
