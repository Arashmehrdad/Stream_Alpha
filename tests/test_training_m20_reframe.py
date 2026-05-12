"""Tests for M20 research reframe artifact."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_reframe import write_m20_reframe

# pylint: disable=missing-function-docstring


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_reframe_writes_policy_schema_and_research_status(tmp_path: Path) -> None:
    factory = tmp_path / "research_labels" / "vol_scaled" / "strategy_candidate_v2_refined_factory"
    _write_csv(
        factory / "candidate_decisions.csv",
        [{"candidate_decision": "V2_STRATEGY_CANDIDATE_ECONOMICS_NEGATIVE"}],
    )

    result = write_m20_reframe(source_run_dir=tmp_path)
    schema = json.loads(
        Path(result["output_files"]["policy_candidate_schema_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert result["recommendation"] == "DESIGN_RESEARCH_ONLY_DECISION_POLICY_EVALUATOR"
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]
    assert schema["schema_version"] == "m20_policy_candidate_v1"
