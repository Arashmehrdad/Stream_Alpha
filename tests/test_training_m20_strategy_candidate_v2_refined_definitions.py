"""Tests for M20 refined v2 strategy definitions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.m20_strategy_candidate_v2_refined_definitions import (
    build_m20_strategy_candidate_v2_refined_definitions,
)

# pylint: disable=missing-function-docstring


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def test_refined_definitions_write_predicate_specs(tmp_path: Path) -> None:
    result = build_m20_strategy_candidate_v2_refined_definitions(source_run_dir=tmp_path)
    rows = _read_csv(Path(result["output_files"]["candidate_definition_specs_csv"]))

    assert rows
    assert all(row["predicate_spec_json"] for row in rows)
    assert result["recommendation"] == "RUN_REFINED_V2_CANDIDATE_FACTORY"


def test_refined_definitions_do_not_use_forbidden_feature_inputs(tmp_path: Path) -> None:
    result = build_m20_strategy_candidate_v2_refined_definitions(source_run_dir=tmp_path)
    rows = _read_csv(Path(result["output_files"]["candidate_definition_specs_csv"]))

    forbidden = (
        "fee_exceedance_label",
        "triple_barrier_label",
        "future_return",
        "gross_value_proxy",
        "net_value_proxy",
        "economic_outcome",
    )
    for row in rows:
        payload = json.loads(row["predicate_spec_json"])
        serialized = json.dumps(payload).lower()
        assert not any(token in row["required_features"].lower() for token in forbidden)
        assert not any(token in serialized for token in forbidden)
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
