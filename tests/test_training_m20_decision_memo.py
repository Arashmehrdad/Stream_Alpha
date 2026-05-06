"""Focused tests for M20 research decision memo."""

from __future__ import annotations

import json
from pathlib import Path

from app.training.m20_decision_memo import write_m20_decision_memo

# pylint: disable=missing-function-docstring


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_sources(base: Path) -> None:
    vol = base / "research_labels" / "vol_scaled"
    _write_json(
        vol / "rank_gate_evidence_packet" / "rank_gate_evidence_packet.json",
        {"evidence_status": "RESEARCH_CONFIRMED_RANK_GATE"},
    )
    _write_json(
        vol / "rank_gate_economics" / "recommendation.json",
        {"recommendation": "KEEP_RESEARCH_ONLY_RANK_GATE_ECONOMICS_CANDIDATE"},
    )
    _write_json(
        vol / "rank_gate_net_diagnostics" / "recommendation.json",
        {"recommendation": "DIAGNOSE_TAIL_AND_CONDITION_CONCENTRATION"},
    )
    _write_json(
        vol / "rank_gate_tail_analysis" / "recommendation.json",
        {"recommendation": "REVIEW_TAIL_CONCENTRATION_BEFORE_ANY_POLICY_OR_STRATEGY_STEP"},
    )
    _write_json(
        vol / "rank_gate_tail_filter" / "recommendation.json",
        {"recommendation": "NO_STABLE_TAIL_FILTER_FOUND"},
    )
    _write_json(
        vol / "model_member_audit" / "model_member_audit_report.json",
        {"recommendation": "EXPORT_AUTOGLUON_MEMBER_PREDICTIONS"},
    )
    _write_json(
        vol / "strategy_selector_design" / "strategy_selector_candidate_spec.json",
        {"evidence_status": "RESEARCH_ONLY"},
    )


def test_decision_memo_writes_outputs(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _write_sources(base)

    result = write_m20_decision_memo(base_run_dir=base)

    assert result["decision"] == "PAUSE_RANK_GATE_AS_STANDALONE_PATH"
    assert "RESEARCH_SIGNAL_CONFIRMED" in result["decision_statuses"]
    assert result["promotion_status"] == "NOT_PROMOTABLE"
    assert Path(result["output_files"]["decision_md"]).exists()
    assert Path(result["output_files"]["evidence_table_csv"]).exists()


def test_decision_memo_handles_missing_optional_sources(tmp_path: Path) -> None:
    result = write_m20_decision_memo(base_run_dir=tmp_path / "run")

    assert result["runtime_status"] == "NOT_RUNTIME_READY"
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]
    assert Path(result["output_files"]["next_forks_csv"]).exists()
