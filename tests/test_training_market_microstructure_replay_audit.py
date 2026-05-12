"""Tests for research-only microstructure coverage/gap/replay audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.market_microstructure_replay_audit import write_microstructure_replay_audit


def test_replay_audit_writes_expected_artifacts(tmp_path: Path) -> None:
    """The replay audit should persist its deterministic artifact contract."""
    _write_source_artifacts(tmp_path)
    output_dir = tmp_path / "replay_audit"

    result = write_microstructure_replay_audit(repo_root=tmp_path, output_dir=output_dir)

    assert result["audit_status"] == "MICROSTRUCTURE_REPLAY_AUDIT_COMPLETE_DATA_BLOCKED"
    for filename in (
        "manifest.json",
        "microstructure_replay_audit.json",
        "microstructure_replay_audit.md",
        "source_artifact_audit.csv",
        "coverage_gap_audit.csv",
        "replay_determinism_audit.csv",
        "data_readiness_audit.csv",
        "blocked_decisions.csv",
        "next_actions.csv",
        "recommendation.json",
    ):
        assert (output_dir / filename).exists()


def test_replay_audit_reports_no_stored_rows_without_inventing_metrics(
    tmp_path: Path,
) -> None:
    """Coverage/gap metrics must remain blocked without stored replay rows."""
    _write_source_artifacts(tmp_path)
    output_dir = tmp_path / "replay_audit"

    result = write_microstructure_replay_audit(repo_root=tmp_path, output_dir=output_dir)

    assert result["stored_replay_rows_available"] is False
    assert result["coverage_gap_metric_status"] == (
        "BLOCKED_NO_STORED_MICROSTRUCTURE_REPLAY_ROWS"
    )
    coverage_rows = _read_csv(output_dir / "coverage_gap_audit.csv")
    assert {row["gap_metrics_available"] for row in coverage_rows} == {"False"}
    assert {row["row_count_available"] for row in coverage_rows} == {"False"}


def test_replay_audit_blocks_when_contract_artifacts_are_missing(tmp_path: Path) -> None:
    """Missing source artifacts should produce a clear restore recommendation."""
    output_dir = tmp_path / "replay_audit"

    result = write_microstructure_replay_audit(repo_root=tmp_path, output_dir=output_dir)

    assert result["audit_status"] == (
        "MICROSTRUCTURE_REPLAY_AUDIT_BLOCKED_MISSING_CONTRACT_ARTIFACTS"
    )
    assert result["recommendation"] == "RESTORE_MICROSTRUCTURE_CONTRACT_ARTIFACTS"
    source_rows = _read_csv(output_dir / "source_artifact_audit.csv")
    assert {row["status"] for row in source_rows} == {"MISSING"}


def test_replay_audit_preserves_research_only_non_claims(tmp_path: Path) -> None:
    """Outputs should keep M20 paused and avoid runtime/promotion/profit claims."""
    _write_source_artifacts(tmp_path)
    output_dir = tmp_path / "replay_audit"

    write_microstructure_replay_audit(repo_root=tmp_path, output_dir=output_dir)

    report = json.loads((output_dir / "microstructure_replay_audit.json").read_text())
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    for flag in (
        "RESEARCH_ONLY",
        "NO_RUNTIME_EFFECT",
        "NOT_BACKTEST",
        "NOT_RUNTIME_READY",
        "NOT_PROMOTABLE",
        "NO_PROFIT_CLAIM",
    ):
        assert flag in report["honesty_flags"]
        assert flag in recommendation["honesty_flags"]
    assert report["m20_research_decision"] == "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"
    assert recommendation["runtime_ready"] is False
    assert recommendation["promotable"] is False
    assert recommendation["profitability_claim"] is False


def test_replay_audit_recommends_isolated_capture_plan_next(tmp_path: Path) -> None:
    """After DU5, the next safe step should be an isolated capture plan."""
    _write_source_artifacts(tmp_path)
    output_dir = tmp_path / "replay_audit"

    result = write_microstructure_replay_audit(repo_root=tmp_path, output_dir=output_dir)

    assert result["recommendation"] == (
        "PLAN_OPTIONAL_ISOLATED_MICROSTRUCTURE_CAPTURE_SERVICE"
    )
    assert result["next_required_action"] == (
        "DESIGN_ISOLATED_RESEARCH_MICROSTRUCTURE_CAPTURE_PLAN"
    )
    blocked = _read_csv(output_dir / "blocked_decisions.csv")
    assert "research_capture_scope" in {row["decision"] for row in blocked}


def _write_source_artifacts(root: Path) -> None:
    _write_json(
        root
        / "artifacts/research_data_upgrade/microstructure_schema_contracts/"
        "microstructure_schema_contracts.json",
        {"schema_version": "microstructure_schema_contracts_v1"},
    )
    _write_json(
        root
        / "artifacts/research_data_upgrade/book_payload_normalizer_contract/"
        "book_payload_normalizer_contract.json",
        {"schema_version": "book_payload_normalizer_contract_v1"},
    )
    _write_json(
        root
        / "artifacts/research_data_upgrade/microstructure_feature_derivation/"
        "microstructure_feature_derivation.json",
        {"schema_version": "microstructure_feature_derivation_v1"},
    )


def _write_json(path: Path, payload: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
