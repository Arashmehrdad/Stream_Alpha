"""Tests for DU12 microstructure research readiness audit."""

from __future__ import annotations

import json
from pathlib import Path

from app.training.microstructure_research_readiness import (
    write_microstructure_research_readiness,
)


def test_readiness_blocks_missing_artifacts(tmp_path: Path) -> None:
    """Missing upstream artifacts should block readiness clearly."""
    result = write_microstructure_research_readiness(
        repo_root=tmp_path,
        output_dir=tmp_path / "readiness",
    )

    assert result["readiness_status"] == (
        "MICROSTRUCTURE_RESEARCH_READINESS_BLOCKED_MISSING_ARTIFACTS"
    )
    assert result["alpha_research_reopen_ready"] is False


def test_readiness_marks_fixture_chain_not_ready(tmp_path: Path) -> None:
    """A complete fixture chain is still not enough to reopen alpha research."""
    _write_artifacts(tmp_path)

    result = write_microstructure_research_readiness(
        repo_root=tmp_path,
        output_dir=tmp_path / "readiness",
    )

    assert result["readiness_status"] == "MICROSTRUCTURE_RESEARCH_NOT_READY_FIXTURE_ONLY"
    assert result["recommendation"] == (
        "COLLECT_MORE_MICROSTRUCTURE_DATA_BEFORE_REOPENING_ALPHA_RESEARCH"
    )


def test_readiness_preserves_non_claims(tmp_path: Path) -> None:
    """Readiness output should preserve non-runtime and non-profit claims."""
    _write_artifacts(tmp_path)
    output_dir = tmp_path / "readiness"

    write_microstructure_research_readiness(repo_root=tmp_path, output_dir=output_dir)

    report = json.loads((output_dir / "microstructure_research_readiness.json").read_text())
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    assert "NO_RUNTIME_EFFECT" in report["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in recommendation["honesty_flags"]
    assert recommendation["runtime_ready"] is False
    assert recommendation["promotable"] is False


def _write_artifacts(root: Path) -> None:
    artifacts = {
        "microstructure_storage_contracts/microstructure_storage_contracts.json": "a",
        "microstructure_capture_service_dry_run/microstructure_capture_service_dry_run.json": "b",
        "microstructure_capture_smoke/microstructure_capture_smoke.json": "c",
        "microstructure_replay/microstructure_replay.json": "d",
        "microstructure_features/microstructure_features.json": "e",
    }
    for relative, schema_version in artifacts.items():
        path = root / "artifacts/research_data_upgrade" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": schema_version}), encoding="utf-8")
