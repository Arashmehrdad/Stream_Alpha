"""Tests for DU9 dry-run capture smoke harness."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.training.microstructure_capture_smoke import write_microstructure_capture_smoke


def test_capture_smoke_writes_dry_run_artifacts(tmp_path: Path) -> None:
    """Smoke harness should write artifacts without executing capture."""
    output_dir = tmp_path / "smoke"

    result = write_microstructure_capture_smoke(repo_root=tmp_path, output_dir=output_dir)

    assert result["smoke_status"] == "BOUNDED_CAPTURE_SMOKE_DRY_RUN_READY"
    assert result["smoke_executed"] is False
    assert result["network_capture_executed"] is False
    assert (output_dir / "smoke_plan.csv").exists()
    assert (output_dir / "operator_preflight.csv").exists()


def test_capture_smoke_execute_is_blocked(tmp_path: Path) -> None:
    """Real smoke execution should require separate approval."""
    with pytest.raises(ValueError, match="Real capture smoke is blocked"):
        write_microstructure_capture_smoke(
            repo_root=tmp_path,
            output_dir=tmp_path / "smoke",
            execute=True,
        )


def test_capture_smoke_preserves_non_claims(tmp_path: Path) -> None:
    """Smoke artifacts should preserve research-only non-claims."""
    output_dir = tmp_path / "smoke"

    write_microstructure_capture_smoke(repo_root=tmp_path, output_dir=output_dir)

    report = json.loads((output_dir / "microstructure_capture_smoke.json").read_text())
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    for flag in ("RESEARCH_ONLY", "NO_RUNTIME_EFFECT", "NOT_PROMOTABLE", "NO_PROFIT_CLAIM"):
        assert flag in report["honesty_flags"]
        assert flag in recommendation["honesty_flags"]
    assert recommendation["runtime_ready"] is False


def test_capture_smoke_recommends_replay_engine_next(tmp_path: Path) -> None:
    """After dry-run smoke, fixtures can drive replay engine implementation."""
    result = write_microstructure_capture_smoke(
        repo_root=tmp_path,
        output_dir=tmp_path / "smoke",
    )

    assert result["recommendation"] == "BUILD_MICROSTRUCTURE_REPLAY_AND_COVERAGE_FROM_FIXTURES"
    assert result["next_required_action"] == "IMPLEMENT_MICROSTRUCTURE_REPLAY_GAP_ENGINE"
