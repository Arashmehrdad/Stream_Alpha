"""Tests for DU8 dry-run microstructure capture service."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.training import microstructure_capture_service as capture_service
from app.training.microstructure_capture_service import (
    build_capture_dry_run_plan,
    write_microstructure_capture_service_dry_run,
)


def test_capture_dry_run_plan_builds_subscription() -> None:
    """Dry-run planning should render the Kraken book subscription."""
    plan = build_capture_dry_run_plan(
        symbols=("BTC/USD",),
        depth=10,
        duration_seconds=60,
        max_events=1000,
    )

    assert plan.dry_run is True
    assert plan.subscription["params"]["channel"] == "book"
    assert plan.subscription["params"]["depth"] == 10


def test_capture_dry_run_rejects_unbounded_parameters() -> None:
    """Duration and event caps should be enforced before implementation."""
    with pytest.raises(ValueError, match="duration_seconds"):
        build_capture_dry_run_plan(
            symbols=("BTC/USD",),
            depth=10,
            duration_seconds=0,
            max_events=1000,
        )
    with pytest.raises(ValueError, match="max_events"):
        build_capture_dry_run_plan(
            symbols=("BTC/USD",),
            depth=10,
            duration_seconds=60,
            max_events=0,
        )


def test_capture_execute_requires_dsn(tmp_path: Path) -> None:
    """Real capture execution must require explicit storage configuration."""
    with pytest.raises(ValueError, match="requires a PostgreSQL DSN"):
        write_microstructure_capture_service_dry_run(
            repo_root=tmp_path,
            output_dir=tmp_path / "capture",
            execute=True,
        )


def test_capture_execute_uses_bounded_capture_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execute mode should call the bounded capture wrapper with approved settings."""

    def fake_capture(**kwargs: object) -> dict[str, object]:
        assert kwargs["dsn"] == "postgresql://example"
        return {"captured_event_count": 2, "duration_seconds": 1, "max_events": 2}

    monkeypatch.setattr(capture_service, "run_bounded_capture_sync", fake_capture)
    result = write_microstructure_capture_service_dry_run(
        repo_root=tmp_path,
        output_dir=tmp_path / "capture",
        execute=True,
        dsn="postgresql://example",
        duration_seconds=1,
        max_events=2,
    )

    assert result["network_capture_executed"] is True
    assert result["captured_event_count"] == 2


def test_capture_dry_run_writes_artifacts(tmp_path: Path) -> None:
    """The dry-run service should persist deterministic artifacts."""
    output_dir = tmp_path / "capture"

    result = write_microstructure_capture_service_dry_run(
        repo_root=tmp_path,
        output_dir=output_dir,
    )

    assert result["capture_service_status"] == "DRY_RUN_CAPTURE_SERVICE_PLAN_DEFINED"
    assert result["network_capture_executed"] is False
    assert (output_dir / "capture_plan.csv").exists()
    assert (output_dir / "subscription_plan.csv").exists()


def test_capture_dry_run_preserves_non_claims(tmp_path: Path) -> None:
    """The dry-run output should preserve research-only non-claims."""
    output_dir = tmp_path / "capture"

    write_microstructure_capture_service_dry_run(repo_root=tmp_path, output_dir=output_dir)

    report = json.loads(
        (output_dir / "microstructure_capture_service_dry_run.json").read_text()
    )
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    for flag in ("RESEARCH_ONLY", "NO_RUNTIME_EFFECT", "NOT_PROMOTABLE", "NO_PROFIT_CLAIM"):
        assert flag in report["honesty_flags"]
        assert flag in recommendation["honesty_flags"]
    assert recommendation["runtime_ready"] is False
