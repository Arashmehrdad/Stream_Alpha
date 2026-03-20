"""Tests for the M7 file-based registry, rollback, and inference resolution."""

# pylint: disable=duplicate-code

from __future__ import annotations

from pathlib import Path

from app.training.compare import compare_run_to_current, write_comparison_artifact
from app.training.promote import promote_run
from app.training.registry import load_current_registry_entry, resolve_inference_model_path
from app.training.rollback import rollback_to_model_version
from tests.training_workflow_helpers import write_run_dir, write_workflow_config


def test_promote_run_updates_current_registry_pointer(tmp_path: Path) -> None:
    """Promoting one run should create and update the current registry pointer."""
    run_dir = write_run_dir(
        tmp_path / "m3",
        "20260319T223002Z",
        mean_net_value_proxy=0.001,
        directional_accuracy=0.55,
        brier_score=0.24,
    )

    current = promote_run(
        run_dir,
        model_version="m3-20260319T223002Z",
        registry_root=tmp_path / "registry",
    )

    assert current["model_version"] == "m3-20260319T223002Z"
    assert current["activation_reason"] == "PROMOTE"


def test_rollback_restores_the_previous_champion(tmp_path: Path) -> None:
    """Rollback should atomically restore an older promoted model version."""
    config_path = write_workflow_config(tmp_path / "training.m7.json")
    registry_root = tmp_path / "registry"
    champion_run = write_run_dir(
        tmp_path / "m3",
        "20260319T223002Z",
        mean_net_value_proxy=0.001,
        directional_accuracy=0.55,
        brier_score=0.24,
    )
    promote_run(
        champion_run,
        model_version="m3-20260319T223002Z",
        registry_root=registry_root,
    )
    challenger_run = write_run_dir(
        tmp_path / "m7",
        "20260320T010101Z",
        mean_net_value_proxy=0.002,
        directional_accuracy=0.555,
        brier_score=0.245,
    )
    comparison = compare_run_to_current(
        run_dir=challenger_run,
        config_path=config_path,
        registry_root=registry_root,
    )
    write_comparison_artifact(challenger_run, comparison)
    promote_run(
        challenger_run,
        model_version="m7-20260320T010101Z",
        registry_root=registry_root,
    )

    restored = rollback_to_model_version(
        "m3-20260319T223002Z",
        registry_root=registry_root,
    )

    assert restored["model_version"] == "m3-20260319T223002Z"
    assert restored["activation_reason"] == "ROLLBACK"
    assert load_current_registry_entry(registry_root)["model_version"] == "m3-20260319T223002Z"


def test_inference_model_resolution_uses_registry_when_override_is_empty(
    tmp_path: Path,
) -> None:
    """Empty inference override should fall back to the current promoted champion."""
    run_dir = write_run_dir(
        tmp_path / "m3",
        "20260319T223002Z",
        mean_net_value_proxy=0.001,
        directional_accuracy=0.55,
        brier_score=0.24,
    )
    current = promote_run(
        run_dir,
        model_version="m3-20260319T223002Z",
        registry_root=tmp_path / "registry",
    )

    resolved = resolve_inference_model_path("", registry_root=tmp_path / "registry")

    assert resolved == current["model_artifact_path"]


def test_inference_model_resolution_prefers_direct_override(tmp_path: Path) -> None:
    """A direct inference override should take precedence over the registry."""
    run_dir = write_run_dir(
        tmp_path / "m3",
        "20260319T223002Z",
        mean_net_value_proxy=0.001,
        directional_accuracy=0.55,
        brier_score=0.24,
    )
    promote_run(
        run_dir,
        model_version="m3-20260319T223002Z",
        registry_root=tmp_path / "registry",
    )
    override_path = run_dir / "model.joblib"

    resolved = resolve_inference_model_path(
        str(override_path),
        registry_root=tmp_path / "registry",
    )

    assert resolved == str(override_path.resolve())
