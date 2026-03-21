"""Tests for the M7 challenger-versus-champion comparison policy."""

# pylint: disable=duplicate-code

from __future__ import annotations

from pathlib import Path

import pytest

from app.training.compare import compare_run_to_current
from app.training.promote import promote_run
from tests.training_workflow_helpers import write_run_dir, write_workflow_config


def test_compare_run_passes_when_challenger_improves_long_only_net_value(tmp_path: Path) -> None:
    """A compatible challenger with better net value should pass promotion policy."""
    config_path = write_workflow_config(tmp_path / "training.m7.json")
    registry_root = tmp_path / "registry"
    champion_run = write_run_dir(
        tmp_path / "m3",
        "20260319T223002Z",
        mean_long_only_net_value_proxy=0.001,
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
        mean_long_only_net_value_proxy=0.0025,
        directional_accuracy=0.553,
        brier_score=0.245,
    )

    comparison = compare_run_to_current(
        run_dir=challenger_run,
        config_path=config_path,
        registry_root=registry_root,
    )

    assert comparison["passed"] is True
    assert comparison["decision"] == "promote"
    assert comparison["challenger"]["economics_contract"]["name"] == "LONG_ONLY_AFTER_COST_PROXY"
    assert comparison["challenger"]["acceptance"]["meets_acceptance_target"] is True


def test_compare_run_fails_when_guardrail_is_broken(tmp_path: Path) -> None:
    """A challenger that regresses too far on accuracy should be rejected."""
    config_path = write_workflow_config(tmp_path / "training.m7.json")
    registry_root = tmp_path / "registry"
    champion_run = write_run_dir(
        tmp_path / "m3",
        "20260319T223002Z",
        mean_long_only_net_value_proxy=0.001,
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
        mean_long_only_net_value_proxy=0.0025,
        directional_accuracy=0.52,
        brier_score=0.245,
    )

    comparison = compare_run_to_current(
        run_dir=challenger_run,
        config_path=config_path,
        registry_root=registry_root,
    )

    assert comparison["passed"] is False
    assert any("Directional accuracy regressed" in reason for reason in comparison["reasons"])


def test_compare_run_fails_fast_when_required_artifact_is_missing(tmp_path: Path) -> None:
    """Missing required run artifacts should stop comparison with an explicit error."""
    config_path = write_workflow_config(tmp_path / "training.m7.json")
    challenger_run = write_run_dir(
        tmp_path / "m7",
        "20260320T010101Z",
        mean_long_only_net_value_proxy=0.0025,
        directional_accuracy=0.553,
        brier_score=0.245,
    )
    (challenger_run / "summary.json").unlink()

    with pytest.raises(ValueError, match="Run artifact directory is incomplete"):
        compare_run_to_current(
            run_dir=challenger_run,
            config_path=config_path,
            registry_root=tmp_path / "registry",
        )


def test_compare_run_fails_when_challenger_is_not_positive_after_costs(
    tmp_path: Path,
) -> None:
    """A negative long-only after-cost challenger should not promote or bootstrap."""
    config_path = write_workflow_config(tmp_path / "training.m7.json")

    challenger_run = write_run_dir(
        tmp_path / "m7",
        "20260320T010101Z",
        mean_long_only_net_value_proxy=-0.0002,
        directional_accuracy=0.60,
        brier_score=0.22,
        persistence_mean_long_only_net_value_proxy=-0.0004,
        dummy_mean_long_only_net_value_proxy=-0.0005,
    )

    comparison = compare_run_to_current(
        run_dir=challenger_run,
        config_path=config_path,
        registry_root=tmp_path / "registry",
    )

    assert comparison["passed"] is False
    assert any("not positive after costs" in reason for reason in comparison["reasons"])
    assert comparison["challenger"]["acceptance"]["winner_after_cost_positive"] is False


def test_compare_run_fails_when_challenger_does_not_beat_dummy_baseline(
    tmp_path: Path,
) -> None:
    """The after-cost promotion contract must beat the dumb baseline."""
    config_path = write_workflow_config(tmp_path / "training.m7.json")

    challenger_run = write_run_dir(
        tmp_path / "m7",
        "20260320T010101Z",
        mean_long_only_net_value_proxy=0.0002,
        directional_accuracy=0.57,
        brier_score=0.23,
        persistence_mean_long_only_net_value_proxy=0.0001,
        dummy_mean_long_only_net_value_proxy=0.00025,
    )

    comparison = compare_run_to_current(
        run_dir=challenger_run,
        config_path=config_path,
        registry_root=tmp_path / "registry",
    )

    assert comparison["passed"] is False
    assert any("dummy_most_frequent" in reason for reason in comparison["reasons"])
