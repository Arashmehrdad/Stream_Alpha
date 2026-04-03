"""Tests for the M7 file-based registry, rollback, and inference resolution."""

# pylint: disable=duplicate-code

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.training.compare import compare_run_to_current, write_comparison_artifact
from app.training.promote import promote_run
from app.training.pretrained_forecasters import (
    DEFAULT_PRETRAINED_ARTIFACT_FORMAT,
    DEFAULT_MOIRAI_1_0_R_BASE_LICENSE_NOTES,
    GENERALIST,
    MODEL_FAMILY_AMAZON_CHRONOS_2,
    MODEL_FAMILY_MOIRAI_BASE,
    RANGE_SPECIALIST,
)
from app.training.registry import (
    build_run_manifest,
    load_current_registry_entry,
    resolve_inference_model_path,
)
from app.training.rollback import rollback_to_model_version
from tests.training_workflow_helpers import write_run_dir, write_workflow_config


def test_promote_run_updates_current_registry_pointer(tmp_path: Path) -> None:
    """Promoting one run should create and update the current registry pointer."""
    run_dir = write_run_dir(
        tmp_path / "m3",
        "20260319T223002Z",
        mean_long_only_net_value_proxy=0.001,
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
        mean_long_only_net_value_proxy=0.002,
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
        mean_long_only_net_value_proxy=0.001,
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
        mean_long_only_net_value_proxy=0.001,
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


def test_run_manifest_carries_economics_contract_and_acceptance(tmp_path: Path) -> None:
    """Run manifests should expose the long-only economics contract explicitly."""
    run_dir = write_run_dir(
        tmp_path / "m7",
        "20260320T010101Z",
        mean_long_only_net_value_proxy=-0.0002,
        directional_accuracy=0.56,
        brier_score=0.24,
        persistence_mean_long_only_net_value_proxy=-0.0003,
        dummy_mean_long_only_net_value_proxy=-0.0004,
    )

    manifest = build_run_manifest(run_dir)

    assert manifest["economics_contract"]["name"] == "LONG_ONLY_AFTER_COST_PROXY"
    assert manifest["acceptance"]["winner_after_cost_positive"] is False
    assert manifest["acceptance"]["meets_acceptance_target"] is False


def test_run_manifest_and_registry_entry_record_autogluon_training_config(
    tmp_path: Path,
) -> None:
    """The existing run-manifest and registry metadata path should carry fit config truth."""
    training_model_config = {
        "presets": "high",
        "time_limit": 900,
        "eval_metric": "log_loss",
        "hyperparameters": None,
        "fit_weighted_ensemble": True,
        "num_bag_folds": 5,
        "num_stack_levels": 1,
        "num_bag_sets": 1,
        "calibrate_decision_threshold": False,
        "verbosity": 0,
    }
    run_dir = write_run_dir(
        tmp_path / "m7",
        "20260401T120000Z",
        model_name="autogluon_tabular",
        mean_long_only_net_value_proxy=0.002,
        directional_accuracy=0.58,
        brier_score=0.22,
        training_model_config=training_model_config,
    )

    manifest = build_run_manifest(run_dir)
    current = promote_run(
        run_dir,
        model_version="m7-20260401T120000Z",
        registry_root=tmp_path / "registry",
    )

    assert manifest["winner"]["training_config"] == training_model_config
    assert current["training_model_config"] == training_model_config


def test_run_manifest_and_registry_entry_record_specialist_registry_metadata(
    tmp_path: Path,
) -> None:
    """Registry paths should preserve specialist model-family discovery metadata."""
    registry_metadata = {
        "model_family": "NEURALFORECAST_NHITS",
        "candidate_role": "TREND_SPECIALIST",
        "scope_regimes": ["TREND_UP", "TREND_DOWN"],
    }
    run_dir = write_run_dir(
        tmp_path / "m20",
        "20260403T120000Z",
        model_name="neuralforecast_nhits",
        mean_long_only_net_value_proxy=0.001,
        directional_accuracy=0.57,
        brier_score=0.23,
        training_model_config={"model_family": "NEURALFORECAST_NHITS"},
        registry_metadata=registry_metadata,
    )

    manifest = build_run_manifest(run_dir)
    current = promote_run(
        run_dir,
        model_version="m20-20260403T120000Z",
        registry_root=tmp_path / "registry",
    )

    assert manifest["winner"]["metadata"] == registry_metadata
    assert current["metadata"] == registry_metadata


def test_run_manifest_and_registry_entry_record_future_pretrained_metadata(
    tmp_path: Path,
) -> None:
    """Registry paths should preserve staged metadata for future pretrained challengers."""
    registry_metadata = {
        "model_family": MODEL_FAMILY_AMAZON_CHRONOS_2,
        "candidate_role": GENERALIST,
        "scope_regimes": ["TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL"],
        "artifact_format": DEFAULT_PRETRAINED_ARTIFACT_FORMAT,
        "pretrained_source": "amazon/chronos-2",
    }
    run_dir = write_run_dir(
        tmp_path / "m20",
        "20260403T123000Z",
        model_name="chronos2_smoke_scaffold",
        mean_long_only_net_value_proxy=0.0005,
        directional_accuracy=0.56,
        brier_score=0.23,
        training_model_config={
            "adapter": "calibrated_forecast_score",
            "model_family": MODEL_FAMILY_AMAZON_CHRONOS_2,
        },
        registry_metadata=registry_metadata,
    )

    manifest = build_run_manifest(run_dir)
    current = promote_run(
        run_dir,
        model_version="m20-20260403T123000Z",
        registry_root=tmp_path / "registry",
    )

    assert manifest["winner"]["metadata"]["model_family"] == MODEL_FAMILY_AMAZON_CHRONOS_2
    assert manifest["winner"]["metadata"]["candidate_role"] == GENERALIST
    assert current["metadata"]["model_family"] == MODEL_FAMILY_AMAZON_CHRONOS_2
    assert current["metadata"]["candidate_role"] == GENERALIST
    assert current["metadata"]["artifact_format"] == DEFAULT_PRETRAINED_ARTIFACT_FORMAT


def test_run_manifest_and_registry_entry_preserve_moirai_license_notes(
    tmp_path: Path,
) -> None:
    """Registry metadata should keep the explicit Apache snapshot note for Moirai."""
    registry_metadata = {
        "model_family": MODEL_FAMILY_MOIRAI_BASE,
        "candidate_role": RANGE_SPECIALIST,
        "scope_regimes": ["RANGE", "HIGH_VOL"],
        "artifact_format": DEFAULT_PRETRAINED_ARTIFACT_FORMAT,
        "pretrained_source": "sktime/moirai-1.0-R-base",
        "license_name": "Apache-2.0",
        "license_notes": DEFAULT_MOIRAI_1_0_R_BASE_LICENSE_NOTES,
    }
    run_dir = write_run_dir(
        tmp_path / "m20",
        "20260403T131500Z",
        model_name="moirai_range_smoke_scaffold",
        mean_long_only_net_value_proxy=0.0004,
        directional_accuracy=0.55,
        brier_score=0.24,
        training_model_config={
            "adapter": "calibrated_forecast_score",
            "model_family": MODEL_FAMILY_MOIRAI_BASE,
        },
        registry_metadata=registry_metadata,
    )

    manifest = build_run_manifest(run_dir)
    current = promote_run(
        run_dir,
        model_version="m20-20260403T131500Z",
        registry_root=tmp_path / "registry",
    )

    assert manifest["winner"]["metadata"]["model_family"] == MODEL_FAMILY_MOIRAI_BASE
    assert manifest["winner"]["metadata"]["candidate_role"] == RANGE_SPECIALIST
    assert manifest["winner"]["metadata"]["license_name"] == "Apache-2.0"
    assert (
        manifest["winner"]["metadata"]["license_notes"]
        == DEFAULT_MOIRAI_1_0_R_BASE_LICENSE_NOTES
    )
    assert current["metadata"]["model_family"] == MODEL_FAMILY_MOIRAI_BASE
    assert current["metadata"]["candidate_role"] == RANGE_SPECIALIST
    assert current["metadata"]["license_name"] == "Apache-2.0"
    assert current["metadata"]["license_notes"] == DEFAULT_MOIRAI_1_0_R_BASE_LICENSE_NOTES


def test_promote_run_rejects_legacy_archived_sklearn_winner(tmp_path: Path) -> None:
    """Legacy sklearn winners must not promote into the authoritative registry."""
    run_dir = write_run_dir(
        tmp_path / "m3",
        "20260319T223002Z",
        model_name="logistic_regression",
        mean_long_only_net_value_proxy=0.001,
        directional_accuracy=0.55,
        brier_score=0.24,
    )

    with pytest.raises(ValueError, match="Legacy archived sklearn model"):
        promote_run(
            run_dir,
            model_version="m3-20260319T223002Z",
            registry_root=tmp_path / "registry",
        )


def test_promote_run_bootstraps_over_legacy_archived_current_entry(tmp_path: Path) -> None:
    """A new authoritative run should replace the archived current pointer without comparison."""
    registry_root = tmp_path / "registry"
    legacy_run = write_run_dir(
        tmp_path / "legacy",
        "20260319T223002Z",
        model_name="logistic_regression",
        mean_long_only_net_value_proxy=0.001,
        directional_accuracy=0.55,
        brier_score=0.24,
    )
    legacy_current = {
        "model_version": "m3-20260319T223002Z",
        "model_name": "logistic_regression",
        "model_artifact_path": str((legacy_run / "model.joblib").resolve()),
        "run_manifest_path": str((legacy_run / "run_manifest.json").resolve()),
    }
    (registry_root).mkdir(parents=True, exist_ok=True)
    (registry_root / "current.json").write_text(
        json.dumps(legacy_current, indent=2),
        encoding="utf-8",
    )

    authoritative_run = write_run_dir(
        tmp_path / "m3",
        "20260401T120000Z",
        model_name="autogluon_tabular",
        mean_long_only_net_value_proxy=0.002,
        directional_accuracy=0.58,
        brier_score=0.22,
    )

    current = promote_run(
        authoritative_run,
        model_version="m3-20260401T120000Z",
        registry_root=registry_root,
    )

    assert current["model_name"] == "autogluon_tabular"
    assert current["model_version"] == "m3-20260401T120000Z"
