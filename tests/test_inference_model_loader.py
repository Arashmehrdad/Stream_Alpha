"""Tests for the M4 saved-model loader."""

# pylint: disable=duplicate-code,too-few-public-methods

from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from app.inference.service import load_model_artifact
from app.training.registry import write_json_atomic


class SerializableProbabilityModel:
    """Tiny serializable classifier stub with binary probabilities."""

    def __init__(self, prob_up: float = 0.7) -> None:
        self._prob_up = prob_up

    def predict_proba(self, rows: list[dict]) -> list[list[float]]:
        """Return a fixed binary probability for each requested row."""
        return [[1.0 - self._prob_up, self._prob_up] for _ in rows]


def _write_artifact(tmp_path: Path) -> Path:
    run_dir = tmp_path / "artifacts" / "training" / "m3" / "20260319T223002Z"
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = run_dir / "model.joblib"
    joblib.dump(
        {
            "model_name": "logistic_regression",
            "trained_at": "2026-03-19T22:30:02Z",
            "feature_columns": ["symbol", "close_price"],
            "expanded_feature_names": ["symbol=BTC/USD", "close_price"],
            "model": SerializableProbabilityModel(),
        },
        artifact_path,
    )
    return artifact_path


def test_load_model_artifact_successfully(tmp_path: Path) -> None:
    """A well-formed saved artifact should load with validated metadata."""
    artifact = load_model_artifact(str(_write_artifact(tmp_path)))

    assert artifact.model_name == "logistic_regression"
    assert artifact.model_version == "m3-20260319T223002Z"
    assert artifact.model_version_source == "RUN_DIR_DERIVED"
    assert artifact.feature_columns == ("symbol", "close_price")
    assert artifact.model_artifact_path.endswith("model.joblib")


def test_load_model_artifact_uses_registry_current_metadata_when_override_is_empty(
    tmp_path: Path,
) -> None:
    """Registry-backed loading should expose the promoted model version metadata."""
    artifact_path = _write_artifact(tmp_path)
    registry_root = tmp_path / "registry"
    write_json_atomic(
        registry_root / "current.json",
        {
            "model_version": "m7-20260320T010101Z",
            "model_artifact_path": str(artifact_path.resolve()),
        },
    )

    artifact = load_model_artifact("", registry_root=registry_root)

    assert artifact.model_version == "m7-20260320T010101Z"
    assert artifact.model_version_source == "REGISTRY_CURRENT"


def test_load_model_artifact_rejects_bad_path(tmp_path: Path) -> None:
    """A missing artifact path should fail fast."""
    missing_path = tmp_path / "missing.joblib"

    with pytest.raises(ValueError, match="INFERENCE_MODEL_PATH does not exist"):
        load_model_artifact(str(missing_path))


def test_load_model_artifact_rejects_malformed_payload(tmp_path: Path) -> None:
    """Missing required keys should fail validation."""
    artifact_path = tmp_path / "bad.joblib"
    joblib.dump({"model_name": "broken"}, artifact_path)

    with pytest.raises(ValueError, match="missing required keys"):
        load_model_artifact(str(artifact_path))
