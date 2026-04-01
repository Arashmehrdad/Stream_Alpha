"""Tests for the M4 saved-model loader."""

# pylint: disable=duplicate-code,too-few-public-methods

from __future__ import annotations

from pathlib import Path

import joblib
import pytest

from app.inference.service import load_model_artifact
from app.training import registry as registry_module
from app.training.autogluon import build_autogluon_tabular_classifier
from app.training.registry import write_json_atomic


class SerializableProbabilityModel:
    """Tiny serializable classifier stub with binary probabilities."""

    def __init__(self, prob_up: float = 0.7) -> None:
        self._prob_up = prob_up

    def predict_proba(self, rows: list[dict]) -> list[list[float]]:
        """Return a fixed binary probability for each requested row."""
        return [[1.0 - self._prob_up, self._prob_up] for _ in rows]


def _write_artifact(
    tmp_path: Path,
    *,
    model_name: str = "runtime_candidate_fixture",
) -> Path:
    run_dir = tmp_path / "artifacts" / "training" / "m3" / "20260319T223002Z"
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = run_dir / "model.joblib"
    joblib.dump(
        {
            "model_name": model_name,
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

    assert artifact.model_name == "runtime_candidate_fixture"
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


def test_load_model_artifact_rejects_legacy_archived_override_model(
    tmp_path: Path,
) -> None:
    """Direct override loading should reject legacy sklearn artifacts."""
    artifact_path = _write_artifact(tmp_path, model_name="logistic_regression")

    with pytest.raises(ValueError, match="Legacy archived sklearn model"):
        load_model_artifact(str(artifact_path))


def test_load_model_artifact_rejects_legacy_archived_registry_current_model(
    tmp_path: Path,
) -> None:
    """Registry-backed loading should reject legacy sklearn current champions."""
    artifact_path = _write_artifact(tmp_path, model_name="hist_gradient_boosting")
    registry_root = tmp_path / "registry"
    write_json_atomic(
        registry_root / "current.json",
        {
            "model_version": "m7-20260320T010101Z",
            "model_name": "hist_gradient_boosting",
            "model_artifact_path": str(artifact_path.resolve()),
        },
    )

    with pytest.raises(ValueError, match="Legacy archived sklearn model"):
        load_model_artifact("", registry_root=registry_root)


def test_load_model_artifact_translates_windows_registry_paths_for_runtime_portability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry-backed loading should resolve Windows host paths inside Linux runtimes."""
    fake_repo_root = tmp_path / "workspace"
    artifact_path = _write_artifact(fake_repo_root)
    registry_root = fake_repo_root / "artifacts" / "registry"
    write_json_atomic(
        registry_root / "current.json",
        {
            "model_version": "m3-20260319T223002Z",
            "model_artifact_path": (
                "Z:\\remote\\Stream_Alpha\\artifacts\\training\\m3\\20260319T223002Z\\model.joblib"
            ),
        },
    )
    monkeypatch.setattr(registry_module, "repo_root", lambda: fake_repo_root)

    artifact = load_model_artifact("", registry_root=registry_root)

    assert Path(artifact.model_artifact_path) == artifact_path.resolve()
    assert artifact.model_version == "m3-20260319T223002Z"
    assert artifact.model_version_source == "REGISTRY_CURRENT"


def test_load_model_artifact_translates_windows_override_paths_for_runtime_portability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit override paths should resolve Windows host paths inside Linux runtimes."""
    fake_repo_root = tmp_path / "workspace"
    artifact_path = _write_artifact(fake_repo_root)
    monkeypatch.setattr(registry_module, "repo_root", lambda: fake_repo_root)

    artifact = load_model_artifact(
        "Z:\\remote\\Stream_Alpha\\artifacts\\training\\m3\\20260319T223002Z\\model.joblib"
    )

    assert Path(artifact.model_artifact_path) == artifact_path.resolve()
    assert artifact.model_version == "m3-20260319T223002Z"
    assert artifact.model_version_source == "RUN_DIR_DERIVED"


def test_load_model_artifact_rejects_malformed_payload(tmp_path: Path) -> None:
    """Missing required keys should fail validation."""
    artifact_path = tmp_path / "bad.joblib"
    joblib.dump({"model_name": "broken"}, artifact_path)

    with pytest.raises(ValueError, match="missing required keys"):
        load_model_artifact(str(artifact_path))


def test_load_model_artifact_supports_self_contained_autogluon_artifact(tmp_path: Path) -> None:
    """The authoritative AutoGluon artifact should load after its fit directory is gone."""
    model = build_autogluon_tabular_classifier(
        {
            "hyperparameters": {"RF": {}, "XT": {}},
            "fit_weighted_ensemble": False,
            "num_bag_folds": 0,
            "num_stack_levels": 0,
            "presets": "medium_quality",
            "time_limit": 30,
            "verbosity": 0,
        }
    )
    rows = [
        {
            "symbol": "BTC/USD",
            "realized_vol_12": 0.10,
            "momentum_3": 0.02,
            "macd_line_12_26": 0.50,
        },
        {
            "symbol": "ETH/USD",
            "realized_vol_12": 0.20,
            "momentum_3": -0.02,
            "macd_line_12_26": -0.40,
        },
        {
            "symbol": "BTC/USD",
            "realized_vol_12": 0.11,
            "momentum_3": 0.03,
            "macd_line_12_26": 0.55,
        },
        {
            "symbol": "ETH/USD",
            "realized_vol_12": 0.22,
            "momentum_3": -0.03,
            "macd_line_12_26": -0.45,
        },
        {
            "symbol": "BTC/USD",
            "realized_vol_12": 0.12,
            "momentum_3": 0.01,
            "macd_line_12_26": 0.52,
        },
        {
            "symbol": "ETH/USD",
            "realized_vol_12": 0.24,
            "momentum_3": -0.01,
            "macd_line_12_26": -0.42,
        },
    ]
    model.fit(rows, [1, 0, 1, 0, 1, 0])
    run_dir = tmp_path / "artifacts" / "training" / "m3" / "20260401T120000Z"
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = run_dir / "model.joblib"
    joblib.dump(
        {
            "model_name": "autogluon_tabular",
            "trained_at": "2026-04-01T12:00:00Z",
            "feature_columns": [
                "symbol",
                "realized_vol_12",
                "momentum_3",
                "macd_line_12_26",
            ],
            "expanded_feature_names": [
                "symbol",
                "realized_vol_12",
                "momentum_3",
                "macd_line_12_26",
            ],
            "model": model,
        },
        artifact_path,
    )

    loaded = load_model_artifact(str(artifact_path))
    probabilities = loaded.model.predict_proba(rows[:2])

    assert loaded.model_name == "autogluon_tabular"
    assert loaded.model_version == "m3-20260401T120000Z"
    assert len(probabilities) == 2
    assert len(probabilities[0]) == 2
