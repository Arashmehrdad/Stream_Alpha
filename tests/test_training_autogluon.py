"""Focused tests for the authoritative AutoGluon wrapper."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.training import autogluon as autogluon_module
from app.training.autogluon import (
    AutoGluonTabularClassifier,
    build_autogluon_tabular_classifier,
)


class _FakePredictionList:
    """Minimal list-like object exposing the pandas-style tolist contract."""

    def __init__(self, values: list[int]) -> None:
        self._values = values

    def tolist(self) -> list[int]:
        return list(self._values)


class _FakeProbabilityTable:
    """Minimal probability table exposing the pandas-style to_dict contract."""

    def __init__(self, rows: list[dict[int, float]]) -> None:
        self._rows = rows

    def to_dict(self, orient: str) -> list[dict[int, float]]:
        assert orient == "records"
        return list(self._rows)


class _FakeLoadedPredictor:
    """Tiny restored predictor used after the fit archive reload."""

    def predict(self, frame) -> _FakePredictionList:
        return _FakePredictionList([1 for _ in range(len(frame))])

    def predict_proba(
        self,
        frame,
        *,
        as_multiclass: bool,
    ) -> _FakeProbabilityTable:
        assert as_multiclass is True
        return _FakeProbabilityTable([{0: 0.2, 1: 0.8} for _ in range(len(frame))])


def _install_fake_tabular_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    recorded: dict[str, object] = {}

    class _FakeTabularPredictor:
        def __init__(
            self,
            *,
            label: str,
            problem_type: str,
            eval_metric: str,
            path: str,
        ) -> None:
            recorded["label"] = label
            recorded["problem_type"] = problem_type
            recorded["eval_metric"] = eval_metric
            self._path = Path(path)

        def fit(self, **kwargs):
            recorded["fit_kwargs"] = kwargs
            self._path.mkdir(parents=True, exist_ok=True)
            (self._path / "metadata.json").write_text("{}", encoding="utf-8")
            return self

        @staticmethod
        def load(path: str, **kwargs) -> _FakeLoadedPredictor:
            recorded["load_path"] = path
            recorded["load_kwargs"] = kwargs
            return _FakeLoadedPredictor()

    monkeypatch.setattr(
        autogluon_module,
        "TabularPredictor",
        _FakeTabularPredictor,
    )
    return recorded


def _rows() -> list[dict[str, object]]:
    return [
        {"symbol": "BTC/USD", "close_price": 101.0},
        {"symbol": "ETH/USD", "close_price": 99.5},
        {"symbol": "SOL/USD", "close_price": 103.2},
    ]


def test_build_autogluon_tabular_classifier_preserves_none_hyperparameters() -> None:
    """Missing or explicit null hyperparameters should stay unset."""
    missing = build_autogluon_tabular_classifier({})
    explicit_null = build_autogluon_tabular_classifier({"hyperparameters": None})

    assert missing.hyperparameters is None
    assert missing.__getstate__()["hyperparameters"] is None
    assert explicit_null.hyperparameters is None


def test_fit_omits_hyperparameters_when_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset hyperparameters should not collapse the fit call into tree-only defaults."""
    recorded = _install_fake_tabular_predictor(monkeypatch)
    classifier = build_autogluon_tabular_classifier(
        {
            "calibrate_decision_threshold": False,
            "num_bag_folds": 5,
            "num_bag_sets": 1,
            "presets": "high",
            "time_limit": 900,
        }
    )

    classifier.fit(_rows(), [1, 0, 1])

    fit_kwargs = recorded["fit_kwargs"]
    assert isinstance(fit_kwargs, dict)
    assert "hyperparameters" not in fit_kwargs


def test_fit_preserves_explicit_hyperparameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit hyperparameters should still pass straight through to AutoGluon."""
    recorded = _install_fake_tabular_predictor(monkeypatch)
    hyperparameters = {"RF": {}, "XT": {}}
    classifier = build_autogluon_tabular_classifier(
        {
            "hyperparameters": hyperparameters,
            "presets": "high",
            "time_limit": 900,
        }
    )

    classifier.fit(_rows(), [1, 0, 1])

    fit_kwargs = recorded["fit_kwargs"]
    assert isinstance(fit_kwargs, dict)
    assert fit_kwargs["hyperparameters"] == hyperparameters


def test_fit_stores_serializes_restores_and_passes_new_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """New fit controls should survive serialization and reach the fit call."""
    recorded = _install_fake_tabular_predictor(monkeypatch)
    classifier = build_autogluon_tabular_classifier(
        {
            "calibrate_decision_threshold": True,
            "num_bag_folds": 5,
            "num_bag_sets": 3,
            "num_stack_levels": 1,
            "presets": "high",
            "time_limit": 900,
        }
    )
    state = classifier.__getstate__()
    restored = AutoGluonTabularClassifier.__new__(AutoGluonTabularClassifier)
    restored.__setstate__(state)

    classifier.fit(_rows(), [1, 0, 1])

    fit_kwargs = recorded["fit_kwargs"]
    assert isinstance(fit_kwargs, dict)
    assert classifier.num_bag_sets == 3
    assert restored.num_bag_sets == 3
    assert fit_kwargs["num_bag_sets"] == 3
    assert classifier.calibrate_decision_threshold is True
    assert restored.calibrate_decision_threshold is True
    assert fit_kwargs["calibrate_decision_threshold"] is True


def test_fit_omits_num_bag_sets_when_bagging_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """num_bag_sets should only be forwarded when bagging is actually enabled."""
    recorded = _install_fake_tabular_predictor(monkeypatch)
    classifier = build_autogluon_tabular_classifier(
        {
            "num_bag_folds": 0,
            "num_bag_sets": 4,
        }
    )

    classifier.fit(_rows(), [1, 0, 1])

    fit_kwargs = recorded["fit_kwargs"]
    assert isinstance(fit_kwargs, dict)
    assert "num_bag_sets" not in fit_kwargs


def test_older_artifact_state_without_new_keys_loads_safely() -> None:
    """Older serialized artifacts should restore with sane defaults for new fields."""
    classifier = AutoGluonTabularClassifier.__new__(AutoGluonTabularClassifier)
    classifier.__setstate__(
        {
            "presets": "medium_quality",
            "time_limit": 120,
            "eval_metric": "log_loss",
            "hyperparameters": None,
            "fit_weighted_ensemble": True,
            "num_bag_folds": 0,
            "num_stack_levels": 0,
            "verbosity": 0,
            "feature_columns": ["symbol", "close_price"],
            "predictor_archive": b"legacy-archive",
        }
    )

    assert classifier.hyperparameters is None
    assert classifier.num_bag_sets == 1
    assert classifier.calibrate_decision_threshold is False
