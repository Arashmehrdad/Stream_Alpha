"""Tests for the M4 saved-model loader."""

# pylint: disable=duplicate-code,too-few-public-methods

from __future__ import annotations

import io
from datetime import datetime, timedelta, timezone
import json
import pickle
from pathlib import Path
import zipfile

import joblib
import numpy.random._pickle as numpy_random_pickle
from numpy.random._mt19937 import MT19937
import pandas as pd
import pytest

from app.inference.service import load_model_artifact
from app.training import autogluon as autogluon_module
from app.training import neuralforecast as neuralforecast_module
from app.training import registry as registry_module
from app.training.autogluon import build_autogluon_tabular_classifier
from app.training.dataset import (
    DatasetSample,
    SourceFeatureRow,
    build_sequence_context_rows,
)
from app.training.neuralforecast import build_neuralforecast_nhits_classifier
from app.training.registry import write_json_atomic


_ORIGINAL_NUMPY_BIT_GENERATOR_CTOR = getattr(
    numpy_random_pickle, "__bit_generator_ctor"
)


class SerializableProbabilityModel:
    """Tiny serializable classifier stub with binary probabilities."""

    def __init__(self, prob_up: float = 0.7) -> None:
        self._prob_up = prob_up

    def predict_proba(self, rows: list[dict]) -> list[list[float]]:
        """Return a fixed binary probability for each requested row."""
        return [[1.0 - self._prob_up, self._prob_up] for _ in rows]


class _FakeNeuralForecastModel:
    def __init__(self, **kwargs) -> None:
        self.h = int(kwargs["h"])
        self.alias = str(kwargs["alias"])
        self.forecast_return = float(kwargs.get("forecast_return", 0.02))


class _FakeLoadedNeuralForecastCore:
    def __init__(self, state: dict[str, object]) -> None:
        self._alias = str(state["alias"])
        self._horizon = int(state["horizon"])
        self._frequency_minutes = int(state["frequency_minutes"])
        self._forecast_return = float(state["forecast_return"])

    def predict(self, df) -> pd.DataFrame:
        frame = pd.DataFrame(df)
        records: list[dict[str, object]] = []
        for unique_id, group in frame.groupby("unique_id"):
            ordered = group.sort_values("ds")
            last_row = ordered.iloc[-1]
            records.append(
                {
                    "unique_id": unique_id,
                    "ds": pd.Timestamp(last_row["ds"])
                    + pd.Timedelta(minutes=self._horizon * self._frequency_minutes),
                    self._alias: float(last_row["y"]) * (1.0 + self._forecast_return),
                }
            )
        return pd.DataFrame.from_records(records)


class _FakeNeuralForecastCore(_FakeLoadedNeuralForecastCore):
    def __init__(self, *, models: list[_FakeNeuralForecastModel], freq: str) -> None:
        frequency_minutes = int(str(freq).removesuffix("min"))
        model = models[0]
        super().__init__(
            {
                "alias": model.alias,
                "horizon": model.h,
                "frequency_minutes": frequency_minutes,
                "forecast_return": model.forecast_return,
            }
        )

    def fit(self, df, val_size: int, sort_df: bool = True) -> "_FakeNeuralForecastCore":
        assert val_size >= 0
        if isinstance(df, list):
            self._fit_df = None
            self._fit_directories = list(df)
            assert sort_df is False
        else:
            self._fit_df = pd.DataFrame(df).copy()
            self._fit_directories = None
        return self

    def save(self, path: str, overwrite: bool, save_dataset: bool) -> None:
        assert overwrite is True
        assert save_dataset is True
        target_dir = Path(path)
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "state.json").write_text(
            json.dumps(
                {
                    "alias": self._alias,
                    "horizon": self._horizon,
                    "frequency_minutes": self._frequency_minutes,
                    "forecast_return": self._forecast_return,
                }
            ),
            encoding="utf-8",
        )

    @staticmethod
    def load(path: str) -> _FakeLoadedNeuralForecastCore:
        state = json.loads((Path(path) / "state.json").read_text(encoding="utf-8"))
        return _FakeLoadedNeuralForecastCore(state)

    def cross_validation(
        self,
        *,
        df,
        n_windows: int,
        step_size: int,
        verbose: bool,
        refit: bool,
        use_init_models: bool,
    ) -> pd.DataFrame:
        assert step_size == 1
        assert verbose is False
        assert use_init_models is True
        assert isinstance(refit, bool)
        frame = pd.DataFrame(df)
        records: list[dict[str, object]] = []
        for unique_id, group in frame.groupby("unique_id"):
            ordered = group.sort_values("ds")
            valid_cutoffs = ordered.iloc[: max(len(ordered) - self._horizon, 0)]
            for _, cutoff_row in valid_cutoffs.tail(n_windows).iterrows():
                cutoff = pd.Timestamp(cutoff_row["ds"])
                records.append(
                    {
                        "unique_id": unique_id,
                        "cutoff": cutoff,
                        "ds": cutoff + pd.Timedelta(minutes=self._horizon * self._frequency_minutes),
                        self._alias: float(cutoff_row["y"]) * (1.0 + self._forecast_return),
                    }
                )
        return pd.DataFrame.from_records(records)


def _install_fake_neuralforecast_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        neuralforecast_module,
        "_import_neuralforecast_runtime",
        lambda model_class_name: (_FakeNeuralForecastCore, _FakeNeuralForecastModel),
    )


def _sequence_source_rows() -> list[SourceFeatureRow]:
    start = datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc)
    rows: list[SourceFeatureRow] = []
    for offset in range(8):
        as_of_time = start + timedelta(minutes=5 * offset)
        close_price = 100.0 + float(offset)
        rows.append(
            SourceFeatureRow(
                row_id=f"BTC/USD|{as_of_time.isoformat()}",
                symbol="BTC/USD",
                interval_begin=as_of_time,
                as_of_time=as_of_time,
                close_price=close_price,
                features={
                    "symbol": "BTC/USD",
                    "close_price": close_price,
                    "realized_vol_12": 0.01 * (offset + 1),
                    "momentum_3": 0.005 * offset,
                    "macd_line_12_26": 0.003 * offset,
                },
            )
        )
    return rows


def _sequence_samples() -> list[DatasetSample]:
    start = datetime(2026, 4, 1, 0, 10, tzinfo=timezone.utc)
    samples: list[DatasetSample] = []
    for offset, label in enumerate((1, 0, 1)):
        as_of_time = start + timedelta(minutes=5 * offset)
        close_price = 102.0 + float(offset)
        future_close_price = close_price * (1.02 if label == 1 else 0.98)
        samples.append(
            DatasetSample(
                row_id=f"BTC/USD|{as_of_time.isoformat()}",
                symbol="BTC/USD",
                interval_begin=as_of_time,
                as_of_time=as_of_time,
                close_price=close_price,
                future_close_price=future_close_price,
                future_return_3=(future_close_price / close_price) - 1.0,
                label=label,
                persistence_prediction=label,
                features={
                    "symbol": "BTC/USD",
                    "close_price": close_price,
                    "realized_vol_12": 0.01 * (offset + 3),
                    "momentum_3": 0.005 * (offset + 2),
                    "macd_line_12_26": 0.003 * (offset + 2),
                },
            )
        )
    return samples


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


def test_load_model_artifact_supports_self_contained_autogluon_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The authoritative AutoGluon artifact should preserve predict contracts after reload."""

    class _FakePredictionList:
        def __init__(self, values: list[int]) -> None:
            self._values = values

        def tolist(self) -> list[int]:
            return list(self._values)

    class _FakeProbabilityTable:
        def __init__(self, rows: list[dict[int, float]]) -> None:
            self._rows = rows

        def to_dict(self, orient: str) -> list[dict[int, float]]:
            assert orient == "records"
            return list(self._rows)

    class _FakeLoadedPredictor:
        def predict(self, frame) -> _FakePredictionList:
            return _FakePredictionList([1 for _ in range(len(frame))])

        def predict_proba(self, frame, *, as_multiclass: bool) -> _FakeProbabilityTable:
            assert as_multiclass is True
            return _FakeProbabilityTable([{0: 0.25, 1: 0.75} for _ in range(len(frame))])

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", "{}")

    monkeypatch.setattr(
        autogluon_module.TabularPredictor,
        "load",
        staticmethod(lambda path, **kwargs: _FakeLoadedPredictor()),
    )
    model = build_autogluon_tabular_classifier(
        {
            "calibrate_decision_threshold": False,
            "fit_weighted_ensemble": True,
            "hyperparameters": None,
            "num_bag_folds": 5,
            "num_bag_sets": 1,
            "num_stack_levels": 1,
            "presets": "high",
            "time_limit": 900,
            "verbosity": 0,
        }
    )
    model._predictor_archive = archive_buffer.getvalue()  # pylint: disable=protected-access
    model._feature_columns = (  # pylint: disable=protected-access
        "symbol",
        "realized_vol_12",
        "momentum_3",
        "macd_line_12_26",
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
            "training_model_config": model.get_training_config(),
            "model": model,
        },
        artifact_path,
    )

    loaded = load_model_artifact(str(artifact_path))
    predictions = loaded.model.predict(rows[:2])
    probabilities = loaded.model.predict_proba(rows[:2])

    assert loaded.model_name == "autogluon_tabular"
    assert loaded.model_version == "m3-20260401T120000Z"
    assert predictions == [1, 1]
    assert len(probabilities) == 2
    assert probabilities == [[0.25, 0.75], [0.25, 0.75]]


def test_load_model_artifact_supports_autogluon_legacy_numpy_random_pickle_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loaded AutoGluon artifacts should score through legacy MT19937 pickle paths."""

    class _LegacyMt19937StateFixture:
        def __reduce__(self):
            state = MT19937().__getstate__()
            return (
                _ORIGINAL_NUMPY_BIT_GENERATOR_CTOR,
                (MT19937,),
                (state, None),
            )

    legacy_payload = pickle.dumps(_LegacyMt19937StateFixture())

    class _FakePredictionList:
        def __init__(self, values: list[int]) -> None:
            self._values = values

        def tolist(self) -> list[int]:
            return list(self._values)

    class _FakeProbabilityTable:
        def __init__(self, rows: list[dict[int, float]]) -> None:
            self._rows = rows

        def to_dict(self, orient: str) -> list[dict[int, float]]:
            assert orient == "records"
            return list(self._rows)

    class _LegacyLoadedPredictor:
        @staticmethod
        def _trigger_legacy_ctor() -> None:
            pickle.loads(legacy_payload)

        def predict(self, frame) -> _FakePredictionList:
            self._trigger_legacy_ctor()
            return _FakePredictionList([1 for _ in range(len(frame))])

        def predict_proba(self, frame, *, as_multiclass: bool) -> _FakeProbabilityTable:
            assert as_multiclass is True
            self._trigger_legacy_ctor()
            return _FakeProbabilityTable([{0: 0.25, 1: 0.75} for _ in range(len(frame))])

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", "{}")

    monkeypatch.setattr(
        autogluon_module.TabularPredictor,
        "load",
        staticmethod(lambda path, **kwargs: _LegacyLoadedPredictor()),
    )
    model = build_autogluon_tabular_classifier(
        {
            "calibrate_decision_threshold": False,
            "fit_weighted_ensemble": True,
            "hyperparameters": None,
            "num_bag_folds": 5,
            "num_bag_sets": 1,
            "num_stack_levels": 1,
            "presets": "high",
            "time_limit": 900,
            "verbosity": 0,
        }
    )
    model._predictor_archive = archive_buffer.getvalue()  # pylint: disable=protected-access
    model._feature_columns = (  # pylint: disable=protected-access
        "symbol",
        "realized_vol_12",
        "momentum_3",
        "macd_line_12_26",
    )
    run_dir = tmp_path / "artifacts" / "training" / "m7" / "20260404T090000Z"
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = run_dir / "model.joblib"
    joblib.dump(
        {
            "model_name": "autogluon_tabular",
            "trained_at": "2026-04-04T09:00:00Z",
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
            "training_model_config": model.get_training_config(),
            "model": model,
        },
        artifact_path,
    )

    loaded = load_model_artifact(str(artifact_path))
    probabilities = loaded.model.predict_proba(
        [
            {
                "symbol": "BTC/USD",
                "realized_vol_12": 0.10,
                "momentum_3": 0.02,
                "macd_line_12_26": 0.50,
            }
        ]
    )

    assert loaded.model_name == "autogluon_tabular"
    assert probabilities == [[0.25, 0.75]]


def test_autogluon_runtime_loader_relaxes_python_version_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime loading should allow the deployed container to read local artifacts."""
    recorded: dict[str, object] = {}

    def _fake_load(path: str, **kwargs):
        recorded["path"] = path
        recorded["require_py_version_match"] = kwargs.get("require_py_version_match")
        return object()

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("metadata.json", "{}")

    monkeypatch.setattr(
        autogluon_module.TabularPredictor,
        "load",
        staticmethod(_fake_load),
    )
    classifier = build_autogluon_tabular_classifier({})
    classifier._predictor_archive = archive_buffer.getvalue()  # pylint: disable=protected-access

    classifier._ensure_predictor()  # pylint: disable=protected-access

    assert recorded["require_py_version_match"] is False


def test_load_model_artifact_supports_self_contained_neuralforecast_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NeuralForecast artifacts should reload and keep the predict_proba contract."""
    _install_fake_neuralforecast_runtime(monkeypatch)
    model = build_neuralforecast_nhits_classifier(
        {
            "candidate_role": "TREND_SPECIALIST",
            "input_size_candles": 2,
            "calibration_windows": 2,
            "max_steps": 5,
            "model_kwargs": {"forecast_return": 0.03},
            "scope_regimes": ["TREND_UP", "TREND_DOWN"],
        }
    )
    source_rows = _sequence_source_rows()
    samples = _sequence_samples()
    model.fit_samples(samples, source_rows=source_rows)

    run_dir = tmp_path / "artifacts" / "training" / "m20" / "20260403T120000Z"
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = run_dir / "model.joblib"
    joblib.dump(
        {
            "model_name": "neuralforecast_nhits",
            "trained_at": "2026-04-03T12:00:00Z",
            "feature_columns": model.get_feature_columns(),
            "expanded_feature_names": model.get_expanded_feature_names(),
            "training_model_config": model.get_training_config(),
            "registry_metadata": model.get_registry_metadata(),
            "model": model,
        },
        artifact_path,
    )

    loaded = load_model_artifact(str(artifact_path))
    context_rows = build_sequence_context_rows(
        target_samples=[samples[-1]],
        source_rows=source_rows,
        feature_columns=tuple(loaded.feature_columns),
        lookback_candles=model.input_size_candles,
    )
    probabilities = loaded.model.predict_proba(context_rows)

    assert loaded.model_name == "neuralforecast_nhits"
    assert loaded.model_version == "20260403T120000Z"
    assert loaded.model.requires_sequence_context() is True
    assert loaded.model.get_sequence_lookback_candles() == 2
    assert len(probabilities) == 1
    assert probabilities[0][1] > 0.0
