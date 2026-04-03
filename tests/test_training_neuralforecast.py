"""Focused tests for the NeuralForecast specialist wrappers."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd
import pytest

from app.training import neuralforecast as neuralforecast_module
from app.training.dataset import (
    SEQUENCE_CONTEXT_KEY,
    DatasetSample,
    SourceFeatureRow,
    build_sequence_context_rows,
)
from app.training.neuralforecast import (
    ForecastToProbabilityCalibrator,
    build_neuralforecast_nhits_classifier,
    build_neuralforecast_patchtst_classifier,
)


class _FakeNeuralForecastModel:
    """Minimal model config stub used by the fake NeuralForecast core."""

    last_init_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs) -> None:
        type(self).last_init_kwargs = dict(kwargs)
        self.h = int(kwargs["h"])
        self.alias = str(kwargs["alias"])
        self.forecast_return = float(kwargs.get("forecast_return", 0.02))


class _FakeLoadedNeuralForecastCore:
    """Minimal restored backend used by predict_proba after artifact reload."""

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
    """Minimal fit/save/cross-validation backend for deterministic wrapper tests."""

    last_fit_val_size: int | None = None
    fit_call_count: int = 0
    save_call_history: list[bool] = []

    def __init__(self, *, models: list[_FakeNeuralForecastModel], freq: str) -> None:
        frequency_minutes = int(str(freq).removesuffix("min"))
        self._model = models[0]
        super().__init__(
            {
                "alias": self._model.alias,
                "horizon": self._model.h,
                "frequency_minutes": frequency_minutes,
                "forecast_return": self._model.forecast_return,
            }
        )
        self._fit_df: pd.DataFrame | None = None
        self._fit_directories: list[str] | None = None
        self.last_val_size: int | None = None

    def fit(self, df, val_size: int, sort_df: bool = True) -> "_FakeNeuralForecastCore":
        type(self).fit_call_count += 1
        self.last_val_size = int(val_size)
        type(self).last_fit_val_size = self.last_val_size
        if isinstance(df, list):
            self._fit_directories = list(df)
            self._fit_df = None
        else:
            self._fit_df = pd.DataFrame(df).copy()
            self._fit_directories = None
        assert sort_df is False or sort_df is True
        return self

    def save(self, path: str, overwrite: bool, save_dataset: bool) -> None:
        assert overwrite is True
        type(self).save_call_history.append(bool(save_dataset))
        if self._fit_directories is not None and save_dataset:
            raise AttributeError("'NeuralForecast' object has no attribute 'ds'")
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
        del df, n_windows, step_size, verbose, refit, use_init_models
        raise AssertionError(
            "Specialist training should not call full NeuralForecast cross_validation",
        )


def _install_fake_neuralforecast_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        neuralforecast_module,
        "_import_neuralforecast_runtime",
        lambda model_class_name: (_FakeNeuralForecastCore, _FakeNeuralForecastModel),
    )


def _source_rows() -> list[SourceFeatureRow]:
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


def _samples() -> list[DatasetSample]:
    start = datetime(2026, 4, 1, 0, 10, tzinfo=timezone.utc)
    samples: list[DatasetSample] = []
    for offset in range(3):
        as_of_time = start + timedelta(minutes=5 * offset)
        close_price = 102.0 + float(offset)
        future_close_price = close_price * 1.02
        samples.append(
            DatasetSample(
                row_id=f"BTC/USD|{as_of_time.isoformat()}",
                symbol="BTC/USD",
                interval_begin=as_of_time,
                as_of_time=as_of_time,
                close_price=close_price,
                future_close_price=future_close_price,
                future_return_3=(future_close_price / close_price) - 1.0,
                label=1,
                persistence_prediction=1,
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


def test_forecast_to_probability_calibrator_serializes(tmp_path: Path) -> None:
    """The forecast-to-probability bridge should survive serialization."""
    calibrator = ForecastToProbabilityCalibrator().fit(
        raw_scores=[-0.2, -0.05, 0.05, 0.2],
        labels=[0, 0, 1, 1],
    )
    calibrator_path = tmp_path / "calibrator.joblib"
    joblib.dump(calibrator, calibrator_path)
    restored = joblib.load(calibrator_path)

    low_prob = restored.predict_proba([-0.1])[0][1]
    high_prob = restored.predict_proba([0.1])[0][1]

    assert low_prob < high_prob


def test_build_sequence_context_rows_preserves_order_and_no_leakage() -> None:
    """Sequence contexts should only use same-symbol history up to the target row."""
    source_rows = _source_rows()
    sample = _samples()[1]

    context_rows = build_sequence_context_rows(
        target_samples=[sample],
        source_rows=source_rows,
        feature_columns=("symbol", "close_price", "realized_vol_12"),
        lookback_candles=3,
    )

    context = context_rows[0][SEQUENCE_CONTEXT_KEY]
    assert len(context) == 3
    assert [row["as_of_time"] for row in context] == sorted(row["as_of_time"] for row in context)
    assert max(row["as_of_time"] for row in context) == sample.as_of_time
    assert all(row["as_of_time"] <= sample.as_of_time for row in context)


def test_neuralforecast_nhits_wrapper_trains_saves_loads_and_scores(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NHITS wrapper artifacts should reload and keep the predict_proba contract."""
    _install_fake_neuralforecast_runtime(monkeypatch)
    classifier = build_neuralforecast_nhits_classifier(
        {
            "candidate_role": "TREND_SPECIALIST",
            "input_size_candles": 2,
            "calibration_windows": 2,
            "max_steps": 5,
            "model_kwargs": {"forecast_return": 0.03},
            "scope_regimes": ["TREND_UP", "TREND_DOWN"],
        }
    )
    source_rows = _source_rows()
    samples = _samples()

    _FakeNeuralForecastCore.save_call_history = []
    classifier.fit_samples(
        samples,
        source_rows=source_rows,
        dataset_export_root=tmp_path / "artifact-save",
    )
    context_rows = build_sequence_context_rows(
        target_samples=[samples[-1]],
        source_rows=source_rows,
        feature_columns=tuple(classifier.get_feature_columns()),
        lookback_candles=classifier.input_size_candles,
    )
    artifact_path = Path("D:/Github/Stream_Alpha/artifacts/tmp/neuralforecast-wrapper-test.joblib")
    if artifact_path.exists():
        artifact_path.unlink()
    joblib.dump(classifier, artifact_path)
    restored = joblib.load(artifact_path)

    probabilities = restored.predict_proba(context_rows)

    assert len(probabilities) == 1
    assert probabilities[0][1] > 0.5
    assert _FakeNeuralForecastCore.save_call_history == [True, False]
    assert restored.get_training_config()["model_family"] == "NEURALFORECAST_NHITS"
    assert restored.get_registry_metadata()["candidate_role"] == "TREND_SPECIALIST"
    artifact_path.unlink()


def test_neuralforecast_patchtst_wrapper_scores_fold_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PatchTST should support the sample-aware fold prediction contract."""
    _install_fake_neuralforecast_runtime(monkeypatch)
    classifier = build_neuralforecast_patchtst_classifier(
        {
            "candidate_role": "RANGE_SPECIALIST",
            "input_size_candles": 2,
            "calibration_windows": 2,
            "max_steps": 5,
            "model_kwargs": {"forecast_return": 0.01},
            "scope_regimes": ["RANGE"],
        }
    )
    source_rows = _source_rows()
    samples = _samples()

    classifier.fit_samples(samples, source_rows=source_rows)
    probabilities = classifier.predict_proba_samples(
        samples,
        source_rows=source_rows,
    )

    assert len(probabilities) == len(samples)
    assert all(len(row) == 2 for row in probabilities)
    assert classifier.get_registry_metadata()["model_family"] == "NEURALFORECAST_PATCHTST"


def test_neuralforecast_wrapper_exports_partitioned_local_files_dataset_deterministically(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Specialist training should export one ordered parquet partition per series."""
    _install_fake_neuralforecast_runtime(monkeypatch)
    classifier = build_neuralforecast_nhits_classifier(
        {
            "dataset_mode": "local_files_partitioned",
            "input_size_candles": 2,
            "calibration_windows": 2,
            "max_steps": 5,
        }
    )

    export_root = tmp_path / "local-files"
    classifier.fit_samples(
        _samples(),
        source_rows=_source_rows(),
        dataset_export_root=export_root,
    )

    manifest_path = export_root / "full_history_fit" / "dataset_export.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parquet_path = (
        export_root
        / "full_history_fit"
        / "unique_id=BTC%2FUSD"
        / "part-00000.parquet"
    )
    frame = pd.read_parquet(parquet_path)

    assert classifier.get_training_config()["dataset_mode"] == "local_files_partitioned"
    assert manifest["source_row_count"] == len(_source_rows())
    assert manifest["symbols"]["BTC/USD"]["series_id"] == "BTC%2FUSD"
    assert frame["ds"].tolist() == sorted(frame["ds"].tolist())


def test_neuralforecast_wrapper_uses_partitioned_local_files_fit_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Specialist fit should use the local-files dataset loader rather than one big DataFrame."""
    _install_fake_neuralforecast_runtime(monkeypatch)
    classifier = build_neuralforecast_nhits_classifier(
        {
            "dataset_mode": "local_files_partitioned",
            "input_size_candles": 2,
            "calibration_windows": 2,
            "max_steps": 5,
        }
    )

    classifier.fit_samples(
        _samples(),
        source_rows=_source_rows(),
        dataset_export_root=tmp_path / "specialist-export",
    )

    assert classifier._backend is not None  # pylint: disable=protected-access
    assert classifier._backend._fit_directories is not None  # pylint: disable=protected-access
    assert classifier._backend._fit_df is None  # pylint: disable=protected-access
    assert classifier._backend._fit_directories[0].endswith("unique_id=BTC%2FUSD")  # pylint: disable=protected-access


def test_neuralforecast_wrapper_calibrator_reuses_single_backend_fit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Calibration should reuse the fitted specialist backend instead of launching a second fit."""
    _install_fake_neuralforecast_runtime(monkeypatch)
    classifier = build_neuralforecast_nhits_classifier(
        {
            "dataset_mode": "local_files_partitioned",
            "input_size_candles": 2,
            "calibration_windows": 2,
            "max_steps": 5,
        }
    )

    _FakeNeuralForecastCore.fit_call_count = 0
    classifier.fit_samples(
        _samples(),
        source_rows=_source_rows(),
        dataset_export_root=tmp_path / "reuse-fit",
    )

    assert _FakeNeuralForecastCore.fit_call_count == 1


def test_neuralforecast_wrapper_uses_positive_val_size_when_early_stopping_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """NeuralForecast fit should reserve a small validation tail when early stopping is on."""
    _install_fake_neuralforecast_runtime(monkeypatch)
    classifier = build_neuralforecast_nhits_classifier(
        {
            "input_size_candles": 2,
            "calibration_windows": 2,
            "max_steps": 5,
            "early_stop_patience_steps": 20,
        }
    )

    _FakeNeuralForecastCore.last_fit_val_size = None
    classifier._fit_backend(  # pylint: disable=protected-access
        source_rows=_source_rows(),
        dataset_export_root=tmp_path / "val-size-fit",
    )

    assert _FakeNeuralForecastCore.last_fit_val_size is not None
    assert _FakeNeuralForecastCore.last_fit_val_size > 0
    assert classifier._resolve_fit_val_size(source_rows=_source_rows()) > 0  # pylint: disable=protected-access


def test_neuralforecast_wrapper_passes_mixed_precision_and_small_window_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Specialist memory controls should flow through to the NeuralForecast model constructor."""
    _install_fake_neuralforecast_runtime(monkeypatch)
    monkeypatch.setattr(neuralforecast_module, "_torch_cuda_is_available", lambda: True)
    classifier = build_neuralforecast_patchtst_classifier(
        {
            "batch_size": 1,
            "input_size_candles": 2,
            "calibration_windows": 2,
            "max_steps": 5,
            "model_kwargs": {
                "precision": "16-mixed",
                "step_size": 16,
                "valid_batch_size": 1,
                "windows_batch_size": 4,
                "inference_windows_batch_size": 4,
            },
        }
    )

    _FakeNeuralForecastModel.last_init_kwargs = None
    classifier._fit_backend(  # pylint: disable=protected-access
        source_rows=_source_rows(),
        dataset_export_root=tmp_path / "memory-controls-fit",
    )

    assert _FakeNeuralForecastModel.last_init_kwargs is not None
    assert _FakeNeuralForecastModel.last_init_kwargs["precision"] == "16-mixed"
    assert _FakeNeuralForecastModel.last_init_kwargs["step_size"] == 16
    assert _FakeNeuralForecastModel.last_init_kwargs["valid_batch_size"] == 1
    assert _FakeNeuralForecastModel.last_init_kwargs["windows_batch_size"] == 4
    assert _FakeNeuralForecastModel.last_init_kwargs["inference_windows_batch_size"] == 4


def test_extract_forecast_value_accepts_unique_id_column() -> None:
    """Forecast parsing should work when NeuralForecast exposes unique_id as a plain column."""
    classifier = build_neuralforecast_nhits_classifier({})
    classifier._series_id_by_symbol = {"BTC/USD": "BTC%2FUSD"}  # pylint: disable=protected-access
    target_timestamp = pd.Timestamp("2026-04-01T00:15:00Z")
    frame = pd.DataFrame(
        [
            {
                "unique_id": "BTC%2FUSD",
                "ds": target_timestamp,
                "NHITS": 105.0,
            }
        ]
    )

    forecast_value = classifier._extract_forecast_value(  # pylint: disable=protected-access
        forecast_frame=frame,
        symbol="BTC/USD",
        target_timestamp=target_timestamp,
    )

    assert forecast_value == 105.0


def test_extract_forecast_value_recovers_unique_id_from_named_index() -> None:
    """Forecast parsing should recover the series id when it arrives through a named index."""
    classifier = build_neuralforecast_nhits_classifier({})
    classifier._series_id_by_symbol = {"BTC/USD": "BTC%2FUSD"}  # pylint: disable=protected-access
    target_timestamp = pd.Timestamp("2026-04-01T00:15:00Z")
    frame = pd.DataFrame(
        [{"ds": target_timestamp, "NHITS": 106.0}],
        index=pd.Index(["BTC%2FUSD"], name="unique_id"),
    )

    forecast_value = classifier._extract_forecast_value(  # pylint: disable=protected-access
        forecast_frame=frame,
        symbol="BTC/USD",
        target_timestamp=target_timestamp,
    )

    assert forecast_value == 106.0


def test_extract_forecast_value_recovers_unique_id_from_reset_index_field() -> None:
    """Forecast parsing should recover the series id when reset_index yields an index field."""
    classifier = build_neuralforecast_patchtst_classifier({})
    classifier._series_id_by_symbol = {"BTC/USD": "BTC%2FUSD"}  # pylint: disable=protected-access
    target_timestamp = pd.Timestamp("2026-04-01T00:15:00Z")
    frame = pd.DataFrame(
        [{"ds": target_timestamp, "PatchTST": 107.0}],
        index=pd.Index(["BTC%2FUSD"]),
    )

    forecast_value = classifier._extract_forecast_value(  # pylint: disable=protected-access
        forecast_frame=frame,
        symbol="BTC/USD",
        target_timestamp=target_timestamp,
    )

    assert forecast_value == 107.0


def test_predict_raw_scores_from_context_rows_batches_sequence_contexts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sequence scoring should batch synthetic contexts and emit progress payloads."""

    class _BatchPredictBackend:
        def __init__(self) -> None:
            self.predict_call_count = 0

        def predict(self, df) -> pd.DataFrame:
            self.predict_call_count += 1
            frame = pd.DataFrame(df)
            records: list[dict[str, object]] = []
            index: list[str] = []
            for unique_id, group in frame.groupby("unique_id"):
                ordered = group.sort_values("ds")
                last_row = ordered.iloc[-1]
                records.append(
                    {
                        "ds": pd.Timestamp(last_row["ds"]) + pd.Timedelta(minutes=15),
                        "NHITS": float(last_row["y"]) * 1.01,
                    }
                )
                index.append(str(unique_id))
            return pd.DataFrame(records, index=pd.Index(index, name="unique_id"))

    classifier = build_neuralforecast_nhits_classifier({})
    classifier._hist_exog_columns = ("close_price", "realized_vol_12")  # pylint: disable=protected-access
    classifier._frequency_minutes = 5  # pylint: disable=protected-access
    backend = _BatchPredictBackend()
    progress_events: list[dict[str, object]] = []
    monkeypatch.setattr(neuralforecast_module, "_PREDICT_CONTEXT_BATCH_SIZE", 2)

    def _context_rows(
        *,
        symbol: str,
        start: datetime,
        close_prices: tuple[float, ...],
    ) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for offset, close_price in enumerate(close_prices):
            rows.append(
                {
                    "symbol": symbol,
                    "as_of_time": start + timedelta(minutes=5 * offset),
                    "close_price": close_price,
                    "realized_vol_12": 0.01 * (offset + 1),
                }
            )
        return rows

    rows = [
        {
            SEQUENCE_CONTEXT_KEY: _context_rows(
                symbol="BTC/USD",
                start=datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
                close_prices=(100.0, 101.0),
            )
        },
        {
            SEQUENCE_CONTEXT_KEY: _context_rows(
                symbol="BTC/USD",
                start=datetime(2026, 4, 1, 0, 5, tzinfo=timezone.utc),
                close_prices=(101.0, 102.0),
            )
        },
        {
            SEQUENCE_CONTEXT_KEY: _context_rows(
                symbol="ETH/USD",
                start=datetime(2026, 4, 1, 0, 0, tzinfo=timezone.utc),
                close_prices=(200.0, 202.0),
            )
        },
    ]

    raw_scores = classifier._predict_raw_scores_from_context_rows(  # pylint: disable=protected-access
        rows,
        backend=backend,
        progress_callback=lambda payload: progress_events.append(dict(payload)),
    )

    assert backend.predict_call_count == 2
    assert raw_scores == pytest.approx([0.01, 0.01, 0.01])
    assert [event["event"] for event in progress_events] == [
        "sequence_scoring_start",
        "sequence_scoring_progress",
        "sequence_scoring_progress",
        "sequence_scoring_complete",
    ]
    assert progress_events[0]["row_count"] == 3
    assert progress_events[0]["batch_count"] == 2
    assert progress_events[1]["completed_batches"] == 1
    assert progress_events[1]["completed_rows"] == 2
    assert progress_events[1]["eta_seconds"] >= 0.0
    assert progress_events[2]["completed_batches"] == 2
    assert progress_events[2]["progress"] == pytest.approx(1.0)
    assert progress_events[3]["elapsed_seconds"] >= 0.0
