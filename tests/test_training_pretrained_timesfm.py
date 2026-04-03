"""Focused tests for the real TimesFM 2.0 pretrained challenger wrapper."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pytest

from app.inference.service import load_model_artifact
from app.training.dataset import DatasetSample, SourceFeatureRow, build_sequence_context_rows
from app.training.pretrained_forecasters import (
    DEFAULT_TIMESFM_2_0_500M_PYTORCH_PRETRAINED_SOURCE,
    HF_REFERENCE_PRETRAINED_ARTIFACT_FORMAT,
    MODEL_FAMILY_GOOGLE_TIMESFM_2_0_500M_PYTORCH,
    TREND_SCOPE_REGIMES,
    TREND_SPECIALIST,
    build_timesfm_2_0_500m_pytorch_classifier,
    validate_pretrained_forecaster_contract,
)


class _FakeTimesFmHparams:
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)


class _FakeTimesFmCheckpoint:
    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)


class _FakeTimesFm:
    init_calls: list[dict[str, object]] = []
    forecast_calls: list[dict[str, object]] = []

    def __init__(self, *, hparams, checkpoint) -> None:
        self.hparams = hparams
        self.checkpoint = checkpoint
        _FakeTimesFm.init_calls.append(
            {
                "hparams": dict(hparams.kwargs),
                "checkpoint": dict(checkpoint.kwargs),
            }
        )

    def forecast(
        self,
        *,
        inputs,
        freq=None,
        window_size=None,
        forecast_context_len=None,
        return_forecast_on_context=False,
        normalize=False,
    ):
        del window_size, return_forecast_on_context
        rendered_inputs = [np.asarray(values, dtype=float) for values in inputs]
        _FakeTimesFm.forecast_calls.append(
            {
                "inputs": [values.tolist() for values in rendered_inputs],
                "freq": list(freq or []),
                "forecast_context_len": forecast_context_len,
                "normalize": normalize,
            }
        )
        horizon_len = int(self.hparams.kwargs["horizon_len"])
        mean_rows: list[np.ndarray] = []
        for values in rendered_inputs:
            slope = float(values[-1] - values[0])
            step = slope / float(max(len(values) - 1, 1))
            horizon = np.asarray(
                [float(values[-1] + ((index + 1) * step)) for index in range(horizon_len)],
                dtype=float,
            )
            mean_rows.append(horizon)
        mean_forecast = np.vstack(mean_rows)
        quantile_forecast = mean_forecast[:, :, None]
        return mean_forecast, quantile_forecast


def _fake_timesfm_module() -> SimpleNamespace:
    return SimpleNamespace(
        TimesFm=_FakeTimesFm,
        TimesFmHparams=_FakeTimesFmHparams,
        TimesFmCheckpoint=_FakeTimesFmCheckpoint,
    )


def _source_rows() -> list[SourceFeatureRow]:
    start = datetime(2026, 4, 4, 0, 0, tzinfo=timezone.utc)
    close_prices = (100.0, 99.0, 98.0, 97.0, 101.0, 103.0, 105.0)
    rows: list[SourceFeatureRow] = []
    for offset, close_price in enumerate(close_prices):
        as_of_time = start + timedelta(minutes=5 * offset)
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
                    "momentum_3": (-0.02 if offset < 4 else 0.03),
                },
            )
        )
    return rows


def _samples(source_rows: list[SourceFeatureRow]) -> list[DatasetSample]:
    labels_by_offset = {3: 0, 4: 1, 5: 1}
    samples: list[DatasetSample] = []
    for offset in (3, 4, 5):
        source_row = source_rows[offset]
        label = labels_by_offset[offset]
        future_close_price = source_row.close_price * (1.03 if label == 1 else 0.97)
        samples.append(
            DatasetSample(
                row_id=source_row.row_id,
                symbol=source_row.symbol,
                interval_begin=source_row.interval_begin,
                as_of_time=source_row.as_of_time,
                close_price=source_row.close_price,
                future_close_price=future_close_price,
                future_return_3=(future_close_price / source_row.close_price) - 1.0,
                label=label,
                persistence_prediction=label,
                features=dict(source_row.features),
            )
        )
    return samples


def test_timesfm_wrapper_fits_with_honest_univariate_input_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TimesFM should load through the shared wrapper and only consume univariate close history."""
    from app.training import pretrained_forecasters as pretrained_module

    _FakeTimesFm.init_calls.clear()
    _FakeTimesFm.forecast_calls.clear()
    monkeypatch.setattr(
        pretrained_module,
        "_import_timesfm_runtime",
        _fake_timesfm_module,
    )

    source_rows = _source_rows()
    samples = _samples(source_rows)
    wrapper = build_timesfm_2_0_500m_pytorch_classifier(
        {
            "context_lookback_candles": 3,
            "calibration_windows": 3,
            "prediction_batch_size": 2,
            "backend": "cpu",
            "normalize_inputs": True,
        }
    )

    wrapper.fit_samples(samples, source_rows=source_rows)
    contract = validate_pretrained_forecaster_contract(wrapper)
    probabilities = wrapper.predict_proba_samples(samples, source_rows=source_rows)

    assert _FakeTimesFm.init_calls == [
        {
            "hparams": {
                "backend": "cpu",
                "per_core_batch_size": 32,
                "horizon_len": 3,
                "context_len": 2048,
                "input_patch_len": 32,
                "output_patch_len": 128,
                "num_layers": 50,
                "model_dims": 1280,
                "use_positional_embedding": False,
                "point_forecast_mode": "median",
            },
            "checkpoint": {
                "huggingface_repo_id": DEFAULT_TIMESFM_2_0_500M_PYTORCH_PRETRAINED_SOURCE,
            },
        }
    ]
    assert contract.registry_metadata["model_family"] == MODEL_FAMILY_GOOGLE_TIMESFM_2_0_500M_PYTORCH
    assert contract.registry_metadata["candidate_role"] == TREND_SPECIALIST
    assert contract.registry_metadata["scope_regimes"] == list(TREND_SCOPE_REGIMES)
    assert contract.registry_metadata["artifact_format"] == HF_REFERENCE_PRETRAINED_ARTIFACT_FORMAT
    assert contract.training_config["input_mapping"] == "UNIVARIATE_CLOSE_PRICE_ONLY"
    assert contract.training_config["frequency_indicator"] == 0
    assert tuple(contract.feature_columns) == ("symbol", "close_price")
    assert _FakeTimesFm.forecast_calls[-2]["inputs"][0] == [99.0, 98.0, 97.0]
    assert _FakeTimesFm.forecast_calls[-1]["freq"] == [0]
    assert _FakeTimesFm.forecast_calls[-1]["normalize"] is True
    assert probabilities[0][1] < probabilities[1][1] <= probabilities[2][1]


def test_timesfm_artifact_reloads_through_saved_model_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TimesFM should persist as a joblib artifact that lazily reloads the HF checkpoint."""
    from app.training import pretrained_forecasters as pretrained_module

    _FakeTimesFm.init_calls.clear()
    _FakeTimesFm.forecast_calls.clear()
    monkeypatch.setattr(
        pretrained_module,
        "_import_timesfm_runtime",
        _fake_timesfm_module,
    )

    source_rows = _source_rows()
    samples = _samples(source_rows)
    wrapper = build_timesfm_2_0_500m_pytorch_classifier(
        {
            "context_lookback_candles": 3,
            "calibration_windows": 3,
            "prediction_batch_size": 2,
            "backend": "cpu",
        }
    )
    wrapper.fit_samples(samples, source_rows=source_rows)
    contract = validate_pretrained_forecaster_contract(wrapper)

    run_dir = tmp_path / "artifacts" / "training" / "m20" / "20260404T180000Z"
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = run_dir / "model.joblib"
    joblib.dump(
        {
            "model_name": "timesfm_2_0_500m_pytorch_trend",
            "trained_at": "2026-04-04T18:00:00Z",
            "feature_columns": list(contract.feature_columns),
            "expanded_feature_names": list(contract.expanded_feature_names),
            "training_model_config": contract.training_config,
            "registry_metadata": contract.registry_metadata,
            "model": wrapper,
        },
        artifact_path,
    )

    loaded_artifact = load_model_artifact(str(artifact_path))
    scoring_rows = build_sequence_context_rows(
        target_samples=[samples[-1]],
        source_rows=source_rows,
        feature_columns=tuple(contract.feature_columns),
        lookback_candles=3,
    )
    probabilities = loaded_artifact.model.predict_proba(scoring_rows)

    assert loaded_artifact.model_name == "timesfm_2_0_500m_pytorch_trend"
    assert loaded_artifact.feature_columns == ("symbol", "close_price")
    assert len(_FakeTimesFm.init_calls) == 2
    assert _FakeTimesFm.init_calls[-1]["checkpoint"] == {
        "huggingface_repo_id": DEFAULT_TIMESFM_2_0_500M_PYTORCH_PRETRAINED_SOURCE,
    }
    assert probabilities[0][1] > 0.5
