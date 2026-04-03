"""Focused tests for the Apache-2.0 Moirai range-specialist wrapper."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd
import pytest

from app.inference.service import load_model_artifact
from app.training.dataset import DatasetSample, SourceFeatureRow, build_sequence_context_rows
from app.training.pretrained_forecasters import (
    DEFAULT_MOIRAI_1_0_R_BASE_LICENSE_NOTES,
    DEFAULT_MOIRAI_1_0_R_BASE_PRETRAINED_SOURCE,
    HF_REFERENCE_PRETRAINED_ARTIFACT_FORMAT,
    MODEL_FAMILY_MOIRAI_BASE,
    RANGE_SCOPE_REGIMES,
    RANGE_SPECIALIST,
    build_moirai_1_0_r_base_range_classifier,
    validate_pretrained_forecaster_contract,
)


class _FakeMOIRAIForecaster:
    """Tiny sktime-style runtime double for the Apache snapshot path."""

    init_calls: list[dict[str, object]] = []
    fit_calls: list[dict[str, object]] = []
    predict_calls: list[dict[str, object]] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = dict(kwargs)
        self._last_fit = None
        _FakeMOIRAIForecaster.init_calls.append(dict(kwargs))

    def fit(self, y, fh):
        frame = pd.DataFrame(y).copy()
        self._last_fit = frame
        _FakeMOIRAIForecaster.fit_calls.append(
            {
                "index_names": list(frame.index.names),
                "values": [float(value) for value in frame.iloc[:, 0].tolist()],
                "fh": list(fh),
            }
        )
        return self

    def predict(self, fh=None, y=None):
        frame = pd.DataFrame(y if y is not None else self._last_fit).copy()
        values = [float(value) for value in frame.iloc[:, 0].tolist()]
        _FakeMOIRAIForecaster.predict_calls.append(
            {
                "values": values,
                "fh": list(fh or []),
            }
        )
        last_value = values[-1]
        recent_mean = sum(values[-3:]) / float(min(len(values), 3))
        reversion = (recent_mean - last_value) * 0.6
        horizon_len = int(max(fh or [1]))
        forecasts = [
            last_value + (reversion * float(step))
            for step in range(1, horizon_len + 1)
        ]
        return pd.DataFrame(
            {"close_price": forecasts},
            index=pd.RangeIndex(start=1, stop=horizon_len + 1),
        )


def _source_rows() -> list[SourceFeatureRow]:
    start = datetime(2026, 4, 5, 0, 0, tzinfo=timezone.utc)
    close_prices = (100.0, 100.4, 100.0, 99.6, 100.1, 100.5, 100.2)
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
                },
            )
        )
    return rows


def _samples(source_rows: list[SourceFeatureRow]) -> list[DatasetSample]:
    labels_by_offset = {3: 1, 4: 0, 5: 0}
    samples: list[DatasetSample] = []
    for offset in (3, 4, 5):
        source_row = source_rows[offset]
        label = labels_by_offset[offset]
        future_close_price = source_row.close_price * (1.02 if label == 1 else 0.98)
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


def test_moirai_wrapper_uses_apache_snapshot_and_explicit_license_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moirai should use the sktime Apache snapshot path and surface the license note."""
    from app.training import pretrained_forecasters as pretrained_module

    _FakeMOIRAIForecaster.init_calls.clear()
    _FakeMOIRAIForecaster.fit_calls.clear()
    _FakeMOIRAIForecaster.predict_calls.clear()
    monkeypatch.setattr(
        pretrained_module,
        "_import_sktime_moirai_runtime",
        lambda: _FakeMOIRAIForecaster,
    )

    source_rows = _source_rows()
    samples = _samples(source_rows)
    wrapper = build_moirai_1_0_r_base_range_classifier(
        {
            "context_lookback_candles": 3,
            "patch_size": 16,
            "num_samples": 25,
            "batch_size": 8,
            "map_location": "cpu",
            "deterministic": True,
        }
    )

    wrapper.fit_samples(samples, source_rows=source_rows)
    contract = validate_pretrained_forecaster_contract(wrapper)
    probabilities = wrapper.predict_proba_samples(samples, source_rows=source_rows)

    assert _FakeMOIRAIForecaster.init_calls == [
        {
            "checkpoint_path": DEFAULT_MOIRAI_1_0_R_BASE_PRETRAINED_SOURCE,
            "context_length": 3,
            "patch_size": 16,
            "num_samples": 25,
            "map_location": "cpu",
            "target_dim": 1,
            "deterministic": True,
            "batch_size": 8,
            "use_source_package": False,
        }
    ]
    assert _FakeMOIRAIForecaster.fit_calls[0]["index_names"] == ["instance", "timestamp"]
    assert _FakeMOIRAIForecaster.predict_calls[0]["values"] == [100.4, 100.0, 99.6]
    assert contract.registry_metadata["model_family"] == MODEL_FAMILY_MOIRAI_BASE
    assert contract.registry_metadata["candidate_role"] == RANGE_SPECIALIST
    assert contract.registry_metadata["scope_regimes"] == list(RANGE_SCOPE_REGIMES)
    assert contract.registry_metadata["artifact_format"] == HF_REFERENCE_PRETRAINED_ARTIFACT_FORMAT
    assert contract.registry_metadata["license_name"] == "Apache-2.0"
    assert contract.registry_metadata["license_notes"] == DEFAULT_MOIRAI_1_0_R_BASE_LICENSE_NOTES
    assert contract.training_config["checkpoint_provider"] == "sktime_snapshot"
    assert contract.training_config["input_mapping"] == "UNIVARIATE_CLOSE_PRICE_ONLY"
    assert probabilities[0][1] > probabilities[1][1] >= probabilities[2][1]


def test_moirai_artifact_reloads_through_saved_model_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moirai should persist as a joblib artifact that lazily reloads the Apache snapshot."""
    from app.training import pretrained_forecasters as pretrained_module

    _FakeMOIRAIForecaster.init_calls.clear()
    _FakeMOIRAIForecaster.fit_calls.clear()
    _FakeMOIRAIForecaster.predict_calls.clear()
    monkeypatch.setattr(
        pretrained_module,
        "_import_sktime_moirai_runtime",
        lambda: _FakeMOIRAIForecaster,
    )

    source_rows = _source_rows()
    samples = _samples(source_rows)
    wrapper = build_moirai_1_0_r_base_range_classifier(
        {
            "context_lookback_candles": 3,
            "map_location": "cpu",
        }
    )
    wrapper.fit_samples(samples, source_rows=source_rows)
    contract = validate_pretrained_forecaster_contract(wrapper)

    run_dir = tmp_path / "artifacts" / "training" / "m20" / "20260405T180000Z"
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = run_dir / "model.joblib"
    joblib.dump(
        {
            "model_name": "moirai_1_0_r_base_range",
            "trained_at": "2026-04-05T18:00:00Z",
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

    assert loaded_artifact.model_name == "moirai_1_0_r_base_range"
    assert loaded_artifact.feature_columns == ("symbol", "close_price")
    assert len(_FakeMOIRAIForecaster.init_calls) == 2
    assert _FakeMOIRAIForecaster.init_calls[-1]["checkpoint_path"] == (
        DEFAULT_MOIRAI_1_0_R_BASE_PRETRAINED_SOURCE
    )
    assert probabilities[0][1] >= 0.0
