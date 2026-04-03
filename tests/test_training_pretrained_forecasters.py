"""Focused tests for the shared pretrained forecaster contract helpers."""

# pylint: disable=missing-function-docstring,too-few-public-methods

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.training.dataset import DatasetSample, SourceFeatureRow
from app.training.pretrained_forecasters import (
    DEFAULT_PRETRAINED_ARTIFACT_FORMAT,
    DEFAULT_SCOPE_REGIMES,
    ForecastToProbabilityCalibrator,
    GENERALIST,
    MODEL_FAMILY_AMAZON_CHRONOS_2,
    PretrainedForecasterArtifactMetadata,
    validate_pretrained_forecaster_contract,
)


class _SmokePretrainedForecaster:
    """Tiny future-family scaffold used to validate the shared contract."""

    def __init__(
        self,
        *,
        metadata: PretrainedForecasterArtifactMetadata | dict[str, object],
        lookback_candles: int = 3,
    ) -> None:
        self._metadata = metadata
        self._lookback_candles = lookback_candles
        self._feature_columns = ("symbol", "close_price", "realized_vol_12")
        self._calibrator = ForecastToProbabilityCalibrator()

    def fit(
        self,
        rows: list[dict[str, object]],
        labels: list[int],
    ) -> "_SmokePretrainedForecaster":
        del rows, labels
        raise ValueError("Pretrained forecaster smoke scaffold requires fit_samples(...)")

    def fit_samples(
        self,
        samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        dataset_export_root=None,
        progress_callback=None,
    ) -> "_SmokePretrainedForecaster":
        del source_rows, dataset_export_root, progress_callback
        raw_scores = [float(sample.future_return_3) for sample in samples]
        labels = [int(sample.label) for sample in samples]
        self._calibrator.fit(raw_scores, labels)
        return self

    def predict(self, rows: list[dict[str, object]]) -> list[int]:
        return [1 if row[1] >= 0.5 else 0 for row in self.predict_proba(rows)]

    def predict_proba(self, rows: list[dict[str, object]]) -> list[list[float]]:
        raw_scores = [float(row["close_price"]) / 100.0 - 1.0 for row in rows]
        return self._calibrator.predict_proba(raw_scores)

    def predict_samples(
        self,
        test_samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        progress_callback=None,
    ) -> list[int]:
        del progress_callback
        return [
            1 if row[1] >= 0.5 else 0
            for row in self.predict_proba_samples(
                test_samples,
                source_rows=source_rows,
            )
        ]

    def predict_proba_samples(
        self,
        test_samples: list[DatasetSample],
        *,
        source_rows: list[SourceFeatureRow],
        progress_callback=None,
    ) -> list[list[float]]:
        del source_rows, progress_callback
        raw_scores = [float(sample.future_return_3) for sample in test_samples]
        return self._calibrator.predict_proba(raw_scores)

    def requires_sequence_context(self) -> bool:
        return True

    def get_sequence_lookback_candles(self) -> int:
        return self._lookback_candles

    def get_feature_columns(self) -> list[str]:
        return list(self._feature_columns)

    def get_expanded_feature_names(self) -> list[str]:
        return list(self._feature_columns)

    def get_training_config(self) -> dict[str, object]:
        return {
            "adapter_kind": "calibrated_forecast_score",
        }

    def get_registry_metadata(
        self,
    ) -> PretrainedForecasterArtifactMetadata | dict[str, object]:
        return self._metadata


def _source_rows() -> list[SourceFeatureRow]:
    start = datetime(2026, 4, 3, 0, 0, tzinfo=timezone.utc)
    rows: list[SourceFeatureRow] = []
    for offset in range(6):
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
                },
            )
        )
    return rows


def _samples() -> list[DatasetSample]:
    start = datetime(2026, 4, 3, 0, 10, tzinfo=timezone.utc)
    samples: list[DatasetSample] = []
    labels = (0, 1, 1)
    for offset, label in enumerate(labels):
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
                    "realized_vol_12": 0.02 * (offset + 1),
                },
            )
        )
    return samples


def test_validate_pretrained_forecaster_contract_accepts_future_generalist_wrapper() -> None:
    """Future pretrained wrappers should validate before we add real model logic."""
    metadata = PretrainedForecasterArtifactMetadata(
        model_family=MODEL_FAMILY_AMAZON_CHRONOS_2,
        candidate_role=GENERALIST,
        scope_regimes=DEFAULT_SCOPE_REGIMES,
        pretrained_source="amazon/chronos-2",
    )
    wrapper = _SmokePretrainedForecaster(metadata=metadata)
    wrapper.fit_samples(_samples(), source_rows=_source_rows())

    contract = validate_pretrained_forecaster_contract(wrapper)
    probabilities = wrapper.predict_proba(
        [
            {
                "symbol": "BTC/USD",
                "close_price": 103.0,
                "realized_vol_12": 0.04,
            }
        ]
    )

    assert contract.registry_metadata["model_family"] == MODEL_FAMILY_AMAZON_CHRONOS_2
    assert contract.registry_metadata["candidate_role"] == GENERALIST
    assert contract.registry_metadata["artifact_format"] == DEFAULT_PRETRAINED_ARTIFACT_FORMAT
    assert contract.training_config["model_family"] == MODEL_FAMILY_AMAZON_CHRONOS_2
    assert contract.training_config["scope_regimes"] == list(DEFAULT_SCOPE_REGIMES)
    assert contract.requires_sequence_context is True
    assert contract.sequence_lookback_candles == 3
    assert len(probabilities) == 1
    assert len(probabilities[0]) == 2


def test_validate_pretrained_forecaster_contract_rejects_missing_candidate_role() -> None:
    """Invalid registry metadata should fail fast before artifact save/load."""
    wrapper = _SmokePretrainedForecaster(
        metadata={
            "model_family": MODEL_FAMILY_AMAZON_CHRONOS_2,
        }
    )
    wrapper.fit_samples(_samples(), source_rows=_source_rows())

    with pytest.raises(ValueError, match="candidate_role"):
        validate_pretrained_forecaster_contract(wrapper)
