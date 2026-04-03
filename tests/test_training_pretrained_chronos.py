"""Focused tests for the real Chronos-2 pretrained challenger wrapper."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import pandas as pd
import pytest

from app.inference.service import load_model_artifact
from app.training.dataset import DatasetSample, SourceFeatureRow, build_sequence_context_rows
from app.training.pretrained_forecasters import (
    DEFAULT_CHRONOS_2_PRETRAINED_SOURCE,
    DEFAULT_SCOPE_REGIMES,
    GENERALIST,
    HF_REFERENCE_PRETRAINED_ARTIFACT_FORMAT,
    MODEL_FAMILY_AMAZON_CHRONOS_2,
    build_chronos2_classifier,
    validate_pretrained_forecaster_contract,
)


class _FakeChronos2Pipeline:
    """Tiny Chronos-2 runtime double that behaves like predict_df()."""

    loads: list[tuple[str, dict[str, object]]] = []

    def __init__(self, pretrained_source: str, load_kwargs: dict[str, object]) -> None:
        self.pretrained_source = pretrained_source
        self.load_kwargs = dict(load_kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_source: str,
        **load_kwargs: object,
    ) -> "_FakeChronos2Pipeline":
        cls.loads.append((pretrained_source, dict(load_kwargs)))
        return cls(pretrained_source, dict(load_kwargs))

    def predict_df(
        self,
        context_df: pd.DataFrame,
        *,
        prediction_length: int,
        quantile_levels: list[float],
        id_column: str,
        timestamp_column: str,
        target: str,
    ) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for series_id, group in context_df.groupby(id_column):
            ordered = group.sort_values(timestamp_column)
            last_row = ordered.iloc[-1]
            current_close = float(last_row[target])
            momentum = float(last_row.get("momentum_3", 0.0))
            realized_vol = float(last_row.get("realized_vol_12", 0.0))
            raw_return = (momentum * 6.0) - (realized_vol * 0.2)
            forecast_value = current_close * (1.0 + raw_return)
            forecast_timestamp = pd.Timestamp(last_row[timestamp_column]) + pd.Timedelta(
                minutes=5 * prediction_length
            )
            record: dict[str, object] = {
                id_column: str(series_id),
                timestamp_column: forecast_timestamp,
                "predictions": forecast_value,
            }
            for level in quantile_levels:
                record[str(level)] = forecast_value
            records.append(record)
        return pd.DataFrame.from_records(records)


def _source_rows() -> list[SourceFeatureRow]:
    start = datetime(2026, 4, 3, 0, 0, tzinfo=timezone.utc)
    momentum_by_offset = {
        0: -0.030,
        1: -0.020,
        2: -0.015,
        3: -0.010,
        4: 0.018,
        5: 0.022,
        6: 0.025,
    }
    rows: list[SourceFeatureRow] = []
    for offset in range(7):
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
                    "realized_vol_12": 0.01 + (0.002 * offset),
                    "momentum_3": momentum_by_offset[offset],
                },
            )
        )
    return rows


def _samples(source_rows: list[SourceFeatureRow]) -> list[DatasetSample]:
    labels_by_offset = {
        3: 0,
        4: 1,
        5: 1,
    }
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


def test_chronos2_wrapper_fits_and_emits_truthful_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chronos-2 should load through the shared wrapper and expose Packet 2 metadata."""
    from app.training import pretrained_forecasters as pretrained_module

    _FakeChronos2Pipeline.loads.clear()
    monkeypatch.setattr(
        pretrained_module,
        "_import_chronos_runtime",
        lambda: _FakeChronos2Pipeline,
    )

    source_rows = _source_rows()
    samples = _samples(source_rows)
    wrapper = build_chronos2_classifier(
        {
            "context_lookback_candles": 3,
            "calibration_windows": 3,
            "prediction_batch_size": 2,
            "covariate_columns": ["realized_vol_12", "momentum_3"],
            "device_map": "cpu",
        }
    )

    wrapper.fit_samples(samples, source_rows=source_rows)
    contract = validate_pretrained_forecaster_contract(wrapper)
    probabilities = wrapper.predict_proba_samples(samples, source_rows=source_rows)

    assert _FakeChronos2Pipeline.loads == [
        (
            DEFAULT_CHRONOS_2_PRETRAINED_SOURCE,
            {"device_map": "cpu", "torch_dtype": "auto"},
        )
    ]
    assert contract.registry_metadata["model_family"] == MODEL_FAMILY_AMAZON_CHRONOS_2
    assert contract.registry_metadata["candidate_role"] == GENERALIST
    assert contract.registry_metadata["scope_regimes"] == list(DEFAULT_SCOPE_REGIMES)
    assert contract.registry_metadata["artifact_format"] == HF_REFERENCE_PRETRAINED_ARTIFACT_FORMAT
    assert contract.registry_metadata["pretrained_source"] == DEFAULT_CHRONOS_2_PRETRAINED_SOURCE
    assert contract.registry_metadata["license_name"] == "Apache-2.0"
    assert contract.training_config["resolved_covariate_columns"] == [
        "realized_vol_12",
        "momentum_3",
    ]
    assert contract.requires_sequence_context is True
    assert contract.sequence_lookback_candles == 3
    assert probabilities[0][1] < probabilities[1][1] < probabilities[2][1]


def test_chronos2_artifact_reloads_through_saved_model_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chronos-2 should persist as a joblib artifact that lazily reloads the HF pipeline."""
    from app.training import pretrained_forecasters as pretrained_module

    _FakeChronos2Pipeline.loads.clear()
    monkeypatch.setattr(
        pretrained_module,
        "_import_chronos_runtime",
        lambda: _FakeChronos2Pipeline,
    )

    source_rows = _source_rows()
    samples = _samples(source_rows)
    wrapper = build_chronos2_classifier(
        {
            "context_lookback_candles": 3,
            "calibration_windows": 3,
            "prediction_batch_size": 2,
            "device_map": "cpu",
        }
    )
    wrapper.fit_samples(samples, source_rows=source_rows)
    contract = validate_pretrained_forecaster_contract(wrapper)

    run_dir = tmp_path / "artifacts" / "training" / "m20" / "20260403T180000Z"
    run_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = run_dir / "model.joblib"
    joblib.dump(
        {
            "model_name": "chronos2_generalist",
            "trained_at": "2026-04-03T18:00:00Z",
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

    assert loaded_artifact.model_name == "chronos2_generalist"
    assert loaded_artifact.model_version == "20260403T180000Z"
    assert loaded_artifact.feature_columns == tuple(contract.feature_columns)
    assert probabilities[0][1] > 0.5
    assert len(_FakeChronos2Pipeline.loads) == 2
    assert _FakeChronos2Pipeline.loads[1] == (
        DEFAULT_CHRONOS_2_PRETRAINED_SOURCE,
        {"device_map": "cpu", "torch_dtype": "auto"},
    )
