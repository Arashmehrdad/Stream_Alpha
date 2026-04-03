"""Shared artifact-scoring helpers for flat and lookback-aware model families."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.training.dataset import (
    SEQUENCE_CONTEXT_KEY,
    DatasetSample,
    SourceFeatureRow,
    build_sequence_context_rows,
)


class InsufficientSequenceHistoryError(ValueError):
    """Raised when a sequence artifact cannot be scored honestly yet."""


def model_requires_sequence_context(model: Any) -> bool:
    """Return whether the saved artifact needs ordered lookback history."""
    if not hasattr(model, "requires_sequence_context"):
        return False
    return bool(model.requires_sequence_context())


def required_sequence_lookback_candles(model: Any) -> int | None:
    """Return the required lookback candles for sequence artifacts."""
    if not model_requires_sequence_context(model):
        return None
    if not hasattr(model, "get_sequence_lookback_candles"):
        raise ValueError(
            "Sequence model artifacts must expose get_sequence_lookback_candles()",
        )
    lookback_candles = int(model.get_sequence_lookback_candles())
    if lookback_candles <= 0:
        raise ValueError("Sequence model lookback candles must be positive")
    return lookback_candles


def build_scoring_rows_for_runtime_row(
    *,
    model: Any,
    feature_columns: tuple[str, ...],
    row: dict[str, Any],
    history_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build the exact scoring rows expected by one saved artifact."""
    if not model_requires_sequence_context(model):
        return [_build_flat_feature_input(feature_columns=feature_columns, row=row)]

    lookback_candles = required_sequence_lookback_candles(model)
    assert lookback_candles is not None
    if history_rows is None:
        raise InsufficientSequenceHistoryError(
            "Sequence model scoring requires ordered feature history rows",
        )

    target_row = _build_flat_feature_input(feature_columns=feature_columns, row=row)
    symbol = str(row["symbol"])
    as_of_time = _require_datetime(value=row.get("as_of_time"), field_name="as_of_time")

    ordered_history = _ordered_runtime_history_rows(
        history_rows=history_rows,
        feature_columns=feature_columns,
        symbol=symbol,
        as_of_time=as_of_time,
        current_row=row,
    )
    if len(ordered_history) < lookback_candles:
        raise InsufficientSequenceHistoryError(
            "Sequence artifact cannot score the current row honestly: "
            f"required {lookback_candles} lookback rows, found {len(ordered_history)}",
        )
    return [
        {
            **target_row,
            "symbol": symbol,
            "as_of_time": as_of_time,
            SEQUENCE_CONTEXT_KEY: ordered_history[-lookback_candles:],
        }
    ]


def build_scoring_rows_for_dataset_sample(
    *,
    model: Any,
    feature_columns: tuple[str, ...],
    sample: DatasetSample,
    source_rows: tuple[SourceFeatureRow, ...] | list[SourceFeatureRow],
) -> list[dict[str, Any]]:
    """Build the exact scoring rows expected for one offline evaluation sample."""
    if not model_requires_sequence_context(model):
        return [_build_sample_feature_input(feature_columns=feature_columns, sample=sample)]

    lookback_candles = required_sequence_lookback_candles(model)
    assert lookback_candles is not None
    return build_sequence_context_rows(
        target_samples=[sample],
        source_rows=source_rows,
        feature_columns=feature_columns,
        lookback_candles=lookback_candles,
    )


def sample_has_required_sequence_history(
    *,
    sample: DatasetSample,
    source_rows: tuple[SourceFeatureRow, ...] | list[SourceFeatureRow],
    lookback_candles: int,
) -> bool:
    """Return whether one sample has enough same-symbol history for sequence scoring."""
    available_rows = [
        source_row
        for source_row in source_rows
        if source_row.symbol == sample.symbol and source_row.as_of_time <= sample.as_of_time
    ]
    return len(available_rows) >= lookback_candles


def score_binary_probabilities(
    *,
    model: Any,
    scoring_rows: list[dict[str, Any]],
) -> list[tuple[float, float]]:
    """Score one or more rows through the shared binary-probability artifact contract."""
    probabilities = model.predict_proba(scoring_rows)
    if len(probabilities) != len(scoring_rows):
        raise ValueError(
            "Model predict_proba must return one binary probability row per input row",
        )

    resolved_rows: list[tuple[float, float]] = []
    for probability_row in probabilities:
        if len(probability_row) != 2:
            raise ValueError("Model predict_proba must return binary probabilities")
        resolved_rows.append((float(probability_row[0]), float(probability_row[1])))
    return resolved_rows


def _build_flat_feature_input(
    *,
    feature_columns: tuple[str, ...],
    row: dict[str, Any],
) -> dict[str, Any]:
    missing_columns = [
        column
        for column in feature_columns
        if column not in row or row[column] is None
    ]
    if missing_columns:
        raise ValueError(
            f"Scoring row is missing required model columns: {missing_columns}",
        )
    return {column: row[column] for column in feature_columns}


def _build_sample_feature_input(
    *,
    feature_columns: tuple[str, ...],
    sample: DatasetSample,
) -> dict[str, Any]:
    feature_input: dict[str, Any] = {}
    for column in feature_columns:
        if column == "symbol":
            feature_input[column] = sample.symbol
        elif column == "close_price":
            feature_input[column] = sample.close_price
        else:
            if column not in sample.features:
                raise ValueError(
                    f"Training sample {sample.row_id} is missing feature {column}",
                )
            feature_input[column] = sample.features[column]
    return feature_input


def _ordered_runtime_history_rows(
    *,
    history_rows: list[dict[str, Any]],
    feature_columns: tuple[str, ...],
    symbol: str,
    as_of_time: datetime,
    current_row: dict[str, Any],
) -> list[dict[str, Any]]:
    combined_rows = [
        history_row
        for history_row in history_rows
        if str(history_row.get("symbol")) == symbol
        and _require_datetime(
            value=history_row.get("as_of_time"),
            field_name="as_of_time",
        )
        <= as_of_time
    ]
    if not any(
        _require_datetime(value=row.get("as_of_time"), field_name="as_of_time") == as_of_time
        for row in combined_rows
    ):
        combined_rows.append(current_row)

    deduplicated_rows: dict[tuple[datetime, datetime], dict[str, Any]] = {}
    for row in combined_rows:
        row_as_of_time = _require_datetime(value=row.get("as_of_time"), field_name="as_of_time")
        interval_begin = _require_datetime(
            value=row.get("interval_begin", row_as_of_time),
            field_name="interval_begin",
        )
        deduplicated_rows[(row_as_of_time, interval_begin)] = row

    ordered_rows = sorted(
        deduplicated_rows.values(),
        key=lambda row: (
            _require_datetime(value=row.get("as_of_time"), field_name="as_of_time"),
            _require_datetime(
                value=row.get("interval_begin", row["as_of_time"]),
                field_name="interval_begin",
            ),
        ),
    )
    return [
        {
            "symbol": symbol,
            "as_of_time": _require_datetime(
                value=row.get("as_of_time"),
                field_name="as_of_time",
            ),
            **_build_flat_feature_input(feature_columns=feature_columns, row=row),
        }
        for row in ordered_rows
    ]


def _require_datetime(*, value: Any, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"Scoring row field {field_name} must be a datetime")
    return value
