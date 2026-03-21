"""Deterministic threshold fitting and artifact orchestration for M8."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

from app.common.time import to_rfc3339, utc_now
from app.regime.artifacts import (
    required_run_artifact_paths,
    write_csv,
    write_json_atomic,
    write_thresholds_artifact,
)
from app.regime.config import RegimeConfig, load_regime_config
from app.regime.dataset import RegimeSourceRow, load_regime_dataset


TREND_UP = "TREND_UP"
TREND_DOWN = "TREND_DOWN"
RANGE = "RANGE"
HIGH_VOL = "HIGH_VOL"

REGIME_LABELS = (TREND_UP, TREND_DOWN, RANGE, HIGH_VOL)


@dataclass(frozen=True, slots=True)
class SymbolThresholds:
    """Per-symbol thresholds fit from canonical historical rows."""

    symbol: str
    fitted_row_count: int
    high_vol_threshold: float
    trend_abs_threshold: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary form for artifact persistence."""
        return {
            "symbol": self.symbol,
            "fitted_row_count": self.fitted_row_count,
            "high_vol_threshold": self.high_vol_threshold,
            "trend_abs_threshold": self.trend_abs_threshold,
        }


@dataclass(frozen=True, slots=True)
class RegimePrediction:
    """One deterministic regime label for one canonical feature row."""

    symbol: str
    interval_begin: datetime
    as_of_time: datetime
    realized_vol_12: float
    momentum_3: float
    macd_line_12_26: float
    regime: str

    def to_csv_row(self) -> dict[str, Any]:
        """Return a CSV-friendly row for artifact persistence."""
        return {
            "symbol": self.symbol,
            "interval_begin": to_rfc3339(self.interval_begin),
            "as_of_time": to_rfc3339(self.as_of_time),
            "realized_vol_12": self.realized_vol_12,
            "momentum_3": self.momentum_3,
            "macd_line_12_26": self.macd_line_12_26,
            "regime": self.regime,
        }


def run_regime_workflow(config_path: Path) -> Path:
    """Run the explicit offline regime workflow and write all required artifacts."""
    resolved_config_path = Path(config_path).resolve()
    config = load_regime_config(resolved_config_path)
    dataset = load_regime_dataset(config)
    thresholds_by_symbol = fit_symbol_thresholds(dataset.rows, config)
    predictions = classify_rows(dataset.rows, thresholds_by_symbol)

    created_at = utc_now()
    run_id = created_at.strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = Path(config.artifact_dir).resolve() / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)

    write_json_atomic(artifact_dir / "run_config.json", config.to_dict())

    thresholds_payload = build_thresholds_payload(
        run_id=run_id,
        created_at=created_at,
        config=config,
        thresholds_by_symbol=thresholds_by_symbol,
    )
    write_thresholds_artifact(artifact_dir / "thresholds.json", thresholds_payload)

    prediction_rows = [prediction.to_csv_row() for prediction in predictions]
    write_csv(artifact_dir / "regime_predictions.csv", prediction_rows)

    by_symbol_summary_rows = build_by_symbol_summary(predictions, thresholds_by_symbol)
    write_csv(artifact_dir / "by_symbol_summary.csv", by_symbol_summary_rows)

    overall_summary = build_overall_summary(
        created_at=created_at,
        run_id=run_id,
        config=config,
        dataset_rows=dataset.rows,
        row_counts_by_symbol=dataset.row_counts_by_symbol,
        predictions=predictions,
    )
    write_json_atomic(artifact_dir / "overall_summary.json", overall_summary)

    run_manifest = build_run_manifest(
        artifact_dir=artifact_dir,
        created_at=created_at,
        run_id=run_id,
        config=config,
        row_counts_by_symbol=dataset.row_counts_by_symbol,
        total_rows=len(dataset.rows),
    )
    write_json_atomic(artifact_dir / "run_manifest.json", run_manifest)
    required_run_artifact_paths(artifact_dir)
    return artifact_dir


def fit_symbol_thresholds(
    rows: Sequence[RegimeSourceRow],
    config: RegimeConfig,
) -> dict[str, SymbolThresholds]:
    """Fit deterministic per-symbol thresholds from canonical feature rows."""
    rows_by_symbol: dict[str, list[RegimeSourceRow]] = {symbol: [] for symbol in config.symbols}
    for row in rows:
        rows_by_symbol.setdefault(row.symbol, []).append(row)

    thresholds_by_symbol: dict[str, SymbolThresholds] = {}
    for symbol in config.symbols:
        symbol_rows = sorted(
            rows_by_symbol.get(symbol, []),
            key=lambda row: (row.interval_begin, row.as_of_time),
        )
        if len(symbol_rows) < config.min_rows_per_symbol:
            raise ValueError(
                "Regime source does not contain enough canonical rows per symbol. "
                f"Required at least {config.min_rows_per_symbol} rows for {symbol}, "
                f"found {len(symbol_rows)}"
            )
        thresholds_by_symbol[symbol] = SymbolThresholds(
            symbol=symbol,
            fitted_row_count=len(symbol_rows),
            high_vol_threshold=compute_percentile(
                [row.realized_vol_12 for row in symbol_rows],
                config.thresholds.high_vol_percentile,
            ),
            trend_abs_threshold=compute_percentile(
                [abs(row.momentum_3) for row in symbol_rows],
                config.thresholds.trend_abs_momentum_percentile,
            ),
        )
    return thresholds_by_symbol


def compute_percentile(values: Sequence[float], percentile: float) -> float:
    """Compute a deterministic linear-interpolated percentile."""
    if not values:
        raise ValueError("Cannot compute a percentile from an empty value sequence")
    ordered_values = sorted(float(value) for value in values)
    if len(ordered_values) == 1:
        return ordered_values[0]

    fraction = percentile / 100.0
    rank = (len(ordered_values) - 1) * fraction
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)
    lower_value = ordered_values[lower_index]
    upper_value = ordered_values[upper_index]
    if lower_index == upper_index:
        return lower_value
    weight = rank - lower_index
    return lower_value + ((upper_value - lower_value) * weight)


def classify_rows(
    rows: Sequence[RegimeSourceRow],
    thresholds_by_symbol: dict[str, SymbolThresholds],
) -> list[RegimePrediction]:
    """Classify each canonical row into one explicit regime label."""
    ordered_rows = sorted(
        rows,
        key=lambda row: (row.symbol, row.interval_begin, row.as_of_time),
    )
    predictions: list[RegimePrediction] = []
    for row in ordered_rows:
        regime = classify_row(row, thresholds_by_symbol)
        predictions.append(
            RegimePrediction(
                symbol=row.symbol,
                interval_begin=row.interval_begin,
                as_of_time=row.as_of_time,
                realized_vol_12=row.realized_vol_12,
                momentum_3=row.momentum_3,
                macd_line_12_26=row.macd_line_12_26,
                regime=regime,
            )
        )
    return predictions


def classify_row(
    row: RegimeSourceRow,
    thresholds_by_symbol: dict[str, SymbolThresholds],
) -> str:
    """Apply the explicit threshold-based regime rules to one row."""
    try:
        thresholds = thresholds_by_symbol[row.symbol]
    except KeyError as error:
        raise ValueError(f"No fitted thresholds were available for symbol {row.symbol}") from error

    if row.realized_vol_12 >= thresholds.high_vol_threshold:
        return HIGH_VOL
    if row.momentum_3 >= thresholds.trend_abs_threshold and row.macd_line_12_26 > 0.0:
        return TREND_UP
    if row.momentum_3 <= -thresholds.trend_abs_threshold and row.macd_line_12_26 < 0.0:
        return TREND_DOWN
    return RANGE


def build_thresholds_payload(
    *,
    run_id: str,
    created_at: datetime,
    config: RegimeConfig,
    thresholds_by_symbol: dict[str, SymbolThresholds],
) -> dict[str, Any]:
    """Build the serving-oriented thresholds artifact payload."""
    return {
        "schema_version": "m8_thresholds_v1",
        "created_at": to_rfc3339(created_at),
        "run_id": run_id,
        "source_table": config.source_table,
        "source_exchange": config.source_exchange,
        "interval_minutes": config.interval_minutes,
        "required_inputs": [
            "realized_vol_12",
            "momentum_3",
            "macd_line_12_26",
        ],
        "regime_labels": list(REGIME_LABELS),
        "classification_rule": [
            "if realized_vol_12 >= high_vol_threshold -> HIGH_VOL",
            "else if momentum_3 >= trend_abs_threshold and macd_line_12_26 > 0 -> TREND_UP",
            "else if momentum_3 <= -trend_abs_threshold and macd_line_12_26 < 0 -> TREND_DOWN",
            "else -> RANGE",
        ],
        "percentiles": {
            "high_vol_percentile": config.thresholds.high_vol_percentile,
            "trend_abs_momentum_percentile": (
                config.thresholds.trend_abs_momentum_percentile
            ),
        },
        "thresholds_by_symbol": {
            symbol: thresholds_by_symbol[symbol].to_dict()
            for symbol in sorted(thresholds_by_symbol)
        },
    }


def build_by_symbol_summary(
    predictions: Sequence[RegimePrediction],
    thresholds_by_symbol: dict[str, SymbolThresholds],
) -> list[dict[str, Any]]:
    """Build per-symbol summary rows for the offline artifact set."""
    predictions_by_symbol: dict[str, list[RegimePrediction]] = {
        symbol: [] for symbol in thresholds_by_symbol
    }
    for prediction in predictions:
        predictions_by_symbol.setdefault(prediction.symbol, []).append(prediction)

    summary_rows: list[dict[str, Any]] = []
    for symbol in sorted(thresholds_by_symbol):
        symbol_predictions = sorted(
            predictions_by_symbol.get(symbol, []),
            key=lambda prediction: (prediction.interval_begin, prediction.as_of_time),
        )
        threshold = thresholds_by_symbol[symbol]
        regime_counts = _count_regimes(symbol_predictions)
        first_prediction = symbol_predictions[0]
        last_prediction = symbol_predictions[-1]
        summary_rows.append(
            {
                "symbol": symbol,
                "fitted_row_count": threshold.fitted_row_count,
                "predicted_row_count": len(symbol_predictions),
                "high_vol_threshold": threshold.high_vol_threshold,
                "trend_abs_threshold": threshold.trend_abs_threshold,
                "trend_up_rows": regime_counts[TREND_UP],
                "trend_down_rows": regime_counts[TREND_DOWN],
                "range_rows": regime_counts[RANGE],
                "high_vol_rows": regime_counts[HIGH_VOL],
                "first_interval_begin": to_rfc3339(first_prediction.interval_begin),
                "last_interval_begin": to_rfc3339(last_prediction.interval_begin),
                "first_as_of_time": to_rfc3339(first_prediction.as_of_time),
                "last_as_of_time": to_rfc3339(last_prediction.as_of_time),
            }
        )
    return summary_rows


def build_overall_summary(  # pylint: disable=too-many-arguments
    *,
    created_at: datetime,
    run_id: str,
    config: RegimeConfig,
    dataset_rows: Sequence[RegimeSourceRow],
    row_counts_by_symbol: dict[str, int],
    predictions: Sequence[RegimePrediction],
) -> dict[str, Any]:
    """Build the overall JSON summary artifact for one offline run."""
    regime_counts = _count_regimes(predictions)
    return {
        "created_at": to_rfc3339(created_at),
        "run_id": run_id,
        "source_table": config.source_table,
        "source_exchange": config.source_exchange,
        "interval_minutes": config.interval_minutes,
        "symbols": list(config.symbols),
        "row_counts": {
            "loaded_rows": len(dataset_rows),
            "rows_by_symbol": {
                symbol: row_counts_by_symbol.get(symbol, 0)
                for symbol in config.symbols
            },
        },
        "regime_counts": regime_counts,
        "time_range": _time_range_payload(dataset_rows),
        "threshold_percentiles": {
            "high_vol_percentile": config.thresholds.high_vol_percentile,
            "trend_abs_momentum_percentile": (
                config.thresholds.trend_abs_momentum_percentile
            ),
        },
    }


def build_run_manifest(  # pylint: disable=too-many-arguments
    *,
    artifact_dir: Path,
    created_at: datetime,
    run_id: str,
    config: RegimeConfig,
    row_counts_by_symbol: dict[str, int],
    total_rows: int,
) -> dict[str, Any]:
    """Build the explicit run manifest describing this artifact set."""
    return {
        "run_id": run_id,
        "created_at": to_rfc3339(created_at),
        "source_table": config.source_table,
        "source_exchange": config.source_exchange,
        "symbols": list(config.symbols),
        "interval_minutes": config.interval_minutes,
        "row_counts": {
            "loaded_rows": total_rows,
            "rows_by_symbol": {
                symbol: row_counts_by_symbol.get(symbol, 0)
                for symbol in config.symbols
            },
        },
        "artifact_paths": {
            "thresholds": str((artifact_dir / "thresholds.json").resolve()),
            "regime_predictions": str((artifact_dir / "regime_predictions.csv").resolve()),
            "by_symbol_summary": str((artifact_dir / "by_symbol_summary.csv").resolve()),
            "overall_summary": str((artifact_dir / "overall_summary.json").resolve()),
            "run_config": str((artifact_dir / "run_config.json").resolve()),
            "run_manifest": str((artifact_dir / "run_manifest.json").resolve()),
        },
    }


def _count_regimes(predictions: Sequence[RegimePrediction]) -> dict[str, int]:
    counts = {label: 0 for label in REGIME_LABELS}
    for prediction in predictions:
        counts[prediction.regime] += 1
    return counts


def _time_range_payload(rows: Sequence[RegimeSourceRow]) -> dict[str, str | None]:
    if not rows:
        return {
            "min_interval_begin": None,
            "max_interval_begin": None,
            "min_as_of_time": None,
            "max_as_of_time": None,
        }
    return {
        "min_interval_begin": to_rfc3339(min(row.interval_begin for row in rows)),
        "max_interval_begin": to_rfc3339(max(row.interval_begin for row in rows)),
        "min_as_of_time": to_rfc3339(min(row.as_of_time for row in rows)),
        "max_as_of_time": to_rfc3339(max(row.as_of_time for row in rows)),
    }
