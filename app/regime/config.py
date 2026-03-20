"""Configuration loading for the M8 offline regime workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.common.serialization import make_json_safe


def _deduplicate(items: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _require_non_empty_string(value: Any, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"Regime config field {field_name} cannot be empty")
    return text


def _require_positive_int(value: Any, field_name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"Regime config field {field_name} must be greater than zero")
    return parsed


def _require_percentile(value: Any, field_name: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 100.0:
        raise ValueError(
            f"Regime config field {field_name} must be within (0, 100]"
        )
    return parsed


@dataclass(frozen=True, slots=True)
class ThresholdConfig:
    """Percentile configuration for per-symbol regime thresholds."""

    high_vol_percentile: float
    trend_abs_momentum_percentile: float

    @property
    def high_vol_fraction(self) -> float:
        """Return the configured volatility percentile as a [0, 1] fraction."""
        return self.high_vol_percentile / 100.0

    @property
    def trend_abs_momentum_fraction(self) -> float:
        """Return the configured momentum percentile as a [0, 1] fraction."""
        return self.trend_abs_momentum_percentile / 100.0


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    """Checked-in configuration for the explicit offline regime run."""

    source_table: str
    source_exchange: str
    interval_minutes: int
    symbols: tuple[str, ...]
    artifact_dir: str
    min_rows_per_symbol: int
    thresholds: ThresholdConfig

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary form for artifact persistence."""
        return make_json_safe(
            {
                "source_table": self.source_table,
                "source_exchange": self.source_exchange,
                "interval_minutes": self.interval_minutes,
                "symbols": list(self.symbols),
                "artifact_dir": self.artifact_dir,
                "min_rows_per_symbol": self.min_rows_per_symbol,
                "thresholds": {
                    "high_vol_percentile": self.thresholds.high_vol_percentile,
                    "trend_abs_momentum_percentile": (
                        self.thresholds.trend_abs_momentum_percentile
                    ),
                },
            }
        )


def load_regime_config(config_path: Path) -> RegimeConfig:
    """Load and validate the checked-in JSON config for one regime run."""
    resolved_config_path = Path(config_path).resolve()
    config_data = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    symbols = _deduplicate(
        [
            _require_non_empty_string(symbol, "symbols")
            for symbol in list(config_data["symbols"])
        ]
    )
    if not symbols:
        raise ValueError("Regime config field symbols must contain at least one symbol")

    return RegimeConfig(
        source_table=_require_non_empty_string(
            config_data["source_table"],
            "source_table",
        ),
        source_exchange=_require_non_empty_string(
            config_data["source_exchange"],
            "source_exchange",
        ),
        interval_minutes=_require_positive_int(
            config_data["interval_minutes"],
            "interval_minutes",
        ),
        symbols=symbols,
        artifact_dir=_require_non_empty_string(
            config_data["artifact_dir"],
            "artifact_dir",
        ),
        min_rows_per_symbol=_require_positive_int(
            config_data["min_rows_per_symbol"],
            "min_rows_per_symbol",
        ),
        thresholds=ThresholdConfig(
            high_vol_percentile=_require_percentile(
                config_data["thresholds"]["high_vol_percentile"],
                "thresholds.high_vol_percentile",
            ),
            trend_abs_momentum_percentile=_require_percentile(
                config_data["thresholds"]["trend_abs_momentum_percentile"],
                "thresholds.trend_abs_momentum_percentile",
            ),
        ),
    )
