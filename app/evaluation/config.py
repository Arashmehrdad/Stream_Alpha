"""Checked-in M18 evaluation configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def default_evaluation_config_path() -> Path:
    """Return the checked-in evaluation config path from the repo root."""
    return Path(__file__).resolve().parents[2] / "configs" / "evaluation.yaml"


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Explicit M18 evaluation thresholds and notes."""

    schema_version: str
    latency_drift_ms_threshold: float
    fill_price_drift_bps_threshold: float
    slippage_drift_bps_threshold: float
    cost_aware_precision_horizon_notes: str
    minimum_comparable_count_notes: str


def default_evaluation_config() -> EvaluationConfig:
    """Return the code-default M18 evaluation config."""
    return EvaluationConfig(
        schema_version="m18_evaluation_config_v1",
        latency_drift_ms_threshold=25.0,
        fill_price_drift_bps_threshold=10.0,
        slippage_drift_bps_threshold=5.0,
        cost_aware_precision_horizon_notes=(
            "Closed-position realized PnL after fees and slippage, linked back "
            "to the BUY opportunity through entry_decision_trace_id."
        ),
        minimum_comparable_count_notes=(
            "Cost-aware precision returns null when a mode has zero actionable "
            "BUY opportunities with comparable closed outcome truth."
        ),
    )


def load_evaluation_config(config_path: Path) -> EvaluationConfig:
    """Load the checked-in evaluation config."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evaluation config must deserialize into a mapping")
    config = EvaluationConfig(
        schema_version=str(payload.get("schema_version", "")).strip(),
        latency_drift_ms_threshold=float(payload["latency_drift_ms_threshold"]),
        fill_price_drift_bps_threshold=float(payload["fill_price_drift_bps_threshold"]),
        slippage_drift_bps_threshold=float(payload["slippage_drift_bps_threshold"]),
        cost_aware_precision_horizon_notes=str(
            payload["cost_aware_precision_horizon_notes"]
        ).strip(),
        minimum_comparable_count_notes=str(
            payload["minimum_comparable_count_notes"]
        ).strip(),
    )
    _validate_config(config)
    return config


def _validate_config(config: EvaluationConfig) -> None:
    if not config.schema_version:
        raise ValueError("schema_version must not be empty")
    positive_checks = (
        (
            config.latency_drift_ms_threshold,
            "latency_drift_ms_threshold must be positive",
        ),
        (
            config.fill_price_drift_bps_threshold,
            "fill_price_drift_bps_threshold must be positive",
        ),
        (
            config.slippage_drift_bps_threshold,
            "slippage_drift_bps_threshold must be positive",
        ),
    )
    for value, message in positive_checks:
        if value <= 0.0:
            raise ValueError(message)
    if not config.cost_aware_precision_horizon_notes:
        raise ValueError("cost_aware_precision_horizon_notes must not be empty")
    if not config.minimum_comparable_count_notes:
        raise ValueError("minimum_comparable_count_notes must not be empty")
