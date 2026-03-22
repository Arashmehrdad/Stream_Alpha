"""Typed checked-in alerting configuration for M17."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def default_alerting_config_path() -> Path:
    """Return the checked-in alerting config path from the repo root."""
    return Path(__file__).resolve().parents[2] / "configs" / "alerting.yaml"


@dataclass(frozen=True, slots=True)
class OrderFailureSpikeConfig:
    """Thresholds for recent order failure spikes."""

    window_minutes: int
    warning_count: int
    critical_count: int


@dataclass(frozen=True, slots=True)
class SignalAlertConfig:
    """Thresholds for signal silence and flood detection."""

    silence_window_intervals: int
    flood_window_intervals: int
    flood_warning_count: int
    flood_critical_count: int


@dataclass(frozen=True, slots=True)
class AlertingArtifactConfig:
    """Explicit M17 artifact destinations."""

    daily_summary_dir: str
    startup_safety_path: str


@dataclass(frozen=True, slots=True)
class AlertingConfig:
    """Full checked-in M17 alerting configuration."""

    schema_version: str
    order_failure_spike: OrderFailureSpikeConfig
    signals: SignalAlertConfig
    artifacts: AlertingArtifactConfig


def load_alerting_config(config_path: Path) -> AlertingConfig:
    """Load the checked-in alerting configuration."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Alerting config must deserialize into a mapping")

    order_failure_payload = _require_mapping(payload, "order_failure_spike")
    signal_payload = _require_mapping(payload, "signals")
    artifacts_payload = _require_mapping(payload, "artifacts")

    config = AlertingConfig(
        schema_version=str(payload.get("schema_version", "")).strip(),
        order_failure_spike=OrderFailureSpikeConfig(
            window_minutes=int(order_failure_payload["window_minutes"]),
            warning_count=int(order_failure_payload["warning_count"]),
            critical_count=int(order_failure_payload["critical_count"]),
        ),
        signals=SignalAlertConfig(
            silence_window_intervals=int(signal_payload["silence_window_intervals"]),
            flood_window_intervals=int(signal_payload["flood_window_intervals"]),
            flood_warning_count=int(signal_payload["flood_warning_count"]),
            flood_critical_count=int(signal_payload["flood_critical_count"]),
        ),
        artifacts=AlertingArtifactConfig(
            daily_summary_dir=str(artifacts_payload["daily_summary_dir"]).strip(),
            startup_safety_path=str(artifacts_payload["startup_safety_path"]).strip(),
        ),
    )
    _validate_config(config)
    return config


def _require_mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Alerting config section '{key}' must be a mapping")
    return value


def _validate_config(config: AlertingConfig) -> None:
    if not config.schema_version:
        raise ValueError("schema_version must not be empty")
    positive_checks = (
        (
            config.order_failure_spike.window_minutes,
            "order_failure_spike.window_minutes must be positive",
        ),
        (
            config.order_failure_spike.warning_count,
            "order_failure_spike.warning_count must be positive",
        ),
        (
            config.order_failure_spike.critical_count,
            "order_failure_spike.critical_count must be positive",
        ),
        (
            config.signals.silence_window_intervals,
            "signals.silence_window_intervals must be positive",
        ),
        (
            config.signals.flood_window_intervals,
            "signals.flood_window_intervals must be positive",
        ),
        (
            config.signals.flood_warning_count,
            "signals.flood_warning_count must be positive",
        ),
        (
            config.signals.flood_critical_count,
            "signals.flood_critical_count must be positive",
        ),
    )
    for value, message in positive_checks:
        if value <= 0:
            raise ValueError(message)

    if config.order_failure_spike.critical_count < config.order_failure_spike.warning_count:
        raise ValueError(
            "order_failure_spike.critical_count must be greater than or equal to warning_count"
        )
    if config.signals.flood_critical_count < config.signals.flood_warning_count:
        raise ValueError(
            "signals.flood_critical_count must be greater than or equal to flood_warning_count"
        )
    if not config.artifacts.daily_summary_dir:
        raise ValueError("artifacts.daily_summary_dir must not be empty")
    if not config.artifacts.startup_safety_path:
        raise ValueError("artifacts.startup_safety_path must not be empty")
