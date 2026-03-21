"""Typed checked-in reliability configuration for M13."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class FreshnessConfig:
    """Explicit freshness thresholds for core reliability checks."""

    feed_max_age_seconds: int
    feature_max_age_seconds: int
    regime_max_age_seconds: int


@dataclass(frozen=True, slots=True)
class HeartbeatConfig:
    """Service heartbeat cadence and stale threshold settings."""

    write_interval_seconds: int
    stale_after_seconds: int


@dataclass(frozen=True, slots=True)
class CircuitBreakerConfig:
    """Simple circuit-breaker thresholds for future recovery wiring."""

    failure_threshold: int
    half_open_after_seconds: int
    success_threshold: int


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    """Explicit recovery primitives kept local-first and inspectable."""

    stale_pending_signal_max_age_intervals: int


@dataclass(frozen=True, slots=True)
class ReliabilityConfig:
    """Full checked-in M13 reliability configuration."""

    schema_version: str
    freshness: FreshnessConfig
    heartbeat: HeartbeatConfig
    circuit_breaker: CircuitBreakerConfig
    recovery: RecoveryConfig


def load_reliability_config(config_path: Path) -> ReliabilityConfig:
    """Load the checked-in reliability foundation config."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Reliability config must deserialize into a mapping")

    freshness_payload = _require_mapping(payload, "freshness")
    heartbeat_payload = _require_mapping(payload, "heartbeat")
    breaker_payload = _require_mapping(payload, "circuit_breaker")
    recovery_payload = _require_mapping(payload, "recovery")

    config = ReliabilityConfig(
        schema_version=str(payload.get("schema_version", "")).strip(),
        freshness=FreshnessConfig(
            feed_max_age_seconds=int(freshness_payload["feed_max_age_seconds"]),
            feature_max_age_seconds=int(freshness_payload["feature_max_age_seconds"]),
            regime_max_age_seconds=int(freshness_payload["regime_max_age_seconds"]),
        ),
        heartbeat=HeartbeatConfig(
            write_interval_seconds=int(heartbeat_payload["write_interval_seconds"]),
            stale_after_seconds=int(heartbeat_payload["stale_after_seconds"]),
        ),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=int(breaker_payload["failure_threshold"]),
            half_open_after_seconds=int(breaker_payload["half_open_after_seconds"]),
            success_threshold=int(breaker_payload["success_threshold"]),
        ),
        recovery=RecoveryConfig(
            stale_pending_signal_max_age_intervals=int(
                recovery_payload["stale_pending_signal_max_age_intervals"]
            ),
        ),
    )
    _validate_config(config)
    return config


def _require_mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Reliability config section '{key}' must be a mapping")
    return value


def _validate_config(config: ReliabilityConfig) -> None:
    if not config.schema_version:
        raise ValueError("schema_version must not be empty")
    if config.freshness.feed_max_age_seconds <= 0:
        raise ValueError("freshness.feed_max_age_seconds must be positive")
    if config.freshness.feature_max_age_seconds <= 0:
        raise ValueError("freshness.feature_max_age_seconds must be positive")
    if config.freshness.regime_max_age_seconds <= 0:
        raise ValueError("freshness.regime_max_age_seconds must be positive")
    if config.heartbeat.write_interval_seconds <= 0:
        raise ValueError("heartbeat.write_interval_seconds must be positive")
    if config.heartbeat.stale_after_seconds <= 0:
        raise ValueError("heartbeat.stale_after_seconds must be positive")
    if config.circuit_breaker.failure_threshold <= 0:
        raise ValueError("circuit_breaker.failure_threshold must be positive")
    if config.circuit_breaker.half_open_after_seconds <= 0:
        raise ValueError("circuit_breaker.half_open_after_seconds must be positive")
    if config.circuit_breaker.success_threshold <= 0:
        raise ValueError("circuit_breaker.success_threshold must be positive")
    if config.recovery.stale_pending_signal_max_age_intervals <= 0:
        raise ValueError(
            "recovery.stale_pending_signal_max_age_intervals must be positive"
        )
