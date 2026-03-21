"""Typed checked-in reliability configuration for M13."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


def default_reliability_config_path() -> Path:
    """Return the checked-in reliability config path from the repo root."""
    return Path(__file__).resolve().parents[2] / "configs" / "reliability.yaml"


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
class ReliabilityArtifactConfig:
    """Explicit reliability artifact destinations."""

    health_snapshot_path: str
    freshness_summary_path: str
    recovery_events_path: str


@dataclass(frozen=True, slots=True)
class ReliabilityConfig:
    """Full checked-in M13 reliability configuration."""

    schema_version: str
    freshness: FreshnessConfig
    heartbeat: HeartbeatConfig
    circuit_breaker: CircuitBreakerConfig
    recovery: RecoveryConfig
    artifacts: ReliabilityArtifactConfig


def load_reliability_config(config_path: Path) -> ReliabilityConfig:
    """Load the checked-in reliability foundation config."""
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Reliability config must deserialize into a mapping")

    freshness_payload = _require_mapping(payload, "freshness")
    heartbeat_payload = _require_mapping(payload, "heartbeat")
    breaker_payload = _require_mapping(payload, "circuit_breaker")
    recovery_payload = _require_mapping(payload, "recovery")
    artifacts_payload = _require_mapping(payload, "artifacts")

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
        artifacts=ReliabilityArtifactConfig(
            health_snapshot_path=str(artifacts_payload["health_snapshot_path"]).strip(),
            freshness_summary_path=str(
                artifacts_payload["freshness_summary_path"]
            ).strip(),
            recovery_events_path=str(artifacts_payload["recovery_events_path"]).strip(),
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
    positive_checks = (
        (
            config.freshness.feed_max_age_seconds,
            "freshness.feed_max_age_seconds must be positive",
        ),
        (
            config.freshness.feature_max_age_seconds,
            "freshness.feature_max_age_seconds must be positive",
        ),
        (
            config.freshness.regime_max_age_seconds,
            "freshness.regime_max_age_seconds must be positive",
        ),
        (
            config.heartbeat.write_interval_seconds,
            "heartbeat.write_interval_seconds must be positive",
        ),
        (
            config.heartbeat.stale_after_seconds,
            "heartbeat.stale_after_seconds must be positive",
        ),
        (
            config.circuit_breaker.failure_threshold,
            "circuit_breaker.failure_threshold must be positive",
        ),
        (
            config.circuit_breaker.half_open_after_seconds,
            "circuit_breaker.half_open_after_seconds must be positive",
        ),
        (
            config.circuit_breaker.success_threshold,
            "circuit_breaker.success_threshold must be positive",
        ),
        (
            config.recovery.stale_pending_signal_max_age_intervals,
            "recovery.stale_pending_signal_max_age_intervals must be positive",
        ),
    )
    for value, message in positive_checks:
        if value <= 0:
            raise ValueError(message)

    path_checks = (
        (
            config.artifacts.health_snapshot_path,
            "artifacts.health_snapshot_path must not be empty",
        ),
        (
            config.artifacts.freshness_summary_path,
            "artifacts.freshness_summary_path must not be empty",
        ),
        (
            config.artifacts.recovery_events_path,
            "artifacts.recovery_events_path must not be empty",
        ),
    )
    for path_value, message in path_checks:
        if not path_value:
            raise ValueError(message)
