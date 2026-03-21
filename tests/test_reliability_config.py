"""Config-loader tests for the M13 reliability foundation."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.reliability.config import load_reliability_config


def test_checked_in_reliability_config_loads() -> None:
    """The checked-in reliability config should load into typed settings."""
    config = load_reliability_config(Path("configs/reliability.yaml"))

    assert config.schema_version == "m13_reliability_v1"
    assert config.freshness.feed_max_age_seconds == 90
    assert config.freshness.feature_max_age_seconds == 390
    assert config.heartbeat.stale_after_seconds == 45
    assert config.circuit_breaker.failure_threshold == 3
    assert config.recovery.stale_pending_signal_max_age_intervals == 1
    assert config.artifacts.health_snapshot_path.endswith("health_snapshot.json")


def test_invalid_reliability_config_is_rejected(tmp_path: Path) -> None:
    """Invalid checked-in values should fail fast with a clear message."""
    config_path = tmp_path / "reliability.yaml"
    config_path.write_text(
        "\n".join(
            [
                "schema_version: m13_reliability_v1",
                "freshness:",
                "  feed_max_age_seconds: 0",
                "  feature_max_age_seconds: 120",
                "  regime_max_age_seconds: 86400",
                "heartbeat:",
                "  write_interval_seconds: 15",
                "  stale_after_seconds: 45",
                "circuit_breaker:",
                "  failure_threshold: 3",
                "  half_open_after_seconds: 60",
                "  success_threshold: 1",
                "recovery:",
                "  stale_pending_signal_max_age_intervals: 1",
                "artifacts:",
                "  health_snapshot_path: artifacts/reliability/health_snapshot.json",
                "  freshness_summary_path: artifacts/reliability/freshness_summary.json",
                "  recovery_events_path: artifacts/reliability/recovery_events.jsonl",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="feed_max_age_seconds must be positive"):
        load_reliability_config(config_path)
