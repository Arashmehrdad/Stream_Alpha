"""Tests for the M13 reliability and recovery audit artifact writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.reliability.m13_reliability_recovery_audit import (
    audit_m13_reliability_recovery,
)


def test_audit_reports_consolidated_when_required_surfaces_exist(tmp_path: Path) -> None:
    """A complete fixture should classify M13 as consolidated."""
    _write_complete_fixture(tmp_path)

    result = audit_m13_reliability_recovery(repo_root=tmp_path)

    assert result["m13_state"] == "M13_RELIABILITY_RECOVERY_CONTROLS_CONSOLIDATED"
    assert result["gap_count"] == 0
    assert result["recommendation"] == "PROCEED_TO_M14_EXPLAINABILITY_AUDIT"
    assert (
        result["next_required_action"]
        == "AUDIT_M14_EXPLAINABILITY_AND_DECISION_TRACE_CONTROLS"
    )
    assert "M20_PAUSED" in result["honesty_flags"]
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]


def test_audit_writes_expected_artifacts(tmp_path: Path) -> None:
    """The audit should persist its deterministic artifact contract."""
    _write_complete_fixture(tmp_path)

    result = audit_m13_reliability_recovery(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    expected_files = {
        "manifest.json",
        "m13_reliability_recovery_audit.json",
        "m13_reliability_recovery_audit.md",
        "reliability_surface_audit.csv",
        "freshness_recovery_audit.csv",
        "persistence_audit.csv",
        "runtime_integration_audit.csv",
        "artifact_status_audit.csv",
        "gap_analysis.csv",
        "next_actions.csv",
        "recommendation.json",
    }
    assert expected_files == {path.name for path in output_dir.iterdir()}
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    assert recommendation["runtime_ready"] is False
    assert recommendation["promotable"] is False
    assert recommendation["profitability_claim"] is False


def test_missing_feature_lag_evaluator_produces_partial_state(tmp_path: Path) -> None:
    """Missing feature lag evaluation should be a blocking M13 gap."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/reliability/service.py",
        _service_fixture().replace("def evaluate_feature_consumer_lag", ""),
    )

    result = audit_m13_reliability_recovery(repo_root=tmp_path)

    assert result["m13_state"] == "M13_RELIABILITY_RECOVERY_CONTROLS_PARTIAL"
    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "feature_consumer_lag_evaluator" in gap_names
    assert result["recommendation"] == "IMPLEMENT_REUSABLE_M13_RELIABILITY_RECOVERY_GAP_FILLS"


def test_missing_reliability_event_persistence_is_reported(tmp_path: Path) -> None:
    """Recovery event persistence is required for M13 recovery truth."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/reliability/store.py",
        _store_fixture().replace("insert_reliability_event", "insert_event_missing"),
    )

    result = audit_m13_reliability_recovery(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "recovery_event_persistence" in gap_names


def test_missing_system_reliability_endpoint_is_reported(tmp_path: Path) -> None:
    """The canonical system reliability endpoint is required for M13 read surfaces."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/inference/main.py",
        '@app.get("/freshness")\ndef x():\n    pass\n',
    )

    result = audit_m13_reliability_recovery(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "system_reliability_endpoint" in gap_names


def test_outputs_preserve_m20_pause_and_no_authority_change(tmp_path: Path) -> None:
    """M13 audit outputs should not reopen M20 or claim runtime changes."""
    _write_complete_fixture(tmp_path)

    result = audit_m13_reliability_recovery(repo_root=tmp_path)
    surface_path = Path(result["output_files"]["reliability_surface_audit_csv"])
    with surface_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert rows
    assert {row["runtime_authority_changed"] for row in rows} == {"False"}
    assert {row["m20_reopened"] for row in rows} == {"False"}
    assert result["runtime_status"] == "NO_RUNTIME_EFFECT"
    assert result["promotion_status"] == "NOT_PROMOTABLE"
    assert result["profitability_status"] == "NO_PROFIT_CLAIM"


def _write_complete_fixture(root: Path) -> None:
    _write(
        root / "configs/reliability.yaml",
        "schema_version: m13_reliability_v1\n"
        "health_snapshot_path\n"
        "freshness_summary_path\n"
        "recovery_events_path\n"
        "system_health_path\n"
        "lag_summary_path\n",
    )
    _write(root / "app/reliability/config.py", "class ReliabilityConfig:\n    pass\n")
    _write(root / "app/reliability/schemas.py", _schemas_fixture())
    _write(root / "app/reliability/service.py", _service_fixture())
    _write(root / "app/reliability/store.py", _store_fixture())
    _write(
        root / "app/reliability/artifacts.py",
        "def write_json_artifact():\n    pass\n"
        "def append_jsonl_artifact():\n    pass\n",
    )
    _write(
        root / "app/inference/main.py",
        '@app.get("/freshness")\ndef freshness():\n    pass\n'
        '@app.get("/reliability/system")\ndef reliability():\n    pass\n',
    )
    _write(
        root / "app/inference/service.py",
        "async def system_reliability_snapshot():\n    pass\n"
        "def _write_system_reliability_artifact():\n    pass\n"
        "def _write_freshness_artifact():\n    pass\n",
    )
    _write(
        root / "app/trading/signal_client.py",
        "async def fetch_system_reliability():\n    pass\n",
    )
    _write(root / "app/trading/runner.py", _runner_fixture())
    _write(
        root / "app/trading/repository.py",
        "def load_latest_service_heartbeat():\n    pass\n",
    )
    _write(root / "app/trading/live.py", "def apply_live_health_gate():\n    pass\n")
    _write(root / "README.md", "M12_GUARDED_LIVE_CONTROLS_CONSOLIDATED\n")
    _write(root / "docs/training.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")
    _write(root / "PLANS.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")


def _schemas_fixture() -> str:
    return (
        "class FreshnessStatus:\n    pass\n"
        "class ServiceHeartbeat:\n    pass\n"
        "class ReliabilityState:\n    pass\n"
        "class RecoveryEvent:\n    pass\n"
        "class FeatureLagSnapshot:\n    pass\n"
        "class SystemReliabilitySnapshot:\n    pass\n"
    )


def _service_fixture() -> str:
    return (
        "FEED_STALE = 'FEED_STALE'\n"
        "FEATURE_STALE = 'FEATURE_STALE'\n"
        "HEARTBEAT_MISSING = 'HEARTBEAT_MISSING'\n"
        "REGIME_ROW_INCOMPATIBLE = 'REGIME_ROW_INCOMPATIBLE'\n"
        "FEATURE_LAG_BREACH = 'FEATURE_LAG_BREACH'\n"
        "BREAKER_OPENED = 'BREAKER_OPENED'\n"
        "BREAKER_HALF_OPENED = 'BREAKER_HALF_OPENED'\n"
        "BREAKER_RESTORED = 'BREAKER_RESTORED'\n"
        "RECOVERY_STALE_PENDING_SIGNAL_CLEARED = 'RECOVERY_STALE_PENDING_SIGNAL_CLEARED'\n"
        "def evaluate_feed_freshness():\n    pass\n"
        "def evaluate_feature_freshness():\n    pass\n"
        "def evaluate_heartbeat_freshness():\n    pass\n"
        "def evaluate_regime_freshness():\n    pass\n"
        "def evaluate_feature_consumer_lag():\n    pass\n"
        "def aggregate_system_reliability():\n    pass\n"
        "def transition_circuit_breaker():\n    pass\n"
        "def evaluate_pending_signal_expiry():\n    pass\n"
    )


def _store_fixture() -> str:
    return (
        "service_heartbeats\n"
        "reliability_state\n"
        "reliability_events\n"
        "reliability_lag_state\n"
        "reliability_system_state\n"
        "class ReliabilityStore:\n    pass\n"
        "def save_service_heartbeat():\n    pass\n"
        "def save_reliability_state():\n    pass\n"
        "def insert_reliability_event():\n    pass\n"
        "def save_feature_lag_state():\n    pass\n"
        "def save_system_reliability_state():\n    pass\n"
    )


def _runner_fixture() -> str:
    return (
        "RecoveryEvent\n"
        "SIGNAL_FETCH_SKIPPED_BREAKER_OPEN\n"
        "def _expire_stale_pending_signals():\n    pass\n"
        "def _write_runner_heartbeat():\n    pass\n"
        "def _record_reliability_event():\n    pass\n"
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
