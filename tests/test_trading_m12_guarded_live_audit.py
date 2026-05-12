"""Tests for the M12 guarded-live audit artifact writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.trading.m12_guarded_live_audit import audit_m12_guarded_live


def test_audit_reports_consolidated_when_required_surfaces_exist(tmp_path: Path) -> None:
    """A complete fixture should classify M12 as consolidated."""
    _write_complete_fixture(tmp_path)

    result = audit_m12_guarded_live(repo_root=tmp_path)

    assert result["m12_state"] == "M12_GUARDED_LIVE_CONTROLS_CONSOLIDATED"
    assert result["gap_count"] == 0
    assert result["recommendation"] == "PROCEED_TO_M13_RELIABILITY_RECOVERY_AUDIT"
    assert result["next_required_action"] == "AUDIT_M13_RELIABILITY_AND_RECOVERY_CONTROLS"
    assert "M20_PAUSED" in result["honesty_flags"]
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]


def test_audit_writes_expected_artifacts(tmp_path: Path) -> None:
    """The audit should persist its deterministic artifact contract."""
    _write_complete_fixture(tmp_path)

    result = audit_m12_guarded_live(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    expected_files = {
        "manifest.json",
        "m12_guarded_live_audit.json",
        "m12_guarded_live_audit.md",
        "guarded_live_surface_audit.csv",
        "startup_gate_audit.csv",
        "live_submit_gate_audit.csv",
        "reconciliation_health_audit.csv",
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


def test_missing_startup_validation_produces_partial_state(tmp_path: Path) -> None:
    """Missing startup validation should be a blocking M12 gap."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/trading/live.py",
        _live_fixture().replace("def validate_live_startup", ""),
    )

    result = audit_m12_guarded_live(repo_root=tmp_path)

    assert result["m12_state"] == "M12_GUARDED_LIVE_CONTROLS_PARTIAL"
    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "startup_validation" in gap_names
    assert result["recommendation"] == "IMPLEMENT_REUSABLE_M12_GUARDED_LIVE_GAP_FILLS"


def test_missing_manual_disable_gate_is_reported(tmp_path: Path) -> None:
    """The audit should catch absent manual-disable submit gating."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/trading/live.py",
        _live_fixture().replace("LIVE_MANUAL_DISABLE_ACTIVE", "LIVE_MANUAL_DISABLE_MISSING"),
    )

    result = audit_m12_guarded_live(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "manual_disable_gate" in gap_names
    assert "manual_disable_blocks_submit" in gap_names


def test_missing_live_safety_persistence_is_reported(tmp_path: Path) -> None:
    """Persisted live safety state is required for guarded live controls."""
    _write_complete_fixture(tmp_path)
    _write(tmp_path / "app/trading/repository.py", "def load_live_safety_state():\n    pass\n")

    result = audit_m12_guarded_live(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "live_safety_persistence" in gap_names


def test_outputs_preserve_m20_pause_and_no_authority_change(tmp_path: Path) -> None:
    """M12 audit outputs should not reopen M20 or claim runtime changes."""
    _write_complete_fixture(tmp_path)

    result = audit_m12_guarded_live(repo_root=tmp_path)
    surface_path = Path(result["output_files"]["guarded_live_surface_audit_csv"])
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
        root / "app/trading/config.py",
        "class LiveConfig:\n    manual_disable_path = ''\n"
        "    startup_checklist_path = ''\n"
        "    live_status_path = ''\n"
        "class PaperProbeConfig:\n    pass\n",
    )
    _write(
        root / "app/trading/schemas.py",
        "class LiveSafetyState:\n    pass\n"
        "class LiveStartupChecklist:\n    pass\n"
        "class BrokerAccount:\n    pass\n"
        "class BrokerSubmitResult:\n    pass\n",
    )
    _write(root / "app/trading/live.py", _live_fixture())
    _write(
        root / "app/trading/repository.py",
        "def save_live_safety_state():\n    pass\n"
        "def load_live_safety_state():\n    pass\n",
    )
    _write(
        root / "app/trading/runner.py",
        "def run():\n"
        "    validate_live_startup()\n"
        "    assert_live_startup_passed()\n"
        "    _refresh_live_reconciliation()\n"
        "    _refresh_live_health_gate()\n",
    )
    _write(
        root / "app/trading/execution.py",
        "LIVE_SIGNAL_STALE = 'LIVE_SIGNAL_STALE'\n"
        "LIVE_SYMBOL_NOT_WHITELISTED = 'LIVE_SYMBOL_NOT_WHITELISTED'\n"
        "LIVE_MAX_ORDER_NOTIONAL_EXCEEDED = 'LIVE_MAX_ORDER_NOTIONAL_EXCEEDED'\n"
        "LIVE_PAPER_PROBE_MAX_ORDERS_PER_RUN_REACHED = 'X'\n"
        "resolve_live_submit_gate\n"
        "record_live_failure\n"
        "def _live_precheck_failure():\n    pass\n",
    )
    _write(root / "README.md", "M11_EXECUTION_INTERFACE_CONSOLIDATED\n")
    _write(root / "docs/training.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")
    _write(root / "PLANS.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")


def _live_fixture() -> str:
    return (
        "LIVE_CONFIRMATION_PHRASE = 'confirm'\n"
        "LIVE_DISABLED = 'LIVE_DISABLED'\n"
        "LIVE_ACCOUNT_ID_MISMATCH = 'LIVE_ACCOUNT_ID_MISMATCH'\n"
        "LIVE_ENVIRONMENT_MISMATCH = 'LIVE_ENVIRONMENT_MISMATCH'\n"
        "LIVE_STARTUP_CHECKS_NOT_PASSED = 'LIVE_STARTUP_CHECKS_NOT_PASSED'\n"
        "LIVE_MANUAL_DISABLE_ACTIVE = 'LIVE_MANUAL_DISABLE_ACTIVE'\n"
        "LIVE_FAILURE_HARD_STOP_ACTIVE = 'LIVE_FAILURE_HARD_STOP_ACTIVE'\n"
        "LIVE_RECONCILIATION_BLOCKED = 'LIVE_RECONCILIATION_BLOCKED'\n"
        "LIVE_RECONCILIATION_BROKER_UNAVAILABLE = 'LIVE_RECONCILIATION_BROKER_UNAVAILABLE'\n"
        "LIVE_RECONCILIATION_ORPHAN_ORDER = 'LIVE_RECONCILIATION_ORPHAN_ORDER'\n"
        "LIVE_RECONCILIATION_ORPHAN_POSITION = 'LIVE_RECONCILIATION_ORPHAN_POSITION'\n"
        "LIVE_RECONCILIATION_ORDER_STATE_MISMATCH = 'LIVE_RECONCILIATION_ORDER_STATE_MISMATCH'\n"
        "LIVE_RECONCILIATION_POSITION_QTY_MISMATCH = 'LIVE_RECONCILIATION_POSITION_QTY_MISMATCH'\n"
        "LIVE_SYSTEM_HEALTH_UNAVAILABLE = 'LIVE_SYSTEM_HEALTH_UNAVAILABLE'\n"
        "LIVE_SYSTEM_HEALTH_STALE = 'LIVE_SYSTEM_HEALTH_STALE'\n"
        "LIVE_SIGNAL_STALE = 'LIVE_SIGNAL_STALE'\n"
        "BROKER_RECONCILIATION_ONLY\n"
        "cross_venue_context\n"
        "can_submit_live_now\n"
        "primary_block_reason_code\n"
        "STREAMALPHA_ENABLE_LIVE\n"
        "execution_mode_live\n"
        "live_config_enabled\n"
        "runtime_live_armed\n"
        "runtime_confirmation\n"
        "alpaca_api_key_present\n"
        "alpaca_api_secret_present\n"
        "alpaca_base_url_present\n"
        "broker_authentication\n"
        "account_environment_match\n"
        "account_id_match\n"
        "symbol_whitelist_non_empty\n"
        "max_order_notional_positive\n"
        "failure_hard_stop_threshold_positive\n"
        "manual_disable_inactive\n"
        "def validate_live_startup():\n    pass\n"
        "def assert_live_startup_passed():\n    pass\n"
        "def resolve_live_submit_gate():\n    pass\n"
        "def derive_live_submit_state():\n    pass\n"
        "def reconcile_live_state():\n    pass\n"
        "def assert_live_reconciliation_clear():\n    pass\n"
        "def apply_live_health_gate():\n    pass\n"
        "def assert_live_health_gate_clear():\n    pass\n"
        "def write_startup_checklist_artifact():\n    pass\n"
        "def write_live_status_artifact():\n    pass\n"
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
