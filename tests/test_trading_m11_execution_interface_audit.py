"""Tests for the M11 execution-interface audit artifact writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.trading.m11_execution_interface_audit import audit_m11_execution_interface


def test_audit_reports_consolidated_when_required_surfaces_exist(tmp_path: Path) -> None:
    """A complete fixture should classify M11 as consolidated."""
    _write_complete_fixture(tmp_path)

    result = audit_m11_execution_interface(repo_root=tmp_path)

    assert result["m11_state"] == "M11_EXECUTION_INTERFACE_CONSOLIDATED"
    assert result["gap_count"] == 0
    assert result["recommendation"] == "PROCEED_TO_M12_GUARDED_LIVE_AUDIT"
    assert result["next_required_action"] == "AUDIT_M12_GUARDED_LIVE_CONTROLS"
    assert "M20_PAUSED" in result["honesty_flags"]
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]


def test_audit_writes_expected_artifacts(tmp_path: Path) -> None:
    """The audit should persist its deterministic artifact contract."""
    _write_complete_fixture(tmp_path)

    result = audit_m11_execution_interface(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    expected_files = {
        "manifest.json",
        "m11_execution_interface_audit.json",
        "m11_execution_interface_audit.md",
        "execution_surface_audit.csv",
        "risk_authority_boundary_audit.csv",
        "order_lifecycle_audit.csv",
        "mode_adapter_audit.csv",
        "gap_analysis.csv",
        "next_actions.csv",
        "recommendation.json",
    }
    assert expected_files == {path.name for path in output_dir.iterdir()}
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    assert recommendation["runtime_ready"] is False
    assert recommendation["promotable"] is False
    assert recommendation["profitability_claim"] is False


def test_missing_order_request_persistence_produces_gap(tmp_path: Path) -> None:
    """Missing idempotent order persistence should be a blocking M11 gap."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/trading/repository.py",
        "def insert_order_event_if_absent():\n    pass\n",
    )

    result = audit_m11_execution_interface(repo_root=tmp_path)

    assert result["m11_state"] == "M11_EXECUTION_INTERFACE_PARTIAL"
    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "order_request_persistence" in gap_names
    assert result["recommendation"] == "IMPLEMENT_REUSABLE_M11_EXECUTION_INTERFACE_GAP_FILLS"


def test_missing_risk_authority_handoff_is_reported(tmp_path: Path) -> None:
    """The audit should catch absent M10-to-M11 handoff evidence."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/trading/runner.py",
        "def run():\n    build_created_event()\n",
    )

    result = audit_m11_execution_interface(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "runner_order_after_risk" in gap_names
    assert "runner_evaluates_risk_before_order_request" in gap_names


def test_missing_live_submit_gate_is_reported(tmp_path: Path) -> None:
    """The audit should catch absent guarded-live submit gates."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/trading/execution.py",
        _execution_fixture().replace("resolve_live_submit_gate\n", ""),
    )

    result = audit_m11_execution_interface(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "live_submit_gate" in gap_names


def test_outputs_preserve_m20_pause_and_no_authority_change(tmp_path: Path) -> None:
    """Execution audit outputs should not reopen M20 or claim runtime changes."""
    _write_complete_fixture(tmp_path)

    result = audit_m11_execution_interface(repo_root=tmp_path)
    surface_path = Path(result["output_files"]["execution_surface_audit_csv"])
    with surface_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert rows
    assert {row["runtime_authority_changed"] for row in rows} == {"False"}
    assert {row["m20_reopened"] for row in rows} == {"False"}
    assert result["runtime_status"] == "NO_RUNTIME_EFFECT"
    assert result["promotion_status"] == "NOT_PROMOTABLE"
    assert result["profitability_status"] == "NO_PROFIT_CLAIM"


def _write_complete_fixture(root: Path) -> None:
    _write(root / "app/trading/execution.py", _execution_fixture())
    _write(
        root / "app/trading/schemas.py",
        'OrderLifecycleState = ("CREATED", "SUBMITTED", "ACCEPTED", "FILLED", "REJECTED")\n'
        "class OrderRequest:\n    pass\n"
        "class OrderLifecycleEvent:\n    pass\n"
        "class RiskDecision:\n    pass\n",
    )
    _write(
        root / "app/trading/repository.py",
        "def ensure_order_request():\n    idempotency_key = 'x'\n"
        "def insert_order_event_if_absent():\n    pass\n",
    )
    _write(
        root / "app/trading/runner.py",
        "def run():\n"
        "    risk_decision = evaluate_risk()\n"
        "    order_request = build_order_request()\n"
        "    build_created_event()\n"
        "    insert_risk_decision()\n"
        "    insert_order_event_if_absent()\n",
    )
    _write(root / "README.md", "M10_RISK_INTERFACE_CONSOLIDATED\n")
    _write(root / "docs/training.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")
    _write(root / "PLANS.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")


def _execution_fixture() -> str:
    return (
        'LIVE_SYMBOL_NOT_WHITELISTED = "LIVE_SYMBOL_NOT_WHITELISTED"\n'
        'LIVE_MAX_ORDER_NOTIONAL_EXCEEDED = "LIVE_MAX_ORDER_NOTIONAL_EXCEEDED"\n'
        'LIVE_SIGNAL_STALE = "LIVE_SIGNAL_STALE"\n'
        'LIVE_BROKER_SUBMIT_FAILED = "LIVE_BROKER_SUBMIT_FAILED"\n'
        "resolve_live_submit_gate\n"
        "class BrokerClient:\n    pass\n"
        "def build_idempotency_key():\n    pass\n"
        "def build_order_request():\n"
        '    if decision.outcome in {"APPROVED", "MODIFIED"}:\n'
        "        return object()\n"
        "def build_pending_order_request():\n    pass\n"
        "def build_created_event():\n    pass\n"
        "class ExecutionAdapter:\n    pass\n"
        "class PaperExecutionAdapter:\n    pass\n"
        "class ShadowExecutionAdapter:\n    pass\n"
        "class LiveExecutionAdapter:\n    pass\n"
        "def build_execution_adapter(mode):\n"
        '    if mode == "paper":\n        return PaperExecutionAdapter()\n'
        '    if mode == "shadow":\n        return ShadowExecutionAdapter()\n'
        '    if mode == "live":\n        return LiveExecutionAdapter()\n'
        "def _terminal_lifecycle_events():\n    pass\n"
        "def _build_live_broker_event():\n    pass\n"
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
