"""Tests for the M10 risk-interface audit artifact writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.trading.m10_risk_interface_audit import audit_m10_risk_interface


def test_audit_reports_consolidated_when_required_surfaces_exist(tmp_path: Path) -> None:
    """A complete fixture should classify M10 as consolidated."""
    _write_complete_fixture(tmp_path)

    result = audit_m10_risk_interface(repo_root=tmp_path)

    assert result["m10_state"] == "M10_RISK_INTERFACE_CONSOLIDATED"
    assert result["gap_count"] == 0
    assert result["recommendation"] == "PROCEED_TO_M11_EXECUTION_INTERFACE_AUDIT"
    assert result["next_required_action"] == "AUDIT_M11_EXECUTION_INTERFACE_WITH_RISK_AUTHORITY"
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]


def test_audit_writes_expected_artifacts(tmp_path: Path) -> None:
    """The audit should persist its deterministic artifact contract."""
    _write_complete_fixture(tmp_path)

    result = audit_m10_risk_interface(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    expected_files = {
        "manifest.json",
        "m10_risk_interface_audit.json",
        "m10_risk_interface_audit.md",
        "risk_surface_audit.csv",
        "regime_context_readiness.csv",
        "authority_boundary_audit.csv",
        "gap_analysis.csv",
        "next_actions.csv",
        "recommendation.json",
    }
    assert expected_files == {path.name for path in output_dir.iterdir()}
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    assert recommendation["runtime_ready"] is False
    assert recommendation["promotable"] is False
    assert recommendation["profitability_claim"] is False


def test_missing_surface_produces_partial_state_and_gap(tmp_path: Path) -> None:
    """A missing persistence surface should become a blocking M10 gap."""
    _write_complete_fixture(tmp_path)
    (tmp_path / "app/trading/repository.py").write_text(
        "class TradingRepository:\n    pass\n",
        encoding="utf-8",
    )

    result = audit_m10_risk_interface(repo_root=tmp_path)

    assert result["m10_state"] == "M10_RISK_INTERFACE_PARTIAL"
    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "risk_decision_persistence" in gap_names
    assert result["recommendation"] == "IMPLEMENT_REUSABLE_M10_RISK_INTERFACE_GAP_FILLS"


def test_missing_trade_allowed_blocker_is_reported(tmp_path: Path) -> None:
    """The audit should catch absent fail-closed trade_allowed handling."""
    _write_complete_fixture(tmp_path)
    (tmp_path / "app/trading/risk_engine.py").write_text(
        "def evaluate_risk():\n    pass\n"
        "def build_risk_decision_log_entry():\n    regime_run_id=signal.regime_run_id\n",
        encoding="utf-8",
    )

    result = audit_m10_risk_interface(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "trade_allowed_blocker" in gap_names
    assert "trade_allowed_fail_closed" in gap_names


def test_outputs_preserve_m20_pause_and_no_authority_change(tmp_path: Path) -> None:
    """Risk audit outputs should not reopen M20 or claim runtime changes."""
    _write_complete_fixture(tmp_path)

    result = audit_m10_risk_interface(repo_root=tmp_path)
    surface_path = Path(result["output_files"]["risk_surface_audit_csv"])
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
        "class RiskConfig:\n    regime_position_fraction_caps = {}\n",
    )
    _write(
        root / "app/trading/risk_engine.py",
        "TRADE_NOT_ALLOWED = 'TRADE_NOT_ALLOWED'\n"
        "def evaluate_risk():\n"
        "    if signal.trade_allowed is False:\n"
        "        return TRADE_NOT_ALLOWED\n"
        "    regime_position_fraction_caps = {}\n"
        "def build_risk_decision_log_entry():\n"
        "    regime_run_id=signal.regime_run_id\n",
    )
    _write(
        root / "app/trading/schemas.py",
        "class ServiceRiskState:\n    pass\n"
        "class RiskDecision:\n    trade_allowed: bool | None = None\n"
        "class RiskDecisionLogEntry:\n    pass\n"
        "class SignalDecision:\n    regime_label: str | None = None\n",
    )
    _write(
        root / "app/trading/repository.py",
        "class TradingRepository:\n    def insert_risk_decision(self):\n        pass\n",
    )
    _write(
        root / "app/trading/decision_trace.py",
        "def enrich_decision_trace_with_risk():\n    pass\n"
        "def build_risk_section():\n    pass\n",
    )
    _write(
        root / "app/trading/runner.py",
        "def run():\n"
        "    risk_decision = evaluate_risk()\n"
        "    build_order_request()\n",
    )
    _write(root / "app/trading/execution.py", "def build_order_request():\n    pass\n")
    _write(
        root / "app/regime/context.py",
        "REGIME_CONTEXT_SCHEMA_VERSION = 'm9_regime_context_v1'\n"
        "def missing_regime_context():\n    pass\n",
    )
    _write(
        root / "configs/paper_trading.yaml",
        "risk:\n  regime_position_fraction_caps:\n    TREND_UP: 0.25\n",
    )
    _write(root / "README.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")
    _write(root / "docs/training.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")
    _write(root / "PLANS.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
