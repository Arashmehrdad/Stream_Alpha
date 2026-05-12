"""Tests for the M16 deployment/environment audit artifact writer."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.runtime.m16_deployment_environment_audit import audit_m16_deployment_environment


def test_audit_reports_consolidated_when_required_surfaces_exist(tmp_path: Path) -> None:
    """A complete fixture should classify M16 as consolidated."""
    _write_complete_fixture(tmp_path)

    result = audit_m16_deployment_environment(repo_root=tmp_path)

    assert result["m16_state"] == "M16_DEPLOYMENT_ENVIRONMENT_CONTROLS_CONSOLIDATED"
    assert result["gap_count"] == 0
    assert result["recommendation"] == "PROCEED_TO_M17_OPERATIONAL_ALERTING_AUDIT"
    assert (
        result["next_required_action"]
        == "AUDIT_M17_OPERATIONAL_ALERTING_AND_INCIDENT_CONTROLS"
    )
    assert "M20_PAUSED" in result["honesty_flags"]
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]


def test_audit_writes_expected_artifacts(tmp_path: Path) -> None:
    """The audit should persist its deterministic artifact contract."""
    _write_complete_fixture(tmp_path)

    result = audit_m16_deployment_environment(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    expected_files = {
        "manifest.json",
        "m16_deployment_environment_audit.json",
        "m16_deployment_environment_audit.md",
        "deployment_surface_audit.csv",
        "runtime_profile_audit.csv",
        "startup_validation_audit.csv",
        "remote_deployment_audit.csv",
        "gap_analysis.csv",
        "next_actions.csv",
        "recommendation.json",
    }
    assert expected_files == {path.name for path in output_dir.iterdir()}
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    assert recommendation["runtime_ready"] is False
    assert recommendation["promotable"] is False
    assert recommendation["profitability_claim"] is False


def test_missing_startup_validator_produces_partial_state(tmp_path: Path) -> None:
    """Missing startup validation should be reported as an M16 gap."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/runtime/validate.py",
        _runtime_validate_fixture().replace("def build_startup_validation_report", ""),
    )

    result = audit_m16_deployment_environment(repo_root=tmp_path)

    assert result["m16_state"] == "M16_DEPLOYMENT_ENVIRONMENT_CONTROLS_PARTIAL"
    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "startup_validation_builder" in gap_names
    assert result["recommendation"] == "IMPLEMENT_REUSABLE_M16_DEPLOYMENT_ENVIRONMENT_GAP_FILLS"


def test_missing_compose_config_check_is_reported(tmp_path: Path) -> None:
    """Compose startup validation must remain visible as the config-check service."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "docker-compose.yml",
        _compose_fixture().replace("config-check:", "config-skip:"),
    )

    result = audit_m16_deployment_environment(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "compose_config_check_service" in gap_names


def test_missing_paper_vps_runtime_overrides_are_reported(tmp_path: Path) -> None:
    """Remote paper deployment must force paper-mode runtime overrides."""
    _write_complete_fixture(tmp_path)
    _write(
        tmp_path / "app/deployment/paper_vps.py",
        _paper_vps_fixture().replace("REMOTE_RUNTIME_ENV_OVERRIDES", ""),
    )

    result = audit_m16_deployment_environment(repo_root=tmp_path)

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert "paper_vps_runtime_overrides" in gap_names


def test_outputs_preserve_m20_pause_and_no_authority_change(tmp_path: Path) -> None:
    """M16 audit outputs should not reopen M20 or claim runtime changes."""
    _write_complete_fixture(tmp_path)

    result = audit_m16_deployment_environment(repo_root=tmp_path)
    surface_path = Path(result["output_files"]["deployment_surface_audit_csv"])
    with surface_path.open(encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))

    assert rows
    assert {row["runtime_authority_changed"] for row in rows} == {"False"}
    assert {row["m20_reopened"] for row in rows} == {"False"}
    assert result["runtime_status"] == "NO_RUNTIME_EFFECT"
    assert result["promotion_status"] == "NOT_PROMOTABLE"
    assert result["profitability_status"] == "NO_PROFIT_CLAIM"


def _write_complete_fixture(root: Path) -> None:
    _write(root / "app/runtime/config.py", _runtime_config_fixture())
    _write(root / "app/runtime/validate.py", _runtime_validate_fixture())
    _write(root / "app/inference/service.py", "build_runtime_metadata\n")
    _write(root / "app/trading/runner.py", "build_runtime_metadata\n")
    _write(root / "docker-compose.yml", _compose_fixture())
    _write(root / "scripts/start-stack.ps1", "param(\ndocker compose\n")
    _write(root / "scripts/stop-stack.ps1", '@("compose", "down")\n')
    _write(
        root / ".env.example",
        "STREAMALPHA_RUNTIME_PROFILE=\nSTREAMALPHA_STARTUP_REPORT_PATH=\n",
    )
    _write(root / "configs/paper_trading.paper.yaml", "mode: paper\n")
    _write(root / "configs/paper_trading.shadow.yaml", "mode: shadow\n")
    _write(root / "configs/paper_trading.live.yaml", "mode: live\n")
    _write(root / "app/deployment/paper_vps.py", _paper_vps_fixture())
    _write(root / "scripts/deploy_paper_vps.ps1", "app.deployment.paper_vps\n")
    _write(root / "scripts/status_paper_vps.ps1", "app.deployment.paper_vps\n")
    _write(root / "scripts/stop_paper_vps.ps1", "app.deployment.paper_vps\n")
    _write(
        root / "tests/test_runtime_validate.py",
        "test_startup_validation_passes_for_paper_with_artifacts\n",
    )
    _write(
        root / "tests/test_deployment_paper_vps.py",
        "test_build_deploy_plan_discovers_bounded_upload_set\n",
    )
    _write(root / "README.md", "M15_OPERATOR_CONSOLE_VISIBILITY_CONTROLS_CONSOLIDATED\n")
    _write(root / "docs/training.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")
    _write(root / "PLANS.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY\n")


def _runtime_config_fixture() -> str:
    return (
        'RUNTIME_PROFILES = ("dev", "paper", "shadow", "live")\n'
        '"dev"\n"paper"\n"shadow"\n"live"\n'
        "paper_trading.paper.yaml\n"
        "paper_trading.shadow.yaml\n"
        "paper_trading.live.yaml\n"
        "def resolve_runtime_profile():\n    pass\n"
        "def profile_trading_config_path():\n    pass\n"
        "def default_startup_report_path():\n    pass\n"
        "def build_runtime_metadata():\n    pass\n"
    )


def _runtime_validate_fixture() -> str:
    return (
        "STARTUP_REPORT_SCHEMA_VERSION = 'm16_startup_report_v1'\n"
        "def build_startup_validation_report():\n    pass\n"
        "def write_startup_validation_report():\n    pass\n"
        '"runtime_profile"\n'
        '"settings_env_parse"\n'
        '"trading_config_path"\n'
        '"profile_matches_execution_mode"\n'
        '"model_artifact"\n'
        '"regime_artifact"\n'
        "APCA_API_KEY_ID\n"
        "STREAMALPHA_ENABLE_LIVE\n"
        "STREAMALPHA_LIVE_CONFIRM\n"
    )


def _compose_fixture() -> str:
    return (
        'profiles: ["dev", "paper", "shadow", "live"]\n'
        "config-check:\n"
        "app.runtime.validate\n"
        "healthcheck:\n"
    )


def _paper_vps_fixture() -> str:
    return (
        "DEFAULT_UPLOAD_ROOTS = ()\n"
        "REMOTE_RUNTIME_ENV_OVERRIDES = {}\n"
        "VPS_HOST_ALIASES = ()\n"
        "def build_remote_env_text():\n    pass\n"
        "def _wrap_ssh_connect_error():\n    pass\n"
        "def build_deploy_plan():\n    pass\n"
    )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
