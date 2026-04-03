"""Focused Windows dry-run tests for VPS deployment scripts."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_POWERSHELL = shutil.which("powershell")


def _write_vps_env(tmp_path: Path) -> Path:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            (
                "ipaddress=203.0.113.40",
                "username=streamalpha",
                "password=super-secret",
                "STREAMALPHA_VPS_APP_DIR=~/stream_alpha_paper",
                "",
            )
        ),
        encoding="utf-8",
    )
    return env_path


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These deployment-script dry-run tests are Windows-specific.",
)
def test_deploy_paper_vps_dry_run_prints_safe_summary(tmp_path: Path) -> None:
    env_path = _write_vps_env(tmp_path)

    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "deploy_paper_vps.ps1"),
            "-EnvFile",
            str(env_path),
            "-DryRun",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Paper VPS deploy dry run" in result.stdout
    assert "remote host: 203.0.113.40" in result.stdout
    assert "remote app dir: ~/stream_alpha_paper" in result.stdout
    assert "started services:" in result.stdout
    assert "inspect challengers: .\\scripts\\show_live_policy_challengers_vps.ps1" in result.stdout
    assert "super-secret" not in result.stdout
    assert "python -m app.deployment.paper_vps deploy" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These deployment-script dry-run tests are Windows-specific.",
)
def test_status_paper_vps_dry_run_prints_safe_summary(tmp_path: Path) -> None:
    env_path = _write_vps_env(tmp_path)

    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "status_paper_vps.ps1"),
            "-EnvFile",
            str(env_path),
            "-DryRun",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Paper VPS status dry run" in result.stdout
    assert "remote host: 203.0.113.40" in result.stdout
    assert "python -m app.deployment.paper_vps status" in result.stdout
    assert "super-secret" not in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These deployment-script dry-run tests are Windows-specific.",
)
def test_stop_paper_vps_dry_run_prints_safe_summary(tmp_path: Path) -> None:
    env_path = _write_vps_env(tmp_path)

    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "stop_paper_vps.ps1"),
            "-EnvFile",
            str(env_path),
            "-DryRun",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Paper VPS stop dry run" in result.stdout
    assert "remote host: 203.0.113.40" in result.stdout
    assert "python -m app.deployment.paper_vps stop" in result.stdout
    assert "super-secret" not in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These deployment-script dry-run tests are Windows-specific.",
)
def test_show_live_policy_challengers_vps_dry_run_prints_safe_summary(tmp_path: Path) -> None:
    env_path = _write_vps_env(tmp_path)

    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "show_live_policy_challengers_vps.ps1"),
            "-EnvFile",
            str(env_path),
            "-DryRun",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Paper VPS challenger scoreboard dry run" in result.stdout
    assert "remote host: 203.0.113.40" in result.stdout
    assert "python -m app.deployment.paper_vps show-challengers" in result.stdout
    assert "super-secret" not in result.stdout
