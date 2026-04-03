"""Focused Windows dry-run tests for the Kraken OHLCVT import script."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_POWERSHELL = shutil.which("powershell")


def _write_fake_python_launcher(
    tmp_path: Path,
    *,
    import_payload: dict[str, object],
) -> None:
    runner_path = tmp_path / "fake_python_runner.py"
    runner_path.write_text(
        "\n".join(
            (
                "import json",
                "import sys",
                f"IMPORT_PAYLOAD = {json.dumps(import_payload)!r}",
                "args = sys.argv[1:]",
                "if args[:2] == ['-m', 'app.ingestion.import_kraken_ohlcvt']:",
                "    print(IMPORT_PAYLOAD)",
                "    raise SystemExit(0)",
                "raise SystemExit(f'unexpected args: {args}')",
            )
        ),
        encoding="utf-8",
    )
    launcher_path = tmp_path / "python.cmd"
    launcher_path.write_text(
        "@echo off\r\n"
        "\"%STREAMALPHA_TEST_REAL_PYTHON%\" \"%~dp0fake_python_runner.py\" %*\r\n",
        encoding="utf-8",
    )


def _script_env(tmp_path: Path) -> dict[str, str]:
    env = dict(os.environ)
    path_value = env.get("Path") or env.get("PATH") or ""
    env["STREAMALPHA_TEST_REAL_PYTHON"] = sys.executable
    env["Path"] = f"{tmp_path};{path_value}" if path_value else str(tmp_path)
    env["PATH"] = env["Path"]
    return env


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script dry-run tests are Windows-specific.",
)
def test_import_script_dry_run_prints_authoritative_command() -> None:
    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "import_kraken_ohlcvt.ps1"),
            "-DryRun",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Kraken OHLCVT import dry run" in result.stdout
    assert "dataset root:" in result.stdout
    assert "command: python -m app.ingestion.import_kraken_ohlcvt" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script tests are Windows-specific.",
)
def test_import_script_prints_summary_from_json_payload(tmp_path: Path) -> None:
    _write_fake_python_launcher(
        tmp_path,
        import_payload={
            "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
            "raw_import": [
                {
                    "symbol": "BTC/USD",
                    "parsed_rows": 10,
                    "selected_rows": 10,
                    "created_rows": 8,
                    "updated_rows": 1,
                    "unchanged_rows": 1,
                }
            ],
            "feature_replay": {
                "generated_rows": 9,
                "created_rows": 7,
                "updated_rows": 1,
                "unchanged_rows": 1,
            },
            "readiness": {
                "raw_rows_total": 10,
                "feature_rows_total": 9,
                "labeled_rows_total": 8,
                "ready_for_training": True,
                "readiness_detail": "feature_ohlc satisfies the configured walk-forward timestamp requirement.",
            },
            "readiness_report_artifact_dir": "D:/tmp/readiness",
        },
    )

    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "import_kraken_ohlcvt.ps1"),
            "-DatasetRoot",
            str(_REPO_ROOT / "Datasets" / "master_q4"),
        ],
        cwd=_REPO_ROOT,
        env=_script_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Kraken OHLCVT import complete" in result.stdout
    assert "imported symbols: BTC/USD, ETH/USD, SOL/USD" in result.stdout
    assert "feature rows total: 9" in result.stdout
    assert "labeled rows total: 8" in result.stdout
    assert "readiness artifact path: D:/tmp/readiness" in result.stdout
