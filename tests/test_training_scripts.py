"""Focused Windows dry-run tests for local M7 operator scripts."""

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
    readiness_payload: dict[str, object],
) -> Path:
    runner_path = tmp_path / "fake_python_runner.py"
    runner_path.write_text(
        "\n".join(
            (
                "import json",
                "import os",
                "import sys",
                "from pathlib import Path",
                f"READINESS_PAYLOAD = {json.dumps(readiness_payload)!r}",
                "args = sys.argv[1:]",
                "if args[:2] == ['-m', 'app.training.readiness']:",
                "    print(READINESS_PAYLOAD)",
                "    raise SystemExit(0)",
                "if args[:2] == ['-m', 'app.training']:",
                "    artifact_root = os.environ.get('STREAMALPHA_TEST_ARTIFACT_ROOT')",
                "    artifact_stamp = os.environ.get(",
                "        'STREAMALPHA_TEST_ARTIFACT_STAMP', '20260402T000000Z'",
                "    )",
                "    training_exit_code = int(",
                "        os.environ.get('STREAMALPHA_TEST_TRAINING_EXIT_CODE', '0')",
                "    )",
                "    if artifact_root:",
                "        artifact_dir = Path.cwd() / artifact_root / artifact_stamp",
                "        artifact_dir.mkdir(parents=True, exist_ok=True)",
                "        required_files = (",
                "            'model.joblib',",
                "            'fold_metrics.csv',",
                "            'oof_predictions.csv',",
                "            'feature_columns.json',",
                "        )",
                "        for file_name in required_files:",
                "            (artifact_dir / file_name).write_text('ok', encoding='utf-8')",
                "        summary_payload = {",
                "            'winner': {'model_name': 'autogluon_tabular'},",
                "            'acceptance': {",
                "                'winner_after_cost_positive': False,",
                "                'meets_acceptance_target': False,",
                "            },",
                "        }",
                "        (artifact_dir / 'summary.json').write_text(",
                "            json.dumps(summary_payload),",
                "            encoding='utf-8',",
                "        )",
                "    print('fake training stdout line')",
                "    print('fake training stderr line', file=sys.stderr)",
                "    raise SystemExit(training_exit_code)",
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
    return launcher_path


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
def test_prepare_script_reports_fastai_as_optional_breadth(
    tmp_path: Path,
) -> None:
    readiness_payload = {
        "config_path": str(_REPO_ROOT / "configs" / "training.m7.json"),
        "config_ok": True,
        "config_error": None,
        "artifact_root": "artifacts/training/m7",
        "source_table": "feature_ohlc",
        "autogluon_installed": True,
        "autogluon_version": "1.5.0",
        "fastai_installed": False,
        "fastai_version": None,
        "fastai_usable": False,
        "fastai_detail": "missing optional breadth only, not a blocker",
        "postgres_reachable": True,
        "postgres_error": None,
        "feature_table_exists": True,
        "row_count": 1146,
        "eligible_rows": 1000,
        "unique_timestamps": 376,
        "required_unique_timestamps": 9,
        "ready_for_training": True,
        "readiness_detail": (
            "feature_ohlc satisfies the configured walk-forward timestamp requirement"
        ),
    }
    _write_fake_python_launcher(tmp_path, readiness_payload=readiness_payload)

    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "prepare_m7_training.ps1"),
            "-DryRun",
        ],
        cwd=_REPO_ROOT,
        env=_script_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "fastai optional breadth: missing (optional breadth only, not a blocker)" in (
        result.stdout
    )
    assert "recommended next command: .\\scripts\\start_m7_training.ps1" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script dry-run tests are Windows-specific.",
)
def test_start_script_dry_run_prints_temp_root_and_training_command(
    tmp_path: Path,
) -> None:
    readiness_payload = {
        "config_path": str(_REPO_ROOT / "configs" / "training.m7.json"),
        "config_ok": True,
        "config_error": None,
        "artifact_root": "artifacts/training/m7",
        "source_table": "feature_ohlc",
        "autogluon_installed": True,
        "autogluon_version": "1.5.0",
        "fastai_installed": False,
        "fastai_version": None,
        "fastai_usable": False,
        "fastai_detail": "missing optional breadth only, not a blocker",
        "postgres_reachable": True,
        "postgres_error": None,
        "feature_table_exists": True,
        "row_count": 1146,
        "eligible_rows": 1000,
        "unique_timestamps": 376,
        "required_unique_timestamps": 9,
        "ready_for_training": True,
        "readiness_detail": (
            "feature_ohlc satisfies the configured walk-forward timestamp requirement"
        ),
    }
    _write_fake_python_launcher(tmp_path, readiness_payload=readiness_payload)

    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "start_m7_training.ps1"),
            "-DryRun",
        ],
        cwd=_REPO_ROOT,
        env=_script_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Local training temp root:" in result.stdout
    assert "CPU mode: sequential_local bagging plus AutoGluon model-level multithreading" in (
        result.stdout
    )
    assert "AutoGluon time budget: 900 seconds" in result.stdout
    assert "artifacts\\tmp\\autogluon" in result.stdout
    assert "Dry run: would run python -m app.training --config .\\configs\\training.m7.json" in (
        result.stdout
    )


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script dry-run tests are Windows-specific.",
)
def test_prepare_script_reports_unusable_fastai_breadth_honestly(
    tmp_path: Path,
) -> None:
    readiness_payload = {
        "config_path": str(_REPO_ROOT / "configs" / "training.m7.json"),
        "config_ok": True,
        "config_error": None,
        "artifact_root": "artifacts/training/m7",
        "source_table": "feature_ohlc",
        "autogluon_installed": True,
        "autogluon_version": "1.5.0",
        "fastai_installed": True,
        "fastai_version": "2.8.7",
        "fastai_usable": False,
        "fastai_detail": "installed but unusable for AutoGluon because IPython is missing",
        "postgres_reachable": True,
        "postgres_error": None,
        "feature_table_exists": True,
        "row_count": 1146,
        "eligible_rows": 1000,
        "unique_timestamps": 376,
        "required_unique_timestamps": 9,
        "ready_for_training": True,
        "readiness_detail": (
            "feature_ohlc satisfies the configured walk-forward timestamp requirement"
        ),
    }
    _write_fake_python_launcher(tmp_path, readiness_payload=readiness_payload)

    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "prepare_m7_training.ps1"),
            "-DryRun",
        ],
        cwd=_REPO_ROOT,
        env=_script_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "fastai optional breadth: 2.8.7 (installed but unusable for AutoGluon because IPython is missing)"
        in result.stdout
    )


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script tests are Windows-specific.",
)
def test_start_script_treats_complete_artifact_with_nonzero_exit_as_completed(
    tmp_path: Path,
) -> None:
    artifact_root = Path("artifacts") / "test-training-scripts" / tmp_path.name
    artifact_root_path = _REPO_ROOT / artifact_root
    if artifact_root_path.exists():
        shutil.rmtree(artifact_root_path)

    readiness_payload = {
        "config_path": str(_REPO_ROOT / "configs" / "training.m7.json"),
        "config_ok": True,
        "config_error": None,
        "artifact_root": str(artifact_root).replace("/", "\\"),
        "source_table": "feature_ohlc",
        "autogluon_installed": True,
        "autogluon_version": "1.5.0",
        "fastai_installed": True,
        "fastai_version": "2.8.7",
        "fastai_usable": True,
        "fastai_detail": "optional breadth available",
        "postgres_reachable": True,
        "postgres_error": None,
        "feature_table_exists": True,
        "row_count": 1146,
        "eligible_rows": 1000,
        "unique_timestamps": 376,
        "required_unique_timestamps": 9,
        "ready_for_training": True,
        "readiness_detail": (
            "feature_ohlc satisfies the configured walk-forward timestamp requirement"
        ),
    }
    _write_fake_python_launcher(tmp_path, readiness_payload=readiness_payload)
    env = _script_env(tmp_path)
    env["STREAMALPHA_TEST_ARTIFACT_ROOT"] = str(artifact_root).replace("/", "\\")
    env["STREAMALPHA_TEST_ARTIFACT_STAMP"] = "20260402T010203Z"
    env["STREAMALPHA_TEST_TRAINING_EXIT_CODE"] = "17"

    try:
        result = subprocess.run(
            [
                _POWERSHELL,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(_REPO_ROOT / "scripts" / "start_m7_training.ps1"),
                "-StatusSeconds",
                "5",
            ],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        if artifact_root_path.exists():
            shutil.rmtree(artifact_root_path)

    assert result.returncode == 0, result.stderr
    assert "M7 training completed" in result.stdout
    assert "winner model name: autogluon_tabular" in result.stdout
    assert "training stdout log:" in result.stdout
    assert "training stderr log:" in result.stdout
    assert "treating the run as completed" in (result.stdout + result.stderr)
