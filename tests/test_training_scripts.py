"""Focused Windows dry-run tests for local M7 operator scripts."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_POWERSHELL = shutil.which("powershell")


def _write_research_config_for_script_test(config_path: Path, *, presets: str) -> None:
    config_path.write_text(
        json.dumps(
            {
                "artifact_root": "artifacts/training/m7",
                "source_table": "feature_ohlc",
                "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
                "time_column": "as_of_time",
                "interval_column": "interval_begin",
                "close_column": "close_price",
                "categorical_feature_columns": ["symbol"],
                "numeric_feature_columns": [
                    "open_price",
                    "high_price",
                    "low_price",
                    "close_price",
                    "vwap",
                    "trade_count",
                    "volume",
                    "log_return_1",
                    "log_return_3",
                    "momentum_3",
                    "return_mean_12",
                    "return_std_12",
                    "realized_vol_12",
                    "rsi_14",
                    "macd_line_12_26",
                    "volume_mean_12",
                    "volume_std_12",
                    "volume_zscore_12",
                    "close_zscore_12",
                    "lag_log_return_1",
                    "lag_log_return_2",
                    "lag_log_return_3",
                ],
                "label_horizon_candles": 3,
                "purge_gap_candles": 3,
                "test_folds": 5,
                "first_train_fraction": 0.5,
                "test_fraction": 0.1,
                "round_trip_fee_bps": 20,
                "comparison_policy": {
                    "primary_metric": "mean_long_only_net_value_proxy",
                    "max_directional_accuracy_regression": 0.01,
                    "max_brier_score_worsening": 0.01,
                },
                "models": {
                    "autogluon_tabular": {
                        "presets": presets,
                        "time_limit": 900,
                        "eval_metric": "log_loss",
                        "hyperparameters": None,
                        "fit_weighted_ensemble": True,
                        "num_bag_folds": 5,
                        "num_stack_levels": 1,
                        "num_bag_sets": 1,
                        "fold_fitting_strategy": "sequential_local",
                        "dynamic_stacking": False,
                        "calibrate_decision_threshold": False,
                        "verbosity": 0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


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
                "if args[:2] in (['-m', 'app.training.readiness'], ['-m', 'app.training.preflight_m20']):",
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


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script dry-run tests are Windows-specific.",
)
def test_analyze_script_dry_run_resolves_newest_run_directory() -> None:
    artifact_root = _REPO_ROOT / "artifacts" / "training" / "m7"
    older_run = artifact_root / "20990101T000000Z-test-analyze-old"
    newer_run = artifact_root / "20990101T000001Z-test-analyze-new"
    analysis_dir = artifact_root / "_analysis"

    try:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        os.utime(analysis_dir, None)
        for run_dir in (older_run, newer_run):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text(
                json.dumps({"winner": {"model_name": "autogluon_tabular"}}),
                encoding="utf-8",
            )
            (run_dir / "oof_predictions.csv").write_text("model_name\n", encoding="utf-8")
        os.utime(older_run, (time.time() - 10, time.time() - 10))
        os.utime(newer_run, None)

        result = subprocess.run(
            [
                _POWERSHELL,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(_REPO_ROOT / "scripts" / "analyze_m7_thresholds.ps1"),
                "-DryRun",
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        for run_dir in (older_run, newer_run):
            if run_dir.exists():
                shutil.rmtree(run_dir)

    assert result.returncode == 0, result.stderr
    assert f"Resolved M7 run dir: {newer_run}" in result.stdout
    assert "Dry run: would run python -m app.training.threshold_analysis" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script dry-run tests are Windows-specific.",
)
def test_policy_candidate_script_dry_run_resolves_newest_run_directory() -> None:
    artifact_root = _REPO_ROOT / "artifacts" / "training" / "m7"
    older_run = artifact_root / "20990101T000010Z-test-policy-old"
    newer_run = artifact_root / "20990101T000011Z-test-policy-new"
    analysis_dir = artifact_root / "_analysis"

    try:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        os.utime(analysis_dir, None)
        for run_dir in (older_run, newer_run):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text(
                json.dumps({"winner": {"model_name": "autogluon_tabular"}}),
                encoding="utf-8",
            )
            (run_dir / "oof_predictions.csv").write_text("model_name\n", encoding="utf-8")
        os.utime(older_run, (time.time() - 10, time.time() - 10))
        os.utime(newer_run, None)

        result = subprocess.run(
            [
                _POWERSHELL,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(_REPO_ROOT / "scripts" / "evaluate_m7_policy_candidates.ps1"),
                "-DryRun",
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        for run_dir in (older_run, newer_run):
            if run_dir.exists():
                shutil.rmtree(run_dir)

    assert result.returncode == 0, result.stderr
    assert f"Resolved M7 run dir: {newer_run}" in result.stdout
    assert "Dry run: would run python -m app.training.policy_candidate_analysis" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script dry-run tests are Windows-specific.",
)
def test_multi_run_policy_candidate_script_dry_run_uses_artifact_root_scan(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "m7-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    for run_name in ("20260401T000001Z", "20260401T000002Z"):
        run_dir = artifact_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(
            json.dumps({"winner": {"model_name": "autogluon_tabular"}}),
            encoding="utf-8",
        )
        (run_dir / "oof_predictions.csv").write_text("model_name\n", encoding="utf-8")

    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "evaluate_m7_policy_candidates_multi_run.ps1"),
            "-ArtifactRoot",
            str(artifact_root),
            "-DryRun",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"Resolved M7 artifact root: {artifact_root}" in result.stdout
    assert "Dry run: would run python -m app.training.multi_run_policy_analysis" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script dry-run tests are Windows-specific.",
)
def test_research_experiment_script_dry_run_lists_discovered_configs(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    artifact_root = tmp_path / "artifacts" / "training" / "m7"
    config_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    _write_research_config_for_script_test(
        config_dir / "training.m7.research.high_quality.json",
        presets="high_quality",
    )
    _write_research_config_for_script_test(
        config_dir / "training.m7.research.best_quality.json",
        presets="best_quality",
    )
    _write_research_config_for_script_test(
        config_dir / "training.m7.research.best_quality_v150.json",
        presets="best_quality_v150",
    )

    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "run_m7_research_experiments.ps1"),
            "-ConfigDirectory",
            str(config_dir),
            "-ArtifactRoot",
            str(artifact_root),
            "-DryRun",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"Resolved research config directory: {config_dir}" in result.stdout
    assert f"Resolved M7 artifact root: {artifact_root}" in result.stdout
    assert "best_quality" in result.stdout
    assert "best_quality_v150" in result.stdout
    assert "high_quality" in result.stdout
    assert "Dry run: would run .\\scripts\\start_m7_training.ps1 -ConfigPath" in result.stdout
    assert "Dry run: would then run .\\scripts\\evaluate_m7_policy_candidates.ps1 -RunDir <new_run_for_best_quality>" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script dry-run tests are Windows-specific.",
)
def test_policy_replay_script_dry_run_resolves_newest_run_directory() -> None:
    artifact_root = _REPO_ROOT / "artifacts" / "training" / "m7"
    older_run = artifact_root / "20990101T000030Z-test-replay-old"
    newer_run = artifact_root / "20990101T000031Z-test-replay-new"
    analysis_dir = artifact_root / "_analysis"

    try:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        os.utime(analysis_dir, None)
        for run_dir in (older_run, newer_run):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text(
                json.dumps({"winner": {"model_name": "autogluon_tabular"}}),
                encoding="utf-8",
            )
            (run_dir / "oof_predictions.csv").write_text("model_name\n", encoding="utf-8")
        os.utime(older_run, (time.time() - 10, time.time() - 10))
        os.utime(newer_run, None)

        result = subprocess.run(
            [
                _POWERSHELL,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(_REPO_ROOT / "scripts" / "evaluate_m7_policy_replay.ps1"),
                "-DryRun",
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        for run_dir in (older_run, newer_run):
            if run_dir.exists():
                shutil.rmtree(run_dir)

    assert result.returncode == 0, result.stderr
    assert f"Resolved M7 run dir: {newer_run}" in result.stdout
    assert "Dry run: would run python -m app.training.policy_replay_analysis" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script dry-run tests are Windows-specific.",
)
def test_multi_run_policy_replay_script_dry_run_uses_artifact_root_scan(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "m7-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    for run_name in ("20260401T000001Z", "20260401T000002Z"):
        run_dir = artifact_root / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "summary.json").write_text(
            json.dumps({"winner": {"model_name": "autogluon_tabular"}}),
            encoding="utf-8",
        )
        (run_dir / "oof_predictions.csv").write_text("model_name\n", encoding="utf-8")

    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "evaluate_m7_policy_replay_multi_run.ps1"),
            "-ArtifactRoot",
            str(artifact_root),
            "-DryRun",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"Resolved M7 artifact root: {artifact_root}" in result.stdout
    assert "Dry run: would run python -m app.training.policy_replay_analysis --multi-run" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script dry-run tests are Windows-specific.",
)
def test_data_regime_script_dry_run_resolves_newest_completed_run_and_ignores_analysis_dir() -> None:
    artifact_root = _REPO_ROOT / "artifacts" / "training" / "m7"
    older_run = artifact_root / "20990101T000020Z-test-data-old"
    newer_run = artifact_root / "20990101T000021Z-test-data-new"
    analysis_dir = artifact_root / "_analysis"

    try:
        analysis_dir.mkdir(parents=True, exist_ok=True)
        os.utime(analysis_dir, None)
        for run_dir in (older_run, newer_run):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text(
                json.dumps({"winner": {"model_name": "autogluon_tabular"}}),
                encoding="utf-8",
            )
            (run_dir / "oof_predictions.csv").write_text("model_name\n", encoding="utf-8")
            (run_dir / "fold_metrics.csv").write_text("model_name\n", encoding="utf-8")
            (run_dir / "dataset_manifest.json").write_text("{}", encoding="utf-8")
        os.utime(older_run, (time.time() - 10, time.time() - 10))
        os.utime(newer_run, None)

        result = subprocess.run(
            [
                _POWERSHELL,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(_REPO_ROOT / "scripts" / "analyze_m7_data_regime.ps1"),
                "-DryRun",
            ],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        for run_dir in (older_run, newer_run):
            if run_dir.exists():
                shutil.rmtree(run_dir)

    assert result.returncode == 0, result.stderr
    assert f"Resolved M7 run dir: {newer_run}" in result.stdout
    assert "Dry run: would run python -m app.training.data_regime_diagnostics" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script dry-run tests are Windows-specific.",
)
def test_live_policy_challenger_script_dry_run_prints_authoritative_command() -> None:
    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "show_live_policy_challengers.ps1"),
            "-DryRun",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Live policy challenger observer dry run" in result.stdout
    assert "trading config: .\\configs\\paper_trading.paper.yaml" in result.stdout
    assert "training config: .\\configs\\training.m7.json" in result.stdout
    assert "command: python -m app.training.live_policy_challenger" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script dry-run tests are Windows-specific.",
)
def test_m20_preflight_script_dry_run_prints_authoritative_command() -> None:
    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "preflight_m20_training.ps1"),
            "-DryRun",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "M20 specialist preflight dry run" in result.stdout
    assert "config path:" in result.stdout
    assert "command: python -m app.training.preflight_m20" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script tests are Windows-specific.",
)
def test_m20_preflight_script_dry_run_honors_require_gpu_flag() -> None:
    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "preflight_m20_training.ps1"),
            "-DryRun",
            "-RequireGpu",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "M20 specialist preflight dry run" in result.stdout
    assert "gpu required: yes" in result.stdout
    assert "--require-gpu" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script dry-run tests are Windows-specific.",
)
def test_start_m20_training_script_dry_run_prints_full_dataset_training_truth() -> None:
    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "start_m20_training.ps1"),
            "-DryRun",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "M20 specialist training dry run" in result.stdout
    assert "training source table: feature_ohlc" in result.stdout
    assert "symbols: BTC/USD, ETH/USD, SOL/USD" in result.stdout
    assert "models: neuralforecast_nhits, neuralforecast_patchtst" in result.stdout
    assert (
        "specialist dataset mode: neuralforecast_nhits=local_files_partitioned, "
        "neuralforecast_patchtst=local_files_partitioned"
        in result.stdout
    )
    assert (
        "specialist memory profile: neuralforecast_nhits(batch=1, valid=1, "
        "windows=8, infer_windows=32, score_chunk=25000, "
        "context_batch=512, step=128, precision=16-mixed); "
        "neuralforecast_patchtst(batch=1, valid=1, windows=4, "
        "infer_windows=16, score_chunk=25000, context_batch=512, "
        "step=128, precision=16-mixed)"
        in result.stdout
    )
    assert "allocator hint: expandable_segments:True" in result.stdout
    assert (
        "progress output: terminal bars disabled by config; see progress.log and "
        "progress_status.json inside the new artifact run directory"
        in result.stdout
    )
    assert "command: python -m app.training --config" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script dry-run tests are Windows-specific.",
)
def test_rescore_m20_training_script_dry_run_prints_score_only_command() -> None:
    result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "rescore_m20_training.ps1"),
            "-DryRun",
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "M20 specialist score-only dry run" in result.stdout
    assert "fitted models dir:" in result.stdout
    assert "models: neuralforecast_nhits, neuralforecast_patchtst" in result.stdout
    assert "command: python -m app.training --config" in result.stdout
    assert "--score-only" in result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script tests are Windows-specific.",
)
def test_status_m20_training_script_handles_empty_partial_and_completed_runs(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "m20"
    artifact_root.mkdir()

    empty_result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "status_m20_training.ps1"),
            "-ArtifactRoot",
            str(artifact_root),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert empty_result.returncode == 0, empty_result.stderr
    assert "run status: no_artifacts" in empty_result.stdout

    partial_dir = artifact_root / "20260426T010000Z"
    partial_dir.mkdir()
    (partial_dir / "run_config.json").write_text(
        json.dumps({"execution_mode": "score_only"}),
        encoding="utf-8",
    )
    (partial_dir / "progress_status.json").write_text(
        json.dumps(
            {
                "state": "running",
                "stage": "sequence_scoring",
                "event": "sequence_scoring_start",
                "model_name": "neuralforecast_nhits",
            }
        ),
        encoding="utf-8",
    )
    (partial_dir / "checkpoint.json").write_text("{}", encoding="utf-8")

    partial_result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "status_m20_training.ps1"),
            "-RunDir",
            str(partial_dir),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert partial_result.returncode == 0, partial_result.stderr
    assert "execution mode: score_only" in partial_result.stdout
    assert "progress model: neuralforecast_nhits" in partial_result.stdout
    assert "run status: incomplete" in partial_result.stdout
    assert "checkpoint: present" in partial_result.stdout

    completed_dir = artifact_root / "20260426T020000Z"
    completed_dir.mkdir()
    (completed_dir / "run_config.json").write_text(
        json.dumps({"execution_mode": "full_training"}),
        encoding="utf-8",
    )
    (completed_dir / "summary.json").write_text(
        json.dumps(
            {
                "winner": {"model_name": "neuralforecast_nhits"},
                "acceptance": {
                    "scope": "recent_window",
                    "verdict_basis": "incumbent_comparison",
                    "incumbent_model_version": "m7-test",
                    "meets_acceptance_target": False,
                },
                "specialist_verdicts": {
                    "neuralforecast_nhits": {
                        "candidate_role": "TREND_SPECIALIST",
                        "verdict": "rejected",
                        "verdict_basis": "incumbent_comparison",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    completed_result = subprocess.run(
        [
            _POWERSHELL,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(_REPO_ROOT / "scripts" / "status_m20_training.ps1"),
            "-RunDir",
            str(completed_dir),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed_result.returncode == 0, completed_result.stderr
    assert "run status: completed" in completed_result.stdout
    assert "verdict basis: incumbent_comparison" in completed_result.stdout
    assert "specialist verdict: neuralforecast_nhits" in completed_result.stdout


@pytest.mark.skipif(
    sys.platform != "win32" or _POWERSHELL is None,
    reason="These operator-script tests are Windows-specific.",
)
def test_start_m20_training_script_tolerates_native_stderr_when_training_exits_zero(
    tmp_path: Path,
) -> None:
    readiness_payload = {
        "config_path": str(_REPO_ROOT / "configs" / "training.m20.json"),
        "source_table": "feature_ohlc",
        "symbols": ["BTC/USD", "ETH/USD", "SOL/USD"],
        "configured_models": ["neuralforecast_nhits", "neuralforecast_patchtst"],
        "preferred_execution_device": "gpu",
        "ready_for_manual_training": True,
        "warnings": [
            "No real registry-backed NHITS or PatchTST specialist candidates exist yet; this batch only prepares the first real run."
        ],
        "blockers": [],
        "registry": {
            "specialist_entry_count": 0,
        },
        "runtime": {
            "lightning": {"installed": True},
            "neuralforecast": {"installed": True},
            "torch": {
                "version": "2.9.0+cu130",
                "cuda_available": True,
                "device_names": ["NVIDIA GeForce RTX 4060 Laptop GPU"],
            },
        },
    }
    _write_fake_python_launcher(tmp_path, readiness_payload=readiness_payload)
    env = _script_env(tmp_path)
    artifact_root = _REPO_ROOT / "artifacts" / "training" / "m20"
    artifact_stamp = f"test-{int(time.time())}"
    generated_artifact_dir = artifact_root / artifact_stamp
    env["STREAMALPHA_TEST_ARTIFACT_ROOT"] = "artifacts/training/m20"
    env["STREAMALPHA_TEST_ARTIFACT_STAMP"] = artifact_stamp
    env["STREAMALPHA_TEST_TRAINING_EXIT_CODE"] = "0"

    try:
        result = subprocess.run(
            [
                _POWERSHELL,
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(_REPO_ROOT / "scripts" / "start_m20_training.ps1"),
            ],
            cwd=_REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    finally:
        shutil.rmtree(generated_artifact_dir, ignore_errors=True)

    assert result.returncode == 0, result.stderr
    assert "M20 specialist training completed" in result.stdout
    assert "winner model name: autogluon_tabular" in result.stdout
