"""Additive M7 retraining workflow around the accepted M3 trainer."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.training.compare import compare_run_to_current, write_comparison_artifact
from app.training.registry import write_run_manifest
from app.training.service import run_training


def run_retraining(config_path: Path) -> Path:
    """Run the existing trainer with M7 config and write comparison metadata."""
    resolved_config_path = Path(config_path).resolve()
    artifact_dir = run_training(resolved_config_path)
    write_run_manifest(artifact_dir)
    comparison_payload = compare_run_to_current(
        run_dir=artifact_dir,
        config_path=resolved_config_path,
    )
    write_comparison_artifact(artifact_dir, comparison_payload)
    return artifact_dir


def main() -> None:
    """Run one explicit M7 retraining job from the checked-in config."""
    parser = argparse.ArgumentParser(description="Run the Stream Alpha M7 retraining flow")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/training.m7.json",
    )
    arguments = parser.parse_args()

    try:
        artifact_dir = run_retraining(Path(arguments.config))
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(str(artifact_dir))


if __name__ == "__main__":
    main()
