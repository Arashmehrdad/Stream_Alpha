"""Module execution entrypoint for the M3 training pipeline."""

# pylint: disable=duplicate-code

from __future__ import annotations

import argparse
from pathlib import Path

from app.training.service import run_training


def main() -> None:
    """Parse CLI arguments and run the configured offline training job."""
    parser = argparse.ArgumentParser(description="Run Stream Alpha M3 offline training")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON training config file",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to an existing artifact directory to resume from its checkpoint",
    )
    parser.add_argument(
        "--parquet-dir",
        default=None,
        help="Path to exported parquet dataset (skip PostgreSQL loading)",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--fit-only",
        action="store_true",
        default=False,
        help="Fit models only (GPU phase) — save fitted estimators, skip scoring",
    )
    mode_group.add_argument(
        "--score-only",
        default=None,
        metavar="FITTED_MODELS_DIR",
        help="Score only (CPU phase) — load pre-fitted models from this directory",
    )
    parser.add_argument(
        "--export-training-frame",
        action="store_true",
        default=False,
        help="Research-only: export row-level market feature frame into the run artifact",
    )
    parser.add_argument(
        "--export-training-frame-only",
        action="store_true",
        default=False,
        help=(
            "Research-only: export row-level market feature frame and exit "
            "before model scoring"
        ),
    )
    parser.add_argument(
        "--confirmation-window-start",
        default=None,
        help="Research-only confirmation window start timestamp, e.g. 2024-04-02T11:30:00Z",
    )
    parser.add_argument(
        "--confirmation-window-end",
        default=None,
        help="Research-only confirmation window end timestamp, e.g. 2025-04-02T11:30:00Z",
    )
    parser.add_argument(
        "--confirmation-tag",
        default=None,
        help="Safe reporting-only tag for a manual confirmation run",
    )
    arguments = parser.parse_args()
    try:
        resume_dir = Path(arguments.resume) if arguments.resume else None
        parquet_dir = Path(arguments.parquet_dir) if arguments.parquet_dir else None
        score_only_dir = Path(arguments.score_only) if arguments.score_only else None
        run_training(
            Path(arguments.config),
            resume_artifact_dir=resume_dir,
            parquet_dir=parquet_dir,
            fit_only=arguments.fit_only,
            score_only_dir=score_only_dir,
            export_training_frame=(
                arguments.export_training_frame
                or arguments.export_training_frame_only
                or None
            ),
            export_training_frame_only=arguments.export_training_frame_only,
            confirmation_window_start=arguments.confirmation_window_start,
            confirmation_window_end=arguments.confirmation_window_end,
            confirmation_tag=arguments.confirmation_tag,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
