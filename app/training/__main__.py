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
    arguments = parser.parse_args()
    try:
        resume_dir = Path(arguments.resume) if arguments.resume else None
        run_training(Path(arguments.config), resume_artifact_dir=resume_dir)
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
