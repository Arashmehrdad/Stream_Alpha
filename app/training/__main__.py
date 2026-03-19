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
    arguments = parser.parse_args()
    try:
        run_training(Path(arguments.config))
    except ValueError as error:
        raise SystemExit(str(error)) from error


if __name__ == "__main__":
    main()
