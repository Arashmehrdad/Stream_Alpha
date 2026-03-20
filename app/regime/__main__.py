"""Module execution entrypoint for the M8 offline regime workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.regime.service import run_regime_workflow


def main() -> None:
    """Parse CLI arguments and run the configured offline regime job."""
    parser = argparse.ArgumentParser(
        description="Run the Stream Alpha M8 offline regime workflow"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON regime config file",
    )
    arguments = parser.parse_args()

    try:
        artifact_dir = run_regime_workflow(Path(arguments.config))
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(str(artifact_dir))


if __name__ == "__main__":
    main()
