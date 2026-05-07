"""Operator helper for M20 specialist confirmation planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_specialist_confirmation_plan import (
    write_m20_specialist_confirmation_plan,
)


def main() -> None:
    """Write the specialist confirmation plan."""
    parser = argparse.ArgumentParser(description="Plan M20 specialist confirmation export")
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--previous-run-dir", required=True)
    parser.add_argument("--fitted-models-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = write_m20_specialist_confirmation_plan(
        base_run_dir=Path(args.base_run_dir),
        previous_run_dir=Path(args.previous_run_dir),
        fitted_models_dir=Path(args.fitted_models_dir),
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"recommendation={result['recommendation']}")
    print(f"primary_candidate={result['primary_candidate']}")


if __name__ == "__main__":
    main()
