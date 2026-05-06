"""Operator helper for research-only M20 fee-exceedance baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_fee_exceedance_baseline import (
    DEFAULT_SCENARIO_NAME,
    train_fee_exceedance_baselines,
)


def main() -> None:
    """Train tiny research-only fee-exceedance baselines."""
    parser = argparse.ArgumentParser(
        description="Train Stream Alpha M20 research-only fee-exceedance baselines",
    )
    parser.add_argument("--run-dir", required=True, help="M20 run directory")
    parser.add_argument(
        "--scenario-name",
        default=DEFAULT_SCENARIO_NAME,
        help="Fee scenario to use. Defaults to current_fee.",
    )
    parser.add_argument(
        "--export-full-predictions",
        action="store_true",
        help="Export full train/validation/test prediction rows for the tiny baseline.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    arguments = parser.parse_args()
    try:
        result = train_fee_exceedance_baselines(
            run_dir=Path(arguments.run_dir),
            scenario_name=arguments.scenario_name,
            export_full_predictions=arguments.export_full_predictions,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if arguments.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    best = max(
        result["baselines"],
        key=lambda row: (row.get("average_precision") or 0.0, row.get("balanced_accuracy") or 0.0),
        default={},
    )
    print(f"run_dir={result['run_dir']}")
    print(f"baseline_dir={result['baseline_dir']}")
    print(f"scenario_name={result['scenario_name']}")
    print(f"honesty_flags={','.join(result['honesty_flags'])}")
    print(f"best_baseline={best.get('baseline_name', 'none')}")
    print(f"best_average_precision={float(best.get('average_precision') or 0.0):.6f}")
    print(f"best_balanced_accuracy={float(best.get('balanced_accuracy') or 0.0):.6f}")
    print(f"recommendation={result['recommendation']}")


if __name__ == "__main__":
    main()
