"""Operator helper for tiny research-only M20 baseline diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_baseline_research import train_completed_run_baselines


def main() -> None:
    """Train/evaluate tiny research-only M20 baselines."""
    parser = argparse.ArgumentParser(
        description="Train Stream Alpha M20 tiny research baselines",
    )
    parser.add_argument("--run-dir", required=True, help="Completed M20 run directory.")
    parser.add_argument("--json", action="store_true", help="Emit report JSON.")
    arguments = parser.parse_args()
    try:
        report = train_completed_run_baselines(run_dir=Path(arguments.run_dir))
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if arguments.json:
        print(json.dumps(make_json_safe(report), sort_keys=True))
        return

    best = max(
        report["baselines"],
        key=lambda row: float(row["metrics"]["balanced_accuracy"]),
        default=None,
    )
    print(f"run_dir={report['run_dir']}")
    print(f"baseline_dir={report['baseline_dir']}")
    print(f"honesty_flags={','.join(report['honesty_flags']) or 'none'}")
    print(f"recommendation={report['recommendation']}")
    if best:
        print(
            "best_baseline="
            f"{best['baseline_name']}"
            f"(balanced_accuracy={float(best['metrics']['balanced_accuracy']):.6f})"
        )


if __name__ == "__main__":
    main()
