"""Operator helper for research-only offline M20 threshold policy evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.policy_eval import (
    DEFAULT_BASELINE_THRESHOLD,
    DEFAULT_THRESHOLD_GRID,
    analyze_completed_run,
)


def main() -> None:
    """Run research-only policy evaluation for one completed M20 run."""
    parser = argparse.ArgumentParser(
        description="Analyze Stream Alpha completed M20 threshold policy candidates",
    )
    parser.add_argument(
        "--run-dir",
        help="Path to the completed M20 artifact directory. Defaults to the newest M20 run.",
    )
    parser.add_argument(
        "--model-name",
        help="Optional model_name inside oof_predictions.csv. Defaults to summary winner.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        help="Optional threshold grid. Defaults to 0.50 through 0.90 in 0.05 steps.",
    )
    parser.add_argument(
        "--baseline-threshold",
        type=float,
        default=DEFAULT_BASELINE_THRESHOLD,
        help="Baseline probability threshold for comparison. Defaults to 0.50.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the saved policy report as JSON.",
    )
    arguments = parser.parse_args()
    try:
        analysis_summary = analyze_completed_run(
            run_dir=Path(arguments.run_dir) if arguments.run_dir else None,
            thresholds=arguments.thresholds or DEFAULT_THRESHOLD_GRID,
            model_name=arguments.model_name,
            baseline_threshold=arguments.baseline_threshold,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if arguments.json:
        print(json.dumps(make_json_safe(analysis_summary), sort_keys=True))
        return

    best_candidate = analysis_summary["best_candidate"]
    print(f"run_dir={analysis_summary['run_dir']}")
    print(f"analysis_dir={analysis_summary['analysis_dir']}")
    print(f"model_name={analysis_summary['model_name']}")
    print(f"baseline_threshold={float(analysis_summary['baseline_threshold']):.2f}")
    print(
        "best_candidate="
        f"threshold_{float(best_candidate['threshold']):.2f}"
        f"(net={float(best_candidate['mean_long_only_net_value_proxy']):.6f},"
        f" trades={int(best_candidate['trade_count'])},"
        f" drawdown={float(best_candidate['max_drawdown_proxy']):.6f})"
    )
    print(
        "delta_vs_baseline_mean_net="
        f"{float(best_candidate['delta_vs_baseline_mean_long_only_net_value_proxy']):.6f}"
    )


if __name__ == "__main__":
    main()
