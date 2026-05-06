"""Operator helper for M20 conditional usefulness diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_conditional_usefulness import analyze_conditional_usefulness


def main() -> None:
    """Analyze conditional usefulness for a research-only M20 fee baseline."""
    parser = argparse.ArgumentParser(
        description="Analyze M20 fee-exceedance baseline conditional usefulness",
    )
    parser.add_argument("--run-dir", required=True, help="M20 run directory")
    parser.add_argument(
        "--prediction-source",
        choices=("auto", "full-test", "sampled-test"),
        default="auto",
        help="Prediction source preference. Defaults to full-test when available.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    arguments = parser.parse_args()
    try:
        result = analyze_conditional_usefulness(
            run_dir=Path(arguments.run_dir),
            prediction_source=arguments.prediction_source,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if arguments.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    breadth = result["search_breadth"]
    print(f"run_dir={result['run_dir']}")
    print(f"conditional_dir={result['conditional_dir']}")
    print(f"baseline_name={result['baseline_name']}")
    print(f"prediction_rows_analyzed={result['prediction_rows_analyzed']}")
    print(f"prediction_source={result['prediction_source']}")
    print(f"slice_count={breadth['slice_count']}")
    print(f"enable_candidates={breadth['enable_candidate_count']}")
    print(f"watchlist_candidates={breadth['watchlist_candidate_count']}")
    print(f"disable_candidates={breadth['disable_candidate_count']}")
    print(f"honesty_flags={','.join(result['honesty_flags'])}")
    print(f"recommendation={result['recommendation']}")


if __name__ == "__main__":
    main()
