"""Operator helper for evaluating M20 research decision policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_decision_policy_evaluator import (
    DEFAULT_OUTPUT_NAME,
    evaluate_m20_decision_policies,
)


def main() -> None:
    """Evaluate M20 decision policies."""
    parser = argparse.ArgumentParser(description="Evaluate M20 decision policies")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--prediction-run-dir", required=True)
    parser.add_argument("--candidate-dir")
    parser.add_argument("--economic-outcome-dir")
    parser.add_argument("--trading-aware-label-dir")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = evaluate_m20_decision_policies(
        source_run_dir=Path(args.source_run_dir),
        prediction_run_dir=Path(args.prediction_run_dir),
        candidate_dir=Path(args.candidate_dir) if args.candidate_dir else None,
        economic_outcome_dir=(
            Path(args.economic_outcome_dir) if args.economic_outcome_dir else None
        ),
        trading_aware_label_dir=(
            Path(args.trading_aware_label_dir) if args.trading_aware_label_dir else None
        ),
        output_name=args.output_name,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"policy_count={result['policy_count']}")
    print(f"best_policy_candidate={result['best_policy_candidate']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
