"""Operator helper for M20 cost-aware specialist policy adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_cost_aware_policy_adjudication import (
    DEFAULT_OUTPUT_NAME,
    write_m20_cost_aware_policy_adjudication,
)


def main() -> None:
    """Write cost-aware specialist policy adjudication artifacts."""
    parser = argparse.ArgumentParser(
        description="Write M20 cost-aware specialist policy adjudication"
    )
    parser.add_argument("--prediction-run-dir", required=True)
    parser.add_argument("--policy-evaluator-dir")
    parser.add_argument("--edge-evaluator-dir")
    parser.add_argument("--adjudication-output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = write_m20_cost_aware_policy_adjudication(
            prediction_run_dir=Path(args.prediction_run_dir),
            policy_evaluator_dir=(
                Path(args.policy_evaluator_dir) if args.policy_evaluator_dir else None
            ),
            edge_evaluator_dir=(
                Path(args.edge_evaluator_dir) if args.edge_evaluator_dir else None
            ),
            adjudication_output_name=args.adjudication_output_name,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"overall_decision={result['overall_decision']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
