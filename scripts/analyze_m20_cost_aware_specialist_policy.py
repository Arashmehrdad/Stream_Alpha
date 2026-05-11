"""Operator helper for M20 cost-aware specialist policy evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_cost_aware_specialist_policy_evaluator import (
    DEFAULT_LABEL_FILE,
    DEFAULT_OUTPUT_NAME,
    analyze_m20_cost_aware_specialist_policy,
)


def main() -> None:
    """Run generic cost-aware specialist policy diagnostics."""
    parser = argparse.ArgumentParser(
        description="Analyze M20 cost-aware specialist policy candidates"
    )
    parser.add_argument("--prediction-run-dir", required=True)
    parser.add_argument("--label-source-run-dir", required=True)
    parser.add_argument("--prediction-source", required=True)
    parser.add_argument("--models")
    parser.add_argument("--edge-evaluator-dir")
    parser.add_argument("--economic-outcome-dir")
    parser.add_argument("--label-file", default=DEFAULT_LABEL_FILE)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    models = (
        [model.strip() for model in args.models.split(",")]
        if args.models is not None
        else None
    )
    try:
        result = analyze_m20_cost_aware_specialist_policy(
            prediction_run_dir=Path(args.prediction_run_dir),
            label_source_run_dir=Path(args.label_source_run_dir),
            prediction_source=args.prediction_source,
            models=models,
            edge_evaluator_dir=(
                Path(args.edge_evaluator_dir)
                if args.edge_evaluator_dir is not None
                else None
            ),
            economic_outcome_dir=(
                Path(args.economic_outcome_dir)
                if args.economic_outcome_dir is not None
                else None
            ),
            label_file=args.label_file,
            output_name=args.output_name,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"best_policy_candidate={result['best_policy_candidate']}")
    print(f"economics_available={result['economics_available']}")
    print(f"recommendation={result['recommendation']}")


if __name__ == "__main__":
    main()
