"""Operator helper for auditing M20 decision-policy inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_policy_input_availability import (
    DEFAULT_OUTPUT_NAME,
    audit_m20_policy_inputs,
)


def main() -> None:
    """Audit available M20 policy inputs."""
    parser = argparse.ArgumentParser(description="Audit M20 policy input availability")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--prediction-run-dir", required=True)
    parser.add_argument("--candidate-dir")
    parser.add_argument("--economic-outcome-dir")
    parser.add_argument("--research-feature-dir")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit_m20_policy_inputs(
        source_run_dir=Path(args.source_run_dir),
        prediction_run_dir=Path(args.prediction_run_dir),
        candidate_dir=Path(args.candidate_dir) if args.candidate_dir else None,
        economic_outcome_dir=(
            Path(args.economic_outcome_dir) if args.economic_outcome_dir else None
        ),
        research_feature_dir=(
            Path(args.research_feature_dir) if args.research_feature_dir else None
        ),
        output_name=args.output_name,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"policy_evaluation_ready={result['policy_evaluation_ready']}")
    print(f"missing_input_count={result['missing_input_count']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
