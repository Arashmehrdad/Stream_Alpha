"""Operator helper for M20 generic strategy slice policy evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_strategy_slice_policy_evaluator import (
    DEFAULT_OUTPUT_NAME,
    analyze_m20_strategy_slice_policy,
)


def main() -> None:
    """Run M20 strategy slice policy evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate generic M20 strategy slice policies"
    )
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--refinement-dir")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--min-policy-rows", type=int, default=100)
    parser.add_argument("--max-tail-loss-rate", type=float, default=0.75)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = analyze_m20_strategy_slice_policy(
        source_run_dir=Path(args.source_run_dir),
        refinement_dir=Path(args.refinement_dir) if args.refinement_dir else None,
        output_name=args.output_name,
        min_policy_rows=args.min_policy_rows,
        max_tail_loss_rate=args.max_tail_loss_rate,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"policy_count={result['policy_count']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
