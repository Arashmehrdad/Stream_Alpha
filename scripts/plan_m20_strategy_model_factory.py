"""Operator helper for the M20 strategy-conditioned model factory plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_strategy_model_factory_plan import (
    DEFAULT_OUTPUT_NAME,
    plan_m20_strategy_model_factory,
)


def main() -> None:
    """Write the M20 strategy-conditioned model factory plan."""
    parser = argparse.ArgumentParser(
        description="Plan a generic M20 strategy-conditioned model factory"
    )
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--slice-policy-dir")
    parser.add_argument("--refinement-dir")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = plan_m20_strategy_model_factory(
        source_run_dir=Path(args.source_run_dir),
        slice_policy_dir=Path(args.slice_policy_dir) if args.slice_policy_dir else None,
        refinement_dir=Path(args.refinement_dir) if args.refinement_dir else None,
        output_name=args.output_name,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"candidate_input_count={result['candidate_input_count']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
