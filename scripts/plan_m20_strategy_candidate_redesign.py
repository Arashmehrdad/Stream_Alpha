"""Operator helper for the M20 generic strategy candidate redesign plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_strategy_candidate_redesign_plan import (
    DEFAULT_OUTPUT_NAME,
    plan_m20_strategy_candidate_redesign,
)


def main() -> None:
    """Write the M20 strategy candidate redesign plan."""
    parser = argparse.ArgumentParser(
        description="Plan generic M20 strategy candidate redesign definitions"
    )
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--prediction-run-dir")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = plan_m20_strategy_candidate_redesign(
        source_run_dir=Path(args.source_run_dir),
        prediction_run_dir=Path(args.prediction_run_dir) if args.prediction_run_dir else None,
        output_name=args.output_name,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"candidate_definition_count={result['candidate_definition_count']}")
    print(f"ready_definition_count={result['ready_definition_count']}")
    print(f"blocked_definition_count={result['blocked_definition_count']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
