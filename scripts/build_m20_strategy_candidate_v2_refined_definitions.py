"""Operator helper for M20 refined v2 strategy definitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_strategy_candidate_v2_refined_definitions import (
    DEFAULT_OUTPUT_NAME,
    build_m20_strategy_candidate_v2_refined_definitions,
)


def main() -> None:
    """Write refined M20 v2 definition artifacts."""
    parser = argparse.ArgumentParser(description="Build M20 refined v2 definitions")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--refinement-plan-dir")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build_m20_strategy_candidate_v2_refined_definitions(
        source_run_dir=Path(args.source_run_dir),
        refinement_plan_dir=(
            Path(args.refinement_plan_dir) if args.refinement_plan_dir else None
        ),
        output_name=args.output_name,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"definition_count={result['definition_count']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
