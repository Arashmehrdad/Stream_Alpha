"""Operator helper for M20 generic strategy candidate refinement analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_strategy_candidate_refinement import (
    DEFAULT_OUTPUT_NAME,
    analyze_m20_strategy_candidate_refinement,
)


def main() -> None:
    """Run M20 strategy candidate refinement analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze generic M20 strategy candidate refinements"
    )
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--candidate-factory-dir")
    parser.add_argument("--training-frame-dir")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--min-slice-rows", type=int, default=100)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = analyze_m20_strategy_candidate_refinement(
        source_run_dir=Path(args.source_run_dir),
        candidate_factory_dir=(
            Path(args.candidate_factory_dir) if args.candidate_factory_dir else None
        ),
        training_frame_dir=Path(args.training_frame_dir) if args.training_frame_dir else None,
        output_name=args.output_name,
        min_slice_rows=args.min_slice_rows,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"candidate_count={result['candidate_count']}")
    print(f"slice_count={result['slice_count']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
