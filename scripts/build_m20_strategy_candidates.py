"""Operator helper for M20 generic strategy candidate factory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_strategy_candidate_factory import (
    DEFAULT_OUTPUT_NAME,
    build_m20_strategy_candidates,
)


def main() -> None:
    """Build generic M20 strategy-conditioned candidate artifacts."""
    parser = argparse.ArgumentParser(description="Build M20 strategy candidates")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--economic-outcome-dir")
    parser.add_argument("--training-frame-dir")
    parser.add_argument("--label-source-run-dir")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build_m20_strategy_candidates(
        source_run_dir=Path(args.source_run_dir),
        economic_outcome_dir=(
            Path(args.economic_outcome_dir) if args.economic_outcome_dir else None
        ),
        training_frame_dir=(
            Path(args.training_frame_dir) if args.training_frame_dir else None
        ),
        label_source_run_dir=(
            Path(args.label_source_run_dir) if args.label_source_run_dir else None
        ),
        output_name=args.output_name,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"best_candidate={result['best_candidate']}")
    print(f"candidate_count={result['candidate_count']}")
    print(f"recommendation={result['recommendation']}")


if __name__ == "__main__":
    main()
