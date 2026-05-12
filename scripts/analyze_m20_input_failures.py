"""Operator helper for M20 input failure analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_input_failure_analysis import analyze_m20_input_failures


def main() -> None:
    """Run M20 input failure analysis."""
    parser = argparse.ArgumentParser(description="Analyze M20 input failures")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--output-name", default="m20_input_failure_analysis")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = analyze_m20_input_failures(
        source_run_dir=Path(args.source_run_dir),
        output_name=args.output_name,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
