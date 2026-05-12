"""Operator helper for M20 research input catalogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_research_input_catalogue import build_m20_research_input_catalogue


def main() -> None:
    """Build M20 research input catalogue."""
    parser = argparse.ArgumentParser(description="Build M20 research input catalogue")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--prediction-run-dir")
    parser.add_argument("--output-name", default="m20_research_input_catalogue")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build_m20_research_input_catalogue(
        source_run_dir=Path(args.source_run_dir),
        prediction_run_dir=Path(args.prediction_run_dir) if args.prediction_run_dir else None,
        output_name=args.output_name,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"safe_computable_blocked_labels={result['safe_computable_blocked_labels']}")
    print(f"recommendation={result['recommendation']}")


if __name__ == "__main__":
    main()
