"""Operator helper for M20 redesigned research inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_redesigned_research_inputs import build_m20_redesigned_research_inputs


def main() -> None:
    """Build M20 redesigned research inputs."""
    parser = argparse.ArgumentParser(description="Build M20 redesigned research inputs")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--output-name", default="m20_redesigned_research_inputs")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build_m20_redesigned_research_inputs(
        source_run_dir=Path(args.source_run_dir),
        output_name=args.output_name,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"row_count={result['row_count']}")
    print(f"recommendation={result['recommendation']}")


if __name__ == "__main__":
    main()
