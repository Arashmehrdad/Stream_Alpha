"""Operator helper for M20 research feature-matrix alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_feature_matrix import build_m20_research_feature_matrix


def main() -> None:
    """Build a research-only M20 feature matrix when safe alignment exists."""
    parser = argparse.ArgumentParser(
        description="Audit and build Stream Alpha M20 research feature matrix",
    )
    parser.add_argument("--run-dir", required=True, help="Completed M20 run directory.")
    parser.add_argument("--json", action="store_true", help="Emit report JSON.")
    arguments = parser.parse_args()
    try:
        report = build_m20_research_feature_matrix(run_dir=Path(arguments.run_dir))
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if arguments.json:
        print(json.dumps(make_json_safe(report), sort_keys=True))
        return

    alignment = report["alignment_report"]
    print(f"run_dir={report['run_dir']}")
    print(f"feature_matrix_dir={report['feature_matrix_dir']}")
    print(f"honesty_flags={','.join(report['honesty_flags']) or 'none'}")
    print(f"recommendation={report['recommendation']}")
    print(f"join_keys={'+'.join(alignment['join_keys_used']) or 'none'}")
    print(f"matched_rows={alignment['matched_rows']}")
    print(f"safe_numeric_feature_count={alignment['safe_numeric_feature_count']}")


if __name__ == "__main__":
    main()
