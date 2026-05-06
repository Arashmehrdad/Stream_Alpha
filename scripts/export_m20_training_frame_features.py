"""Operator helper for M20 research training-frame feature exports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_training_frame_export import export_m20_training_frame_features


def main() -> None:
    """Export or report the blocker for M20 row-level market features."""
    parser = argparse.ArgumentParser(
        description="Export Stream Alpha M20 research training-frame features",
    )
    parser.add_argument("--run-dir", required=True, help="Completed M20 run directory.")
    parser.add_argument("--json", action="store_true", help="Emit report JSON.")
    arguments = parser.parse_args()
    report = export_m20_training_frame_features(run_dir=Path(arguments.run_dir))

    if arguments.json:
        print(json.dumps(make_json_safe(report), sort_keys=True))
        return

    details = report["report"]
    print(f"run_dir={report['run_dir']}")
    print(f"export_dir={report['export_dir']}")
    print(f"honesty_flags={','.join(report['honesty_flags']) or 'none'}")
    print(f"recommendation={report['recommendation']}")
    print(f"selected_source={details['selected_source_file'] or 'none'}")
    print(f"safe_market_feature_count={details['safe_market_feature_count']}")
    print(f"row_count={details['row_count']}")


if __name__ == "__main__":
    main()
