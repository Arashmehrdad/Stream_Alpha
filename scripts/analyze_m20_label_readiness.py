"""Operator helper for research-only M20 label readiness diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.label_readiness import analyze_label_readiness


def main() -> None:
    """Analyze research label readiness for one completed M20 run."""
    parser = argparse.ArgumentParser(
        description="Analyze Stream Alpha M20 research-label readiness",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed M20 run directory containing research_labels/.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the readiness report as JSON.",
    )
    arguments = parser.parse_args()
    try:
        report = analyze_label_readiness(run_dir=Path(arguments.run_dir))
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if arguments.json:
        print(json.dumps(make_json_safe(report), sort_keys=True))
        return

    print(f"run_dir={report['run_dir']}")
    print(f"readiness_dir={report['readiness_dir']}")
    print(f"honesty_flags={','.join(report['honesty_flags']) or 'none'}")
    print(f"recommendation={report['recommendation']}")
    print(f"triple_ready={report['triple_barrier']['ready']}")
    print(f"fee_ready={report['fee_exceedance']['ready']}")
    print(f"meta_ready={report['meta_label']['ready']}")


if __name__ == "__main__":
    main()
