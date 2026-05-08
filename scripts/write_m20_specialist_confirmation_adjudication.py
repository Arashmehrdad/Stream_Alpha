"""Operator helper for M20 specialist confirmation adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_specialist_confirmation_adjudication import (
    write_m20_specialist_confirmation_adjudication,
)


def main() -> None:
    """Write M20 specialist confirmation adjudication artifacts."""
    parser = argparse.ArgumentParser(
        description="Write M20 specialist confirmation adjudication"
    )
    parser.add_argument("--confirmation-run-dir", required=True)
    parser.add_argument("--original-run-dir")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = write_m20_specialist_confirmation_adjudication(
        confirmation_run_dir=Path(args.confirmation_run_dir),
        original_run_dir=(
            None if args.original_run_dir is None else Path(args.original_run_dir)
        ),
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"overall_status={result['overall_status']}")
    print(f"patchtst_decision={result['patchtst_decision']}")
    print(f"nhits_decision={result['nhits_decision']}")


if __name__ == "__main__":
    main()
