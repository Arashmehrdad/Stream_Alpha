"""Operator helper for comparing M20 confirmation-window slice evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_confirmation_compare import compare_m20_confirmation_slices


def main() -> None:
    """Compare original and confirmation conditional usefulness slices."""
    parser = argparse.ArgumentParser(description="Compare M20 confirmation slices")
    parser.add_argument("--original-run-dir", required=True, help="Original M20 research run")
    parser.add_argument("--confirmation-run-dir", required=True, help="Confirmation M20 run")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    arguments = parser.parse_args()
    result = compare_m20_confirmation_slices(
        original_run_dir=Path(arguments.original_run_dir),
        confirmation_run_dir=Path(arguments.confirmation_run_dir),
    )
    if arguments.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"status={result['status']}")
    print(f"original_run_dir={result['original_run_dir']}")
    print(f"confirmation_run_dir={result['confirmation_run_dir']}")
    print(f"recommendation={result['recommendation']}")
    if "message" in result:
        print(f"message={result['message']}")


if __name__ == "__main__":
    main()
