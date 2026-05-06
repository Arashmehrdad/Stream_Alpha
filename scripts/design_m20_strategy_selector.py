"""Operator helper for research-only M20 selector design artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_strategy_selector_design import design_m20_strategy_selector


def main() -> None:
    """Create a research-only M20 strategy selector design artifact."""
    parser = argparse.ArgumentParser(description="Design M20 research selector")
    parser.add_argument("--original-run-dir", required=True)
    parser.add_argument("--confirmation-run-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = design_m20_strategy_selector(
        original_run_dir=Path(args.original_run_dir),
        confirmation_run_dir=Path(args.confirmation_run_dir),
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"selector_id={result['selector_id']}")
    print(f"design_dir={result['design_dir']}")
    print(f"confirmed_conditions={result['confirmed_condition_count']}")
    print(f"watchlist_conditions={result['watchlist_condition_count']}")
    print(f"disable_gaps={result['disable_gap_count']}")
    print(f"recommendation={result['recommendation']}")


if __name__ == "__main__":
    main()
