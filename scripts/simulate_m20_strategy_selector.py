"""Operator helper for research-only M20 selector simulations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_strategy_selector_simulation import simulate_m20_strategy_selector


def main() -> None:
    """Run the offline research-only selector simulation."""
    parser = argparse.ArgumentParser(description="Simulate M20 research selector")
    parser.add_argument("--original-run-dir", required=True)
    parser.add_argument("--confirmation-run-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = simulate_m20_strategy_selector(
        original_run_dir=Path(args.original_run_dir),
        confirmation_run_dir=Path(args.confirmation_run_dir),
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"selector_id={result['selector_id']}")
    print(f"simulation_dir={result['simulation_dir']}")
    print(f"policy_count={result['policy_count']}")
    print(f"recommendation={result['recommendation']}")


if __name__ == "__main__":
    main()
