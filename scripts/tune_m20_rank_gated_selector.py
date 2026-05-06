"""Operator helper for research-only M20 rank-gated selector evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_rank_gated_selector import tune_m20_rank_gated_selector


def main() -> None:
    """Run the offline research-only rank-gated selector evaluation."""
    parser = argparse.ArgumentParser(description="Tune M20 rank-gated selector")
    parser.add_argument("--original-run-dir", required=True)
    parser.add_argument("--confirmation-run-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = tune_m20_rank_gated_selector(
        original_run_dir=Path(args.original_run_dir),
        confirmation_run_dir=Path(args.confirmation_run_dir),
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={result['output_dir']}")
    print(f"policy_count={result['policy_count']}")
    print(f"best_stable_policy={result['best_stable_policy']}")
    print(f"recommendation={result['recommendation']}")


if __name__ == "__main__":
    main()
