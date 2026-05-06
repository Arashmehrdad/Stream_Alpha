"""Operator helper for research-only M20 strategy-family scaffold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_strategy_family_scaffold import design_m20_strategy_families


def main() -> None:
    """Write M20 strategy-family scaffold artifacts."""
    parser = argparse.ArgumentParser(description="Design M20 research strategy families")
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = design_m20_strategy_families(base_run_dir=Path(args.base_run_dir))
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"family_count={result['family_count']}")
    print(f"rank_gate_usage={result['rank_gate_usage']}")


if __name__ == "__main__":
    main()
