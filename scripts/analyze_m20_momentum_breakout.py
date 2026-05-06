"""Operator helper for research-only M20 momentum-breakout diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_momentum_breakout_research import analyze_m20_momentum_breakout


def main() -> None:
    """Run M20 momentum-breakout research diagnostics."""
    parser = argparse.ArgumentParser(description="Analyze M20 momentum-breakout setups")
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = analyze_m20_momentum_breakout(base_run_dir=Path(args.base_run_dir))
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"setup_count={result.get('setup_count', 0)}")
    print(f"recommendation={result['recommendation']}")


if __name__ == "__main__":
    main()
