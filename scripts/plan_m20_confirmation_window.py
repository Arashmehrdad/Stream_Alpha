"""Operator helper for manual-only M20 confirmation-window planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_confirmation_planner import plan_m20_confirmation_window


def main() -> None:
    """Write confirmation plan artifacts without launching long jobs."""
    parser = argparse.ArgumentParser(description="Plan M20 confirmation-window workflow")
    parser.add_argument("--run-dir", required=True, help="Current M20 research run directory")
    parser.add_argument("--config-path", help="M20 config path to inspect")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    arguments = parser.parse_args()
    result = plan_m20_confirmation_window(
        run_dir=Path(arguments.run_dir),
        config_path=Path(arguments.config_path) if arguments.config_path else None,
    )
    if arguments.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"run_dir={result['run_dir']}")
    print(f"confirmation_plan_dir={result['confirmation_plan_dir']}")
    print(f"candidate_id={result['candidate_id']}")
    print(f"baseline_name={result['baseline_name']}")
    print(f"target_slice_count={result['target_slice_count']}")
    print(f"window_override_supported={result['window_override_support']['supported']}")
    print(f"honesty_flags={','.join(result['honesty_flags'])}")
    for recommendation in result["recommendation"]:
        print(f"recommendation={recommendation}")


if __name__ == "__main__":
    main()
