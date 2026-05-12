"""Operator helper for the isolated research microstructure capture plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe  # pylint: disable=wrong-import-position
from app.training.market_microstructure_capture_plan import (  # pylint: disable=wrong-import-position
    DEFAULT_OUTPUT_DIR,
    write_microstructure_capture_plan,
)


def main() -> None:
    """Write the isolated capture planning artifact set."""
    parser = argparse.ArgumentParser(description="Plan isolated microstructure capture")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    result = write_microstructure_capture_plan(repo_root=repo_root, output_dir=output_dir)
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"capture_plan_status={result['capture_plan_status']}")
    print(f"capture_implemented={result['capture_implemented']}")
    print(f"runtime_wiring_changed={result['runtime_wiring_changed']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
