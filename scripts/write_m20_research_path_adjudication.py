"""Operator helper for research-only M20 path adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_research_path_adjudication import (
    write_m20_research_path_adjudication,
)


def main() -> None:
    """Write M20 research path adjudication artifacts."""
    parser = argparse.ArgumentParser(description="Write M20 research path adjudication")
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = write_m20_research_path_adjudication(base_run_dir=Path(args.base_run_dir))
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"decision={result['decision']}")
    print(f"recommended_next_action={result['recommended_next_action']}")


if __name__ == "__main__":
    main()
