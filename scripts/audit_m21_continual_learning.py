"""Operator helper for M21 continual-learning guarded-workflow audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe  # pylint: disable=wrong-import-position
from app.continual_learning.m21_continual_learning_audit import (  # pylint: disable=wrong-import-position
    DEFAULT_OUTPUT_DIR,
    audit_m21_continual_learning,
)


def main() -> None:
    """Write the M21 continual-learning audit artifact set."""
    parser = argparse.ArgumentParser(
        description="Audit M21 continual-learning guarded workflow"
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    result = audit_m21_continual_learning(repo_root=repo_root, output_dir=output_dir)
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"m21_state={result['m21_state']}")
    print(f"gap_count={result['gap_count']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
