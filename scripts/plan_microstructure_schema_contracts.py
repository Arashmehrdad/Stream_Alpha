"""CLI for writing research-only microstructure schema and replay contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe  # pylint: disable=wrong-import-position
from app.training.market_microstructure_schema_contracts import (  # pylint: disable=wrong-import-position
    DEFAULT_OUTPUT_DIR,
    write_market_microstructure_schema_contracts,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write research-only market microstructure schema contracts.",
    )
    parser.add_argument(
        "--repo-root",
        default=str(REPO_ROOT),
        help="Repository root. Defaults to the current checkout.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Optional override for the artifact output directory.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    """Write the contract artifacts and print the key result fields."""
    args = _parser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    result = write_market_microstructure_schema_contracts(
        repo_root=repo_root,
        output_dir=output_dir,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"contract_status={result['contract_status']}")
    print(f"table_contract_count={result['table_contract_count']}")
    print(f"blocked_decision_count={result['blocked_decision_count']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
