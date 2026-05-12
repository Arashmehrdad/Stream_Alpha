"""Write research-only microstructure storage contracts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.training.microstructure_storage_contracts import (  # pylint: disable=wrong-import-position
    DEFAULT_OUTPUT_DIR,
    write_microstructure_storage_contracts,
)


def main() -> None:
    """Run the dry-run schema contract writer."""
    parser = argparse.ArgumentParser(description="Prepare microstructure research schema")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--allow-apply", action="store_true")
    parser.add_argument("--dsn", default="")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    result = write_microstructure_storage_contracts(
        repo_root=repo_root,
        output_dir=output_dir,
        apply=args.apply,
        allow_apply=args.allow_apply,
        dsn=args.dsn or None,
    )
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"storage_contract_status={result['storage_contract_status']}")
    print(f"ddl_apply_executed={result['ddl_apply_executed']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
