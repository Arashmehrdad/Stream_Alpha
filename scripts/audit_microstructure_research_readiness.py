"""Audit microstructure research readiness."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.training.microstructure_research_readiness import (  # pylint: disable=wrong-import-position
    DEFAULT_OUTPUT_DIR,
    write_microstructure_research_readiness,
)


def main() -> None:
    """Write research readiness artifacts."""
    parser = argparse.ArgumentParser(description="Audit microstructure research readiness")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    result = write_microstructure_research_readiness(repo_root=repo_root, output_dir=output_dir)
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"readiness_status={result['readiness_status']}")
    print(f"alpha_research_reopen_ready={result['alpha_research_reopen_ready']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
