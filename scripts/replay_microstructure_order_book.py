"""Run fixture-backed microstructure replay artifact generation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.training.microstructure_replay import (  # pylint: disable=wrong-import-position
    DEFAULT_OUTPUT_DIR,
    write_microstructure_replay,
)


def main() -> None:
    """Write replay/gap/determinism artifacts."""
    parser = argparse.ArgumentParser(description="Replay fixture microstructure order book")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    result = write_microstructure_replay(repo_root=repo_root, output_dir=output_dir)
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"replay_status={result['replay_status']}")
    print(f"replay_row_count={result['replay_row_count']}")
    print(f"gap_count={result['gap_count']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
