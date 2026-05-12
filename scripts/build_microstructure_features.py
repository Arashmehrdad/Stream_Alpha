"""Build research-only microstructure features from fixture replay rows."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.training.microstructure_feature_builder import (  # pylint: disable=wrong-import-position
    DEFAULT_OUTPUT_DIR,
    write_microstructure_features,
)


def main() -> None:
    """Write microstructure feature artifacts."""
    parser = argparse.ArgumentParser(description="Build fixture microstructure features")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    result = write_microstructure_features(repo_root=repo_root, output_dir=output_dir)
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"feature_build_status={result['feature_build_status']}")
    print(f"feature_row_count={result['feature_row_count']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
