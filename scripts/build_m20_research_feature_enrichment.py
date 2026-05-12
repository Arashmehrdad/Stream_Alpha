"""Operator helper for M20 research feature enrichment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_research_feature_enrichment import (
    DEFAULT_OUTPUT_NAME,
    build_m20_research_feature_enrichment,
)


def main() -> None:
    """Build M20 research-only feature enrichment artifacts."""
    parser = argparse.ArgumentParser(description="Build M20 research feature enrichment")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--regime-thresholds-path", required=True)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build_m20_research_feature_enrichment(
        source_run_dir=Path(args.source_run_dir),
        regime_thresholds_path=Path(args.regime_thresholds_path),
        output_name=args.output_name,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"rows_written={result['rows_written']}")
    print(f"features_added={','.join(result['added_features'])}")
    print(f"blocked_features={len(result['blocked_features'])}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
