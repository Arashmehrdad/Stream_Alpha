"""Operator helper for M20 safe feature availability audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_safe_feature_availability import (
    DEFAULT_OUTPUT_NAME,
    audit_m20_safe_feature_availability,
)


def main() -> None:
    """Write M20 safe feature availability audit artifacts."""
    parser = argparse.ArgumentParser(description="Audit M20 safe feature availability")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--regime-thresholds-path")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit_m20_safe_feature_availability(
        source_run_dir=Path(args.source_run_dir),
        regime_thresholds_path=(
            Path(args.regime_thresholds_path) if args.regime_thresholds_path else None
        ),
        output_name=args.output_name,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"safe_computable_feature_count={result['safe_computable_feature_count']}")
    print(f"blocked_feature_count={result['blocked_feature_count']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
