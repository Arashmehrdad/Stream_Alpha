"""Operator helper for M20 policy validation audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_policy_validation_audit import (
    DEFAULT_OUTPUT_NAME,
    audit_m20_policy_validation,
)


def main() -> None:
    """Audit M20 policy validation readiness."""
    parser = argparse.ArgumentParser(description="Audit M20 policy validation")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--policy-eval-dir")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = audit_m20_policy_validation(
        source_run_dir=Path(args.source_run_dir),
        policy_eval_dir=Path(args.policy_eval_dir) if args.policy_eval_dir else None,
        output_name=args.output_name,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"policy_count={result['policy_count']}")
    print(f"recommendation={result['recommendation']}")
    print(f"next_required_action={result['next_required_action']}")


if __name__ == "__main__":
    main()
