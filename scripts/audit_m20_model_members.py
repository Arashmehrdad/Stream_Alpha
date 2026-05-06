"""Operator helper for research-only M20 model/member audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_model_member_audit import audit_m20_model_members


def main() -> None:
    """Audit model/member candidates without running long jobs."""
    parser = argparse.ArgumentParser(description="Audit M20 model/member candidates")
    parser.add_argument("--run-dir", required=True, help="Current research run directory")
    parser.add_argument("--previous-run-dir", help="Previous completed M20 run directory")
    parser.add_argument("--fitted-models-dir", help="Source fitted models directory")
    parser.add_argument(
        "--load-autogluon-metadata",
        action="store_true",
        help="Allow optional metadata-only AutoGluon inspection. Never retrains.",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    arguments = parser.parse_args()
    result = audit_m20_model_members(
        run_dir=Path(arguments.run_dir),
        previous_run_dir=Path(arguments.previous_run_dir) if arguments.previous_run_dir else None,
        fitted_models_dir=Path(arguments.fitted_models_dir) if arguments.fitted_models_dir else None,
        load_autogluon_metadata=arguments.load_autogluon_metadata,
    )
    if arguments.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"run_dir={result['run_dir']}")
    print(f"audit_dir={result['audit_dir']}")
    print(f"candidate_count={result['candidate_count']}")
    print(f"autogluon_member_count={result['autogluon_member_count']}")
    print(f"honesty_flags={','.join(result['honesty_flags'])}")
    for recommendation in result["recommendations"]:
        print(f"recommendation={recommendation}")


if __name__ == "__main__":
    main()
