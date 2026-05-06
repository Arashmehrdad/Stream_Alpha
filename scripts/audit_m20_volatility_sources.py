"""Operator helper for research-only M20 volatility source audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.volatility_audit import audit_completed_run_volatility_sources


def main() -> None:
    """Audit volatility sources and generate vol-scaled labels when possible."""
    parser = argparse.ArgumentParser(
        description="Audit Stream Alpha M20 volatility sources for research labels",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Completed M20 run directory.",
    )
    parser.add_argument(
        "--model-name",
        help="Optional model_name inside oof_predictions.csv. Defaults to summary winner.",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=12,
        help="Rolling lookback for computed research volatility proxy.",
    )
    parser.add_argument("--json", action="store_true", help="Emit audit JSON.")
    arguments = parser.parse_args()
    try:
        audit = audit_completed_run_volatility_sources(
            run_dir=Path(arguments.run_dir),
            model_name=arguments.model_name,
            lookback=arguments.lookback,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if arguments.json:
        print(json.dumps(make_json_safe(audit), sort_keys=True))
        return

    print(f"run_dir={audit['run_dir']}")
    print(f"audit_dir={audit['audit_dir']}")
    print(f"model_name={audit['model_name']}")
    print(f"volatility_source={audit['volatility_source']}")
    print(f"selected_volatility_column={audit['selected_volatility_column']}")
    print(f"honesty_flags={','.join(audit['honesty_flags']) or 'none'}")
    print(f"recommendation={audit['recommendation']}")


if __name__ == "__main__":
    main()
