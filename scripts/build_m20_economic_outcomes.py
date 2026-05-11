"""Operator helper for M20 safe economic outcome artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_economic_outcome_artifacts import (
    DEFAULT_OUTPUT_NAME,
    build_m20_economic_outcome_artifacts,
)


def main() -> None:
    """Build research-only safe economic outcome artifacts."""
    parser = argparse.ArgumentParser(description="Build M20 economic outcome artifacts")
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--label-source-run-dir")
    parser.add_argument("--prediction-run-dir")
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--fee-bps", type=float, default=20.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--horizon-candles", type=int)
    parser.add_argument("--price-column", default="close_price")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = build_m20_economic_outcome_artifacts(
        source_run_dir=Path(args.source_run_dir),
        label_source_run_dir=(
            Path(args.label_source_run_dir) if args.label_source_run_dir else None
        ),
        prediction_run_dir=(
            Path(args.prediction_run_dir) if args.prediction_run_dir else None
        ),
        output_name=args.output_name,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        horizon_candles=args.horizon_candles,
        price_column=args.price_column,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"economics_computable={result['economics_computable']}")
    print(f"rows_written={result['rows_written']}")
    print(f"recommendation={result['recommendation']}")


if __name__ == "__main__":
    main()
