"""Operator helper for research-only M20 OOF signal diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.oof_signal_diagnostics import (
    DEFAULT_DIAGNOSTIC_THRESHOLDS,
    diagnose_completed_run,
)


def main() -> None:
    """Run research-only OOF diagnostics for one completed M20 run."""
    parser = argparse.ArgumentParser(
        description="Diagnose Stream Alpha completed M20 OOF signal distribution",
    )
    parser.add_argument(
        "--run-dir",
        help="Path to the completed M20 artifact directory. Defaults to the newest M20 run.",
    )
    parser.add_argument(
        "--model-name",
        help="Optional model_name inside oof_predictions.csv. Defaults to summary winner.",
    )
    parser.add_argument(
        "--score-columns",
        nargs="+",
        help="Optional score columns to inspect. Defaults to discovered probability/score columns.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        help="Optional thresholds. Defaults to 0.05 through 0.95 in 0.05 steps.",
    )
    parser.add_argument("--symbol", help="Optional symbol filter for focused diagnostics.")
    parser.add_argument("--fold-index", type=int, help="Optional fold-index filter.")
    parser.add_argument("--regime-label", help="Optional regime-label filter.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the saved diagnostics summary as JSON.",
    )
    arguments = parser.parse_args()
    try:
        diagnostics = diagnose_completed_run(
            run_dir=Path(arguments.run_dir) if arguments.run_dir else None,
            model_name=arguments.model_name,
            score_columns=arguments.score_columns,
            thresholds=arguments.thresholds or DEFAULT_DIAGNOSTIC_THRESHOLDS,
            symbol=arguments.symbol,
            fold_index=arguments.fold_index,
            regime_label=arguments.regime_label,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if arguments.json:
        print(json.dumps(make_json_safe(diagnostics), sort_keys=True))
        return

    print(f"run_dir={diagnostics['run_dir']}")
    print(f"diagnostics_dir={diagnostics['diagnostics_dir']}")
    print(f"model_name={diagnostics['model_name']}")
    print(f"primary_score_column={diagnostics['primary_score_column']}")
    print(f"honesty_flags={','.join(diagnostics['honesty_flags']) or 'none'}")


if __name__ == "__main__":
    main()
