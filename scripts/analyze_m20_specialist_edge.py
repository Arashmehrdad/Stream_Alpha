"""Operator helper for generic M20 specialist edge evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_specialist_edge_evaluator import (
    DEFAULT_LABEL_FILE,
    DEFAULT_OUTPUT_NAME,
    analyze_m20_specialist_edge,
)


def main() -> None:
    """Run generic specialist edge diagnostics from existing artifacts."""
    parser = argparse.ArgumentParser(
        description="Analyze generic M20 specialist edge from saved artifacts"
    )
    parser.add_argument("--prediction-run-dir", required=True)
    parser.add_argument("--label-source-run-dir", required=True)
    parser.add_argument("--prediction-source", required=True)
    parser.add_argument(
        "--models",
        help="Optional comma-separated model list. Defaults to discovered files.",
    )
    parser.add_argument("--label-file", default=DEFAULT_LABEL_FILE)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    models = (
        [model.strip() for model in args.models.split(",")]
        if args.models is not None
        else None
    )
    try:
        result = analyze_m20_specialist_edge(
            prediction_run_dir=Path(args.prediction_run_dir),
            label_source_run_dir=Path(args.label_source_run_dir),
            prediction_source=args.prediction_source,
            models=models,
            label_file=args.label_file,
            output_name=args.output_name,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"best_candidate={result['best_candidate']}")
    print(f"joined_rows={result['joined_rows']}")


if __name__ == "__main__":
    main()
