"""Operator helper for exporting existing M20 specialist OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.m20_specialist_prediction_export import (
    export_existing_m20_specialist_predictions,
)


def main() -> None:
    """Export sanitized existing NHITS/PatchTST OOF predictions."""
    parser = argparse.ArgumentParser(
        description="Export existing M20 specialist OOF predictions"
    )
    parser.add_argument("--base-run-dir", required=True)
    parser.add_argument("--previous-run-dir", required=True)
    parser.add_argument("--prediction-source", default="oof_20260427")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = export_existing_m20_specialist_predictions(
        base_run_dir=Path(args.base_run_dir),
        previous_run_dir=Path(args.previous_run_dir),
        prediction_source=args.prediction_source,
    )
    if args.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return
    print(f"output_dir={Path(result['output_files']['manifest_json']).parent}")
    print(f"exported_row_count={result['exported_row_count']}")
    print("runtime_status=NO_RUNTIME_EFFECT")


if __name__ == "__main__":
    main()
