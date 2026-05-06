"""Operator helper for research-only M20 trading-aware label artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.common.serialization import make_json_safe
from app.training.research_labels import (
    DEFAULT_FIXED_BARRIER_BPS,
    DEFAULT_LABEL_HORIZON,
    DEFAULT_META_SIGNAL_THRESHOLD,
    generate_completed_run_research_labels,
    generate_training_frame_research_labels,
)


def main() -> None:
    """Generate research-only M20 label artifacts for one completed run."""
    parser = argparse.ArgumentParser(
        description="Generate Stream Alpha M20 research-only trading-aware labels",
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
        "--source",
        choices=("oof", "training-frame", "auto"),
        default="auto",
        help="Research label source. Auto uses training_frame when OOF artifacts are absent.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_LABEL_HORIZON,
        help="Forward horizon for research labels. Defaults to 3 rows/candles.",
    )
    parser.add_argument(
        "--fixed-barrier-bps",
        type=float,
        default=DEFAULT_FIXED_BARRIER_BPS,
        help="Fixed fallback barrier in bps when volatility/price path is unavailable.",
    )
    parser.add_argument(
        "--meta-signal-threshold",
        type=float,
        default=DEFAULT_META_SIGNAL_THRESHOLD,
        help="Default incumbent signal threshold used for research meta-labels.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the saved research-label manifest as JSON.",
    )
    parser.add_argument(
        "--use-volatility",
        action="store_true",
        help="Also audit volatility sources and write vol-scaled research labels when possible.",
    )
    arguments = parser.parse_args()
    try:
        run_dir = Path(arguments.run_dir) if arguments.run_dir else None
        use_training_frame = (
            arguments.source == "training-frame"
            or (
                arguments.source == "auto"
                and run_dir is not None
                and (run_dir / "training_frame" / "m20_training_frame_features.csv").exists()
                and not (run_dir / "oof_predictions.csv").exists()
            )
        )
        if use_training_frame:
            result = generate_training_frame_research_labels(
                run_dir=run_dir or Path(""),
                horizon=arguments.horizon,
                fixed_barrier_bps=arguments.fixed_barrier_bps,
            )
        elif arguments.use_volatility:
            from app.training.volatility_audit import audit_completed_run_volatility_sources

            result = audit_completed_run_volatility_sources(
                run_dir=run_dir,
                model_name=arguments.model_name,
            )
        else:
            result = generate_completed_run_research_labels(
                run_dir=run_dir,
                model_name=arguments.model_name,
                horizon=arguments.horizon,
                fixed_barrier_bps=arguments.fixed_barrier_bps,
                meta_signal_threshold=arguments.meta_signal_threshold,
            )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if arguments.json:
        print(json.dumps(make_json_safe(result), sort_keys=True))
        return

    if result.get("source") == "training_frame":
        print(f"run_dir={result['run_dir']}")
        print(f"label_dir={result['label_dir']}")
        print(f"source={result['source']}")
        print(f"volatility_source={result.get('volatility_source', '')}")
        print(f"honesty_flags={','.join(result['honesty_flags']) or 'none'}")
        diagnostics = result["diagnostics"]
        print(
            "triple_positive_event_rate="
            f"{float(diagnostics['triple_barrier']['positive_event_rate']):.6f}"
        )
        print(
            "fee_after_cost_positive_event_rate="
            f"{float(diagnostics['fee_exceedance']['after_cost_positive_event_rate']):.6f}"
        )
        return

    if arguments.use_volatility:
        print(f"run_dir={result['run_dir']}")
        print(f"audit_dir={result['audit_dir']}")
        print(f"model_name={result['model_name']}")
        print(f"volatility_source={result['volatility_source']}")
        print(f"honesty_flags={','.join(result['honesty_flags']) or 'none'}")
        print(f"recommendation={result['recommendation']}")
        return

    diagnostics = result["diagnostics"]
    print(f"run_dir={result['run_dir']}")
    print(f"label_dir={result['label_dir']}")
    print(f"model_name={result['model_name']}")
    print(f"honesty_flags={','.join(result['honesty_flags']) or 'none'}")
    print(
        "triple_positive_event_rate="
        f"{float(diagnostics['triple_barrier']['positive_event_rate']):.6f}"
    )
    print(
        "fee_after_cost_positive_event_rate="
        f"{float(diagnostics['fee_exceedance']['after_cost_positive_event_rate']):.6f}"
    )


if __name__ == "__main__":
    main()
