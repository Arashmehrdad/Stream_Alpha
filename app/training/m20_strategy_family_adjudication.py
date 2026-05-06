"""Research-only M20 strategy-family diagnostic adjudication."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "strategy_family_adjudication"
FAMILY_DIRS = {
    "momentum_breakout": "momentum_breakout_research",
    "range_mean_reversion": "range_mean_reversion_research",
    "volatility_expansion": "volatility_expansion_research",
}
HONESTY_FLAGS = (
    "RESEARCH_ONLY_STRATEGY_FAMILY_ADJUDICATION",
    "EXISTING_ARTIFACTS_ONLY",
    "NOT_BACKTEST",
    "NOT_PNL",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_RUNTIME_EFFECT",
    "NO_REGISTRY_WRITE",
    "NO_PROMOTION_EFFECT",
    "NO_PROFIT_CLAIM",
)


def adjudicate_m20_strategy_families(*, base_run_dir: Path) -> dict[str, Any]:
    """Compare completed M20 strategy-family diagnostics from existing artifacts."""
    base_dir = Path(base_run_dir).resolve()
    vol_dir = base_dir / "research_labels" / "vol_scaled"
    output_dir = vol_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = _output_files(output_dir)

    setup_rows = _setup_comparison(vol_dir)
    overlap_rows = _rank_gate_overlap_summary(vol_dir)
    family_rows = _family_comparison(setup_rows)
    experiments = _recommended_next_experiments(family_rows)
    recommendation = _recommendation(family_rows)
    report = {
        "base_run_dir": str(base_dir),
        "recommendation": recommendation["recommendation"],
        "family_count": len(family_rows),
        "setup_count": len(setup_rows),
        "primary_family": recommendation["primary_family"],
        "secondary_family": recommendation["secondary_family"],
        "watchlist_family": recommendation["watchlist_family"],
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": recommendation["blockers"],
        "output_files": output_files,
    }
    manifest = {
        "base_run_dir": str(base_dir),
        "output_dir": str(output_dir),
        "family_dirs": FAMILY_DIRS,
        "source": "existing_strategy_family_diagnostics",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(
        _markdown(report, family_rows, setup_rows),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["family_comparison_csv"]), family_rows)
    write_csv_artifact(Path(output_files["setup_comparison_csv"]), setup_rows)
    write_csv_artifact(Path(output_files["rank_gate_overlap_summary_csv"]), overlap_rows)
    write_csv_artifact(Path(output_files["recommended_next_experiments_csv"]), experiments)
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(report)


def _setup_comparison(vol_dir: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for family, directory in FAMILY_DIRS.items():
        metrics = _read_csv(vol_dir / directory / "setup_metrics.csv")
        recommendation = _read_json(vol_dir / directory / "recommendation.json")
        for setup_name in sorted({row.get("setup_name", "") for row in metrics}):
            rows = [row for row in metrics if row.get("setup_name") == setup_name]
            lifts = [_float(row.get("lift_vs_base")) for row in rows]
            frequencies = [_float(row.get("setup_frequency")) for row in rows]
            positive_rates = [_float(row.get("setup_positive_rate")) for row in rows]
            net_values = [_float(row.get("net_proxy_mean")) for row in rows]
            min_lift = min(lifts) if lifts else 0.0
            output.append(
                {
                    "family_id": family,
                    "setup_name": setup_name,
                    "source_recommendation": recommendation.get("recommendation", ""),
                    "run_count_observed": len(rows),
                    "min_lift": min_lift,
                    "avg_lift": sum(lifts) / len(lifts) if lifts else 0.0,
                    "max_lift": max(lifts) if lifts else 0.0,
                    "min_setup_frequency": min(frequencies) if frequencies else 0.0,
                    "max_setup_frequency": max(frequencies) if frequencies else 0.0,
                    "min_positive_rate": min(positive_rates) if positive_rates else 0.0,
                    "max_positive_rate": max(positive_rates) if positive_rates else 0.0,
                    "net_proxy_mean_avg": sum(net_values) / len(net_values) if net_values else 0.0,
                    "setup_classification": _classify_setup(min_lift),
                }
            )
    return sorted(
        output,
        key=lambda row: (-_float(row["min_lift"]), str(row["family_id"]), str(row["setup_name"])),
    )


def _rank_gate_overlap_summary(vol_dir: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for family, directory in FAMILY_DIRS.items():
        rows = _read_csv(vol_dir / directory / "rank_gate_overlap.csv")
        for setup_name in sorted({row.get("setup_name", "") for row in rows}):
            group = [row for row in rows if row.get("setup_name") == setup_name]
            overlap_rates = [_float(row.get("rank_gate_overlap_rate")) for row in group]
            overlap_positive_rates = [_float(row.get("overlap_positive_rate")) for row in group]
            output.append(
                {
                    "family_id": family,
                    "setup_name": setup_name,
                    "run_count_observed": len(group),
                    "avg_rank_gate_overlap_rate": (
                        sum(overlap_rates) / len(overlap_rates) if overlap_rates else 0.0
                    ),
                    "max_rank_gate_overlap_rate": max(overlap_rates) if overlap_rates else 0.0,
                    "avg_overlap_positive_rate": (
                        sum(overlap_positive_rates) / len(overlap_positive_rates)
                        if overlap_positive_rates else 0.0
                    ),
                }
            )
    return sorted(output, key=lambda row: (str(row["family_id"]), str(row["setup_name"])))


def _family_comparison(setup_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for family in sorted({str(row["family_id"]) for row in setup_rows}):
        rows = [row for row in setup_rows if row["family_id"] == family]
        best = max(rows, key=lambda row: _float(row["min_lift"]), default={})
        keep_primary = sum(
            1 for row in rows if row["setup_classification"] == "KEEP_PRIMARY_RESEARCH_CANDIDATE"
        )
        keep_secondary = sum(
            1 for row in rows if row["setup_classification"] == "KEEP_SECONDARY_RESEARCH_CANDIDATE"
        )
        watchlist = sum(1 for row in rows if row["setup_classification"] == "WATCHLIST_ONLY")
        output.append(
            {
                "family_id": family,
                "setup_count": len(rows),
                "best_setup": best.get("setup_name", ""),
                "best_min_lift": best.get("min_lift", 0.0),
                "best_avg_lift": best.get("avg_lift", 0.0),
                "primary_candidate_count": keep_primary,
                "secondary_candidate_count": keep_secondary,
                "watchlist_count": watchlist,
                "family_classification": _threshold_family_classification(
                    _float(best.get("min_lift")),
                    keep_primary,
                    keep_secondary,
                    watchlist,
                ),
            }
        )
    ranked = sorted(output, key=lambda row: -_float(row["best_min_lift"]))
    for index, row in enumerate(ranked):
        row["family_classification"] = _ranked_family_classification(index, row)
    return ranked


def _recommended_next_experiments(
    family_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    experiments = []
    for index, family in enumerate(family_rows, start=1):
        classification = str(family.get("family_classification", ""))
        action = {
            "PRIMARY_FAMILY": "Test this family first in the next research-only diagnostic.",
            "SECONDARY_FAMILY": "Retain as a second research branch after the primary family.",
            "WATCHLIST_FAMILY": "Retain for review, but do not prioritize the next batch.",
        }.get(classification, "Keep as weak evidence only unless new features or labels change.")
        experiments.append(
            {
                "rank": index,
                "family_id": family["family_id"],
                "best_setup": family["best_setup"],
                "best_min_lift": family["best_min_lift"],
                "family_classification": classification,
                "next_action": action,
            }
        )
    return experiments


def _recommendation(family_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    primary = next(
        (row for row in family_rows if row["family_classification"] == "PRIMARY_FAMILY"),
        family_rows[0] if family_rows else {},
    )
    primary_family = str(primary.get("family_id", ""))
    if primary_family == "volatility_expansion":
        recommendation = "TEST_VOLATILITY_EXPANSION_NEXT"
    elif primary_family == "momentum_breakout":
        recommendation = "TEST_MOMENTUM_BREAKOUT_NEXT"
    elif primary_family:
        recommendation = "COMPARE_TOP_FAMILIES_IN_COMBO_NEXT"
    else:
        recommendation = "PAUSE_STRATEGY_FAMILY_PATH"
    return {
        "recommendation": recommendation,
        "primary_family": primary_family,
        "secondary_family": _family_with_class(family_rows, "SECONDARY_FAMILY"),
        "watchlist_family": _family_with_class(family_rows, "WATCHLIST_FAMILY"),
        "honesty_flags": list(HONESTY_FLAGS),
        "blockers": [
            "SETUP_LIFT_IS_NOT_PNL",
            "NOT_BACKTEST",
            "NO_PROFIT_CLAIM",
            "NOT_RUNTIME_READY",
            "NOT_PROMOTABLE",
        ],
    }


def _family_with_class(rows: Sequence[Mapping[str, Any]], classification: str) -> str:
    for row in rows:
        if row.get("family_classification") == classification:
            return str(row.get("family_id", ""))
    return ""


def _classify_setup(min_lift: float) -> str:
    if min_lift >= 1.50:
        return "KEEP_PRIMARY_RESEARCH_CANDIDATE"
    if min_lift >= 1.30:
        return "KEEP_SECONDARY_RESEARCH_CANDIDATE"
    if min_lift >= 1.05:
        return "WATCHLIST_ONLY"
    return "WEAK_OR_UNSTABLE"


def _threshold_family_classification(
    best_min_lift: float,
    primary_count: int,
    secondary_count: int,
    watchlist_count: int,
) -> str:
    if best_min_lift >= 1.50 and primary_count >= 1:
        return "PRIMARY_FAMILY"
    if best_min_lift >= 1.30 and secondary_count >= 1:
        return "SECONDARY_FAMILY"
    if best_min_lift >= 1.05 and watchlist_count >= 1:
        return "WATCHLIST_FAMILY"
    return "WEAK_FAMILY"


def _ranked_family_classification(index: int, row: Mapping[str, Any]) -> str:
    best_min_lift = _float(row.get("best_min_lift"))
    if index == 0 and best_min_lift >= 1.50:
        return "PRIMARY_FAMILY"
    if best_min_lift >= 1.30:
        return "SECONDARY_FAMILY"
    if best_min_lift >= 1.05:
        return "WATCHLIST_FAMILY"
    return "WEAK_FAMILY"


def _markdown(
    report: Mapping[str, Any],
    family_rows: Sequence[Mapping[str, Any]],
    setup_rows: Sequence[Mapping[str, Any]],
) -> str:
    best = setup_rows[0] if setup_rows else {}
    family_lines = [
        (
            f"- `{row['family_id']}`: `{row['family_classification']}`, "
            f"best `{row['best_setup']}` min lift `{_float(row['best_min_lift']):.6f}`"
        )
        for row in family_rows
    ]
    return "\n".join(
        [
            "# M20 Strategy-Family Adjudication",
            "",
            f"- Recommendation: `{report['recommendation']}`",
            f"- Primary family: `{report['primary_family']}`",
            f"- Best setup: `{best.get('family_id', '')}:{best.get('setup_name', '')}`",
            f"- Honesty flags: `{', '.join(HONESTY_FLAGS)}`",
            "",
            "## Family Summary",
            *family_lines,
            "",
            "This packet uses existing research artifacts only. Setup lift is not PnL,",
            "not a backtest, not a runtime selector, and not profitability evidence.",
            "",
        ]
    )


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "report.json"),
        "report_md": str(output_dir / "report.md"),
        "family_comparison_csv": str(output_dir / "family_comparison.csv"),
        "setup_comparison_csv": str(output_dir / "setup_comparison.csv"),
        "rank_gate_overlap_summary_csv": str(output_dir / "rank_gate_overlap_summary.csv"),
        "recommended_next_experiments_csv": str(output_dir / "recommended_next_experiments.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _float(value: Any) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return 0.0
    return converted if math.isfinite(converted) else 0.0


def read_strategy_family_adjudication_csv(path: Path) -> list[dict[str, str]]:
    """Read strategy-family adjudication CSV for tests."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
