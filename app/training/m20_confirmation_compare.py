"""Compare M20 confirmation-window conditional slice evidence."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_confirmation_planner import PLAN_DIR_NAME
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


CONDITIONAL_DIR_NAME = "conditional_usefulness_full_test"
COMPARISON_DIR_NAME = "confirmation_comparison"
MIN_ROWS = 1000
MIN_POSITIVES = 50


def compare_m20_confirmation_slices(
    *,
    original_run_dir: Path,
    confirmation_run_dir: Path,
) -> dict[str, Any]:
    """Compare original full-test slice evidence against a future confirmation run."""
    original = Path(original_run_dir).resolve()
    confirmation = Path(confirmation_run_dir).resolve()
    if not confirmation.exists():
        return make_json_safe(
            {
                "status": "CONFIRMATION_RUN_NOT_AVAILABLE",
                "original_run_dir": str(original),
                "confirmation_run_dir": str(confirmation),
                "message": "Confirmation run directory is missing; comparison pending.",
                "recommendation": "INCONCLUSIVE_LOW_SAMPLE",
            }
        )
    original_rows = _read_slice_rows(original)
    confirmation_rows = _read_slice_rows(confirmation)
    if not confirmation_rows:
        return make_json_safe(
            {
                "status": "CONFIRMATION_RUN_NOT_AVAILABLE",
                "original_run_dir": str(original),
                "confirmation_run_dir": str(confirmation),
                "message": "Confirmation conditional usefulness slices are missing.",
                "recommendation": "INCONCLUSIVE_LOW_SAMPLE",
            }
        )
    targets = _read_targets(original)
    comparisons = _comparison_rows(original_rows, confirmation_rows, targets)
    summary = _summary(comparisons)
    output_dir = original / "research_labels" / "vol_scaled" / PLAN_DIR_NAME / COMPARISON_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "confirmation_slice_comparison_json": str(
            output_dir / "confirmation_slice_comparison.json"
        ),
        "confirmation_slice_comparison_csv": str(output_dir / "confirmation_slice_comparison.csv"),
        "confirmation_slice_comparison_md": str(output_dir / "confirmation_slice_comparison.md"),
    }
    report = {
        "status": "CONFIRMATION_COMPARISON_COMPLETE",
        "original_run_dir": str(original),
        "confirmation_run_dir": str(confirmation),
        "summary": summary,
        "recommendation": _recommend(summary),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["confirmation_slice_comparison_json"]), report)
    write_csv_artifact(Path(output_files["confirmation_slice_comparison_csv"]), comparisons)
    Path(output_files["confirmation_slice_comparison_md"]).write_text(
        _markdown(report, comparisons),
        encoding="utf-8",
    )
    return make_json_safe({**report, "comparison_rows": comparisons})


def classify_confirmation(row: Mapping[str, Any]) -> str:
    """Classify one confirmation slice using the frozen confirmation rules."""
    row_count = _to_int(row.get("row_count"))
    positive_count = _to_int(row.get("positive_count"))
    top_5_lift = _to_float(row.get("top_5_lift"))
    top_5_precision = _to_float(row.get("top_5_precision"))
    positive_rate = _to_float(row.get("positive_rate"))
    average_precision = _to_float(row.get("average_precision"))
    if row_count < MIN_ROWS or positive_count < MIN_POSITIVES:
        return "INCONCLUSIVE"
    if top_5_precision <= positive_rate:
        return "NOT_CONFIRMED"
    pr_lift = average_precision - positive_rate
    if top_5_lift >= 1.5 and pr_lift >= 0.03:
        return "STRONGLY_CONFIRMED"
    if top_5_lift >= 1.2 and pr_lift >= 0.015:
        return "CONFIRMED"
    if top_5_lift <= 1.0 or average_precision <= positive_rate:
        return "NOT_CONFIRMED"
    return "INCONCLUSIVE"


def _comparison_rows(
    original_rows: Sequence[Mapping[str, str]],
    confirmation_rows: Sequence[Mapping[str, str]],
    targets: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    original_by_id = {_slice_id(row): row for row in original_rows}
    confirmation_by_id = {_slice_id(row): row for row in confirmation_rows}
    target_ids = [
        _slice_id(row) for row in targets
    ] or [
        _slice_id(row)
        for row in original_rows
        if row.get("classification") in {"ENABLE_CANDIDATE", "WATCHLIST_CANDIDATE"}
    ]
    rows = []
    for slice_id in sorted(dict.fromkeys(target_ids)):
        original = original_by_id.get(slice_id, {})
        confirmation = confirmation_by_id.get(slice_id, {})
        if not confirmation:
            status = "MISSING_IN_CONFIRMATION"
        else:
            status = classify_confirmation(confirmation)
        rows.append(
            {
                "slice_family": original.get("slice_family") or _family(slice_id),
                "slice_value": original.get("slice_value") or _value(slice_id),
                "slice_id": slice_id,
                "original_classification": original.get("classification", ""),
                "confirmation_status": status,
                "original_row_count": _to_int(original.get("row_count")),
                "confirmation_row_count": _to_int(confirmation.get("row_count")),
                "original_positive_count": _to_int(original.get("positive_count")),
                "confirmation_positive_count": _to_int(confirmation.get("positive_count")),
                "original_top_5_lift": _to_float(original.get("top_5_lift")),
                "confirmation_top_5_lift": _to_float(confirmation.get("top_5_lift")),
                "top_5_lift_delta": (
                    _to_float(confirmation.get("top_5_lift"))
                    - _to_float(original.get("top_5_lift"))
                ),
                "original_pr_auc_lift_over_base": (
                    _to_float(original.get("average_precision"))
                    - _to_float(original.get("positive_rate"))
                ),
                "confirmation_pr_auc_lift_over_base": (
                    _to_float(confirmation.get("average_precision"))
                    - _to_float(confirmation.get("positive_rate"))
                ),
            }
        )
    return rows


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        "slice_count": len(rows),
        "strongly_confirmed_count": _count(rows, "STRONGLY_CONFIRMED"),
        "confirmed_count": _count(rows, "CONFIRMED"),
        "not_confirmed_count": _count(rows, "NOT_CONFIRMED"),
        "inconclusive_count": _count(rows, "INCONCLUSIVE"),
        "missing_in_confirmation_count": _count(rows, "MISSING_IN_CONFIRMATION"),
    }


def _recommend(summary: Mapping[str, int]) -> str:
    confirmed = summary.get("strongly_confirmed_count", 0) + summary.get("confirmed_count", 0)
    if confirmed >= 3:
        return "CONFIRM_FEE_GATE_FOR_RESEARCH_POLICY"
    not_confirmed = summary.get("not_confirmed_count", 0)
    if not_confirmed >= confirmed and not_confirmed > 0:
        return "REJECT_CONDITIONAL_SLICE"
    if summary.get("missing_in_confirmation_count", 0):
        return "INCONCLUSIVE_LOW_SAMPLE"
    return "NEED_MORE_WINDOWS"


def _markdown(report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        [
            "# M20 Confirmation Slice Comparison",
            "",
            f"- Status: `{report['status']}`",
            f"- Recommendation: `{report['recommendation']}`",
            f"- Summary: `{report['summary']}`",
            "",
            "## Slice Results",
            "",
            *[
                (
                    f"- `{row['slice_id']}`: `{row['confirmation_status']}` "
                    f"(top-5 lift {row['original_top_5_lift']} -> "
                    f"{row['confirmation_top_5_lift']})"
                )
                for row in rows[:30]
            ],
            "",
            "Research-only comparison. No runtime or promotion decision is made.",
            "",
        ]
    )


def _read_slice_rows(run_dir: Path) -> list[dict[str, str]]:
    return _read_csv_rows(
        run_dir
        / "research_labels"
        / "vol_scaled"
        / CONDITIONAL_DIR_NAME
        / "conditional_usefulness_by_slice.csv"
    )


def _read_targets(run_dir: Path) -> list[dict[str, str]]:
    return _read_csv_rows(
        run_dir
        / "research_labels"
        / "vol_scaled"
        / PLAN_DIR_NAME
        / "confirmation_slice_targets.csv"
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def _slice_id(row: Mapping[str, Any]) -> str:
    return f"{row.get('slice_family', '')}={row.get('slice_value', '')}"


def _family(slice_id: str) -> str:
    return slice_id.split("=", 1)[0] if "=" in slice_id else slice_id


def _value(slice_id: str) -> str:
    return slice_id.split("=", 1)[1] if "=" in slice_id else ""


def _count(rows: Sequence[Mapping[str, Any]], status: str) -> int:
    return len([row for row in rows if row.get("confirmation_status") == status])


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
