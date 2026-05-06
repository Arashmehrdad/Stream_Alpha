"""Research-only M20 strategy selector design artifact generation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DESIGN_DIR_NAME = "strategy_selector_design"
SELECTOR_ID = "fee_exceedance_gate_v0_research"
PRIMARY_SLICES = {
    "momentum=flat",
    "range=low",
    "symbol=BTC/USD",
    "macd=positive",
    "volume=low",
}
DISABLE_GAP_SLICES = {"month=2026-04", "quarter=2026Q2"}
HONESTY_FLAGS = (
    "RESEARCH_ONLY_STRATEGY_SELECTOR_DESIGN",
    "NOT_RUNTIME_SELECTOR",
    "NOT_PROMOTABLE",
    "NO_REGISTRY_WRITE",
    "NO_RUNTIME_EFFECT",
    "NO_PROMOTION_EFFECT",
    "NO_PROFITABILITY_CLAIM",
    "STRATEGY_ENSEMBLE_DESIGN_ONLY",
    "CONFIRMED_FEE_EXCEEDANCE_GATE_INPUT",
    "SINGLE_MODEL_GATE_ONLY",
    "MULTI_STRATEGY_ARCHITECTURE_NOT_IMPLEMENTED",
    "DISABLE_GAPS_UNTESTED",
    "FUTURE_CONFIRMATION_REQUIRED",
    "AUTOGLUON_MEMBER_PREDICTIONS_STILL_MISSING",
    "PATCHTST_NHITS_CONDITIONAL_USEFULNESS_UNKNOWN",
)


def design_m20_strategy_selector(
    *,
    original_run_dir: Path,
    confirmation_run_dir: Path,
) -> dict[str, Any]:
    """Create a research-only selector design from confirmed fee-gate evidence."""
    # pylint: disable=too-many-locals
    original = Path(original_run_dir).resolve()
    confirmation = Path(confirmation_run_dir).resolve()
    output_dir = original / "research_labels" / "vol_scaled" / DESIGN_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows = _load_comparison_rows(original)
    if not comparison_rows:
        comparison_rows = _fallback_comparison_rows(original, confirmation)
    ledger_rows = _read_csv_rows(
        original
        / "research_labels"
        / "vol_scaled"
        / "model_member_audit"
        / "strategy_ensemble_candidate_ledger.csv"
    )
    evidence_rows = [_evidence_row(row) for row in comparison_rows]
    unknown_rows = [
        row for row in evidence_rows
        if row["proposed_selector_action"] in {
            "DISABLE_GAP_UNTESTED",
            "UNKNOWN_RESEARCH_CANDIDATE",
        }
    ]
    confirmed = [
        _condition(row) for row in evidence_rows
        if row["proposed_selector_action"] == "ENABLE_RESEARCH_CANDIDATE"
    ]
    watchlist = [
        _condition(row) for row in evidence_rows
        if row["proposed_selector_action"] == "WATCHLIST_RESEARCH_CANDIDATE"
    ]
    output_files = _output_files(output_dir)
    selector_spec = {
        "selector_id": SELECTOR_ID,
        "selector_version": "v0_research",
        "selector_type": "OPPORTUNITY_GATE",
        "target": "fee_exceedance",
        "source_model": "logistic_regression_tiny",
        "source_runs": [str(original), str(confirmation)],
        "confirmation_run": str(confirmation),
        "evidence_status": "CONFIRMED_RESEARCH_ONLY",
        "promotable": False,
        "runtime_enabled": False,
        "registry_write": False,
        "rule_mode": "WEIGHTED_CONFIRMED_SLICES",
        "supported_rule_modes": [
            "ANY_CONFIRMED_SLICE",
            "WEIGHTED_CONFIRMED_SLICES",
            "STRICT_MULTI_CONDITION",
        ],
        "confirmed_conditions": confirmed,
        "watchlist_conditions": watchlist,
        "unknown_conditions": _unknown_candidates(ledger_rows),
        "disable_gap_conditions": sorted(DISABLE_GAP_SLICES),
        "required_future_confirmations": [
            "Confirm month=2026-04 and quarter=2026Q2 in a comparable later window.",
            (
                "Run research-only selector simulation on original plus "
                "confirmation windows."
            ),
        ],
        "intended_future_role": (
            "Allow strategy modules to search for trades only when fee-exceedance "
            "probability and confirmed market conditions suggest movement may exceed costs."
        ),
        "forbidden_current_use": [
            "runtime routing",
            "registry promotion",
            "paper/live execution",
            "profitability claim",
            "final long/short decision",
        ],
    }
    rule_rows = _rule_rows(selector_spec)
    report = {
        "selector_id": SELECTOR_ID,
        "design_dir": str(output_dir),
        "original_run_dir": str(original),
        "confirmation_run_dir": str(confirmation),
        "selector_type": "OPPORTUNITY_GATE",
        "rule_mode": "WEIGHTED_CONFIRMED_SLICES",
        "confirmed_condition_count": len(confirmed),
        "watchlist_condition_count": len(watchlist),
        "disable_gap_count": len(DISABLE_GAP_SLICES),
        "primary_slices_confirmed": sorted(PRIMARY_SLICES.intersection(set(confirmed))),
        "honesty_flags": list(HONESTY_FLAGS),
        "recommendation": (
            "A. run research-only selector simulation on original + confirmation runs, "
            "while separately planning disable/gap confirmation later."
        ),
        "output_files": output_files,
    }
    manifest = {
        "selector_id": SELECTOR_ID,
        "design_dir": str(output_dir),
        "runtime_effect": "none_research_only",
        "registry_write": False,
        "promotion_effect": False,
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["strategy_selector_manifest_json"]), manifest)
    write_json_artifact(Path(output_files["strategy_selector_design_report_json"]), report)
    Path(output_files["strategy_selector_design_report_md"]).write_text(
        _report_markdown(report, selector_spec),
        encoding="utf-8",
    )
    write_json_artifact(Path(output_files["strategy_selector_candidate_spec_json"]), selector_spec)
    write_csv_artifact(Path(output_files["strategy_selector_candidate_rules_csv"]), rule_rows)
    write_csv_artifact(
        Path(output_files["strategy_selector_condition_weights_csv"]),
        _weight_rows(evidence_rows),
    )
    write_csv_artifact(Path(output_files["strategy_selector_evidence_table_csv"]), evidence_rows)
    write_csv_artifact(
        Path(output_files["strategy_selector_unknowns_and_gaps_csv"]),
        unknown_rows or [_empty_unknown_row()],
    )
    Path(output_files["strategy_selector_manual_next_steps_md"]).write_text(
        _manual_next_steps(),
        encoding="utf-8",
    )
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "selector_spec": selector_spec,
            "evidence_rows": evidence_rows,
        }
    )


def _evidence_row(row: Mapping[str, str]) -> dict[str, Any]:
    slice_id = row.get("slice_id") or f"{row.get('slice_family')}={row.get('slice_value')}"
    status = row.get("confirmation_status", "")
    action = _action(slice_id, status)
    weight = _weight(row, action)
    return {
        "slice_family": row.get("slice_family", ""),
        "slice_value": row.get("slice_value", ""),
        "slice_id": slice_id,
        "original_status": row.get("original_classification", ""),
        "confirmation_status": status,
        "original_rows": _to_int(row.get("original_row_count")),
        "confirmation_rows": _to_int(row.get("confirmation_row_count")),
        "original_positive_rate": "",
        "confirmation_positive_rate": "",
        "original_top5_lift": _to_float(row.get("original_top_5_lift")),
        "confirmation_top5_lift": _to_float(row.get("confirmation_top_5_lift")),
        "original_pr_auc": "",
        "confirmation_pr_auc": "",
        "original_pr_auc_lift_over_base": _to_float(
            row.get("original_pr_auc_lift_over_base")
        ),
        "confirmation_pr_auc_lift_over_base": _to_float(
            row.get("confirmation_pr_auc_lift_over_base")
        ),
        "evidence_weight": weight,
        "proposed_selector_action": action,
        "notes": _notes(slice_id, status, action),
    }


def _action(slice_id: str, status: str) -> str:
    if slice_id in DISABLE_GAP_SLICES and status == "MISSING_IN_CONFIRMATION":
        return "DISABLE_GAP_UNTESTED"
    if status in {"STRONGLY_CONFIRMED", "CONFIRMED"}:
        return "ENABLE_RESEARCH_CANDIDATE"
    if status == "MISSING_IN_CONFIRMATION":
        return "UNKNOWN_RESEARCH_CANDIDATE"
    if status == "INCONCLUSIVE":
        return "UNKNOWN_RESEARCH_CANDIDATE"
    if status == "NOT_CONFIRMED":
        return "REJECT_RESEARCH_CANDIDATE"
    return "WATCHLIST_RESEARCH_CANDIDATE"


def _weight(row: Mapping[str, str], action: str) -> float:
    if action in {"DISABLE_GAP_UNTESTED", "UNKNOWN_RESEARCH_CANDIDATE"}:
        return 0.0
    if action == "REJECT_RESEARCH_CANDIDATE":
        return -1.0
    status_bonus = 2.0 if row.get("confirmation_status") == "STRONGLY_CONFIRMED" else 1.0
    lift = max(_to_float(row.get("confirmation_top_5_lift")) - 1.0, 0.0)
    pr_lift = max(_to_float(row.get("confirmation_pr_auc_lift_over_base")), 0.0)
    rows = min(_to_int(row.get("confirmation_row_count")) / 10000.0, 2.0)
    positives = min(_to_int(row.get("confirmation_positive_count")) / 1000.0, 2.0)
    return round(status_bonus + lift + pr_lift * 5.0 + rows * 0.2 + positives * 0.2, 6)


def _rule_rows(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "selector_id": spec["selector_id"],
            "rule_mode": spec["rule_mode"],
            "condition": condition,
            "selector_output": "ALLOW_STRATEGY_SEARCH",
            "current_use": "research_design_only",
        }
        for condition in spec["confirmed_conditions"]
    ] + [
        {
            "selector_id": spec["selector_id"],
            "rule_mode": spec["rule_mode"],
            "condition": condition,
            "selector_output": "HOLD_OR_SKIP",
            "current_use": "watchlist_research_only",
        }
        for condition in spec["watchlist_conditions"]
    ]


def _weight_rows(evidence_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "slice_family": row["slice_family"],
            "slice_value": row["slice_value"],
            "confirmation_status": row["confirmation_status"],
            "evidence_weight": row["evidence_weight"],
            "proposed_selector_action": row["proposed_selector_action"],
        }
        for row in evidence_rows
    ]


def _unknown_candidates(ledger_rows: Sequence[Mapping[str, str]]) -> list[str]:
    names = sorted(
        {
            row.get("model_name", "")
            for row in ledger_rows
            if row.get("model_name") in {"neuralforecast_patchtst", "neuralforecast_nhits"}
        }
    )
    return [f"{name}:CONDITIONAL_USEFULNESS_UNKNOWN" for name in names]


def _load_comparison_rows(original: Path) -> list[dict[str, str]]:
    return _read_csv_rows(
        original
        / "research_labels"
        / "vol_scaled"
        / "confirmation_plan"
        / "confirmation_comparison"
        / "confirmation_slice_comparison.csv"
    )


def _fallback_comparison_rows(original: Path, confirmation: Path) -> list[dict[str, str]]:
    del confirmation
    return _read_csv_rows(
        original
        / "research_labels"
        / "vol_scaled"
        / "conditional_usefulness_full_test"
        / "conditional_usefulness_by_slice.csv"
    )


def _output_files(output_dir: Path) -> dict[str, str]:
    names = {
        "strategy_selector_manifest_json": "strategy_selector_manifest.json",
        "strategy_selector_design_report_json": "strategy_selector_design_report.json",
        "strategy_selector_design_report_md": "strategy_selector_design_report.md",
        "strategy_selector_candidate_spec_json": "strategy_selector_candidate_spec.json",
        "strategy_selector_candidate_rules_csv": "strategy_selector_candidate_rules.csv",
        "strategy_selector_condition_weights_csv": "strategy_selector_condition_weights.csv",
        "strategy_selector_evidence_table_csv": "strategy_selector_evidence_table.csv",
        "strategy_selector_unknowns_and_gaps_csv": "strategy_selector_unknowns_and_gaps.csv",
        "strategy_selector_manual_next_steps_md": "strategy_selector_manual_next_steps.md",
    }
    return {key: str(output_dir / name) for key, name in names.items()}


def _report_markdown(report: Mapping[str, Any], spec: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# M20 Strategy Selector Design",
            "",
            f"- Selector: `{report['selector_id']}`",
            f"- Type: `{report['selector_type']}`",
            f"- Rule mode: `{report['rule_mode']}`",
            f"- Confirmed conditions: `{report['confirmed_condition_count']}`",
            f"- Watchlist conditions: `{report['watchlist_condition_count']}`",
            f"- Disable gaps: `{report['disable_gap_count']}`",
            f"- Honesty flags: `{', '.join(report['honesty_flags'])}`",
            "",
            "## Confirmed Primary Inputs",
            "",
            *[f"- `{condition}`" for condition in sorted(PRIMARY_SLICES)],
            "",
            "## Future Architecture",
            "",
            "- Regime detector identifies trend/range/spark/chop.",
            "- Fee-exceedance gate checks whether movement may exceed fees/slippage.",
            "- Strategy modules decide breakout, momentum, mean reversion, or range bounce.",
            "- Risk engine decides final approval.",
            "- If no confirmed gate or strategy condition is active, HOLD.",
            "",
            "This architecture is not implemented in runtime.",
            "",
            "## Recommendation",
            "",
            f"- {report['recommendation']}",
            "",
            "## Forbidden Current Use",
            "",
            *[f"- `{item}`" for item in spec["forbidden_current_use"]],
            "",
        ]
    )


def _manual_next_steps() -> str:
    return "\n".join(
        [
            "# M20 Strategy Selector Manual Next Steps",
            "",
            "1. Confirm disable/gap slices in a comparable later window.",
            "2. Run research-only selector simulation on original plus confirmation runs.",
            "3. Add AutoGluon member prediction export later.",
            "4. Run conditional usefulness on PatchTST/NHITS if row-level predictions exist.",
            (
                "5. Design strategy families: momentum/breakout, "
                "range/mean-reversion, abstention/HOLD."
            ),
            "",
            "Do not run long commands from this design artifact automatically.",
            "",
        ]
    )


def _notes(slice_id: str, status: str, action: str) -> str:
    if slice_id in DISABLE_GAP_SLICES:
        return "Original disable slice outside confirmation window; gap remains untested."
    if action == "ENABLE_RESEARCH_CANDIDATE":
        return "Confirmed in original and confirmation evidence; research selector input only."
    if status == "MISSING_IN_CONFIRMATION":
        return "Missing from confirmation window; no runtime conclusion."
    return "Research-only condition; no runtime action."


def _empty_unknown_row() -> dict[str, str]:
    return {
        "slice_family": "",
        "slice_value": "",
        "slice_id": "",
        "proposed_selector_action": "",
        "notes": "No unknown or gap rows emitted.",
    }


def _condition(row: Mapping[str, Any]) -> str:
    return f"{row['slice_family']}={row['slice_value']}"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


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
