"""Research-only M20 specialist confirmation export plan."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "specialist_confirmation_plan"
PRIMARY_MODEL = "neuralforecast_patchtst"
SECONDARY_MODEL = "neuralforecast_nhits"
HONESTY_FLAGS = (
    "RESEARCH_ONLY_SPECIALIST_CONFIRMATION_PLAN",
    "EXISTING_ARTIFACTS_ONLY",
    "MANUAL_LONG_RUN_ONLY",
    "NO_EXPORT_EXECUTED",
    "NO_SCORE_ONLY_RERUN_EXECUTED",
    "NO_MODEL_RETRAIN",
    "NO_RUNTIME_EFFECT",
    "NO_REGISTRY_WRITE",
    "NO_PROMOTION_EFFECT",
    "NOT_BACKTEST",
    "NO_PROFIT_CLAIM",
    "NOT_PROMOTABLE",
)
REQUIRED_SCHEMA = (
    "symbol",
    "interval_begin",
    "fold_index",
    "model_name",
    "candidate_id",
    "prediction_source",
    "confirmation_window_start",
    "confirmation_window_end",
    "confirmation_tag",
    "row_id",
    "as_of_time",
    "y_true",
    "y_pred",
    "prob_up",
    "confidence",
    "regime_label",
)


def write_m20_specialist_confirmation_plan(
    *,
    base_run_dir: Path,
    previous_run_dir: Path,
    fitted_models_dir: Path,
) -> dict[str, Any]:
    """Write a manual-only confirmation plan for M20 specialist predictions."""
    # pylint: disable=too-many-locals
    base_dir = Path(base_run_dir).resolve()
    previous_dir = Path(previous_run_dir).resolve()
    fitted_dir = Path(fitted_models_dir).resolve()
    vol_dir = base_dir / "research_labels" / "vol_scaled"
    output_dir = vol_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    conditional_dir = vol_dir / "specialist_conditional_usefulness"
    comparison = _read_csv(conditional_dir / "comparison.csv")
    by_slice = _read_csv(conditional_dir / "by_slice.csv")
    targets = _target_slices(by_slice)
    blockers = _blockers(fitted_dir=fitted_dir)
    recommendation = _recommendation(comparison, blockers)
    output_files = _output_files(output_dir)
    plan = {
        "recommendation": recommendation,
        "primary_candidate": PRIMARY_MODEL,
        "secondary_candidate": SECONDARY_MODEL,
        "base_run_dir": str(base_dir),
        "previous_run_dir": str(previous_dir),
        "fitted_models_dir": str(fitted_dir),
        "target_slice_count": len(targets),
        "required_schema": list(REQUIRED_SCHEMA),
        "confirmation_goal": (
            "Confirm whether PatchTST specialist top-k and slice usefulness survives "
            "on another comparable window before any specialist strategy claim."
        ),
        "manual_only": True,
        "codex_must_not_execute_long_exports": True,
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "output_files": output_files,
    }
    manifest = {
        "output_dir": str(output_dir),
        "source_conditional_dir": str(conditional_dir),
        "recommendation": recommendation,
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["plan_json"]), plan)
    write_json_artifact(
        Path(output_files["required_export_schema_json"]),
        {
            "required_columns": list(REQUIRED_SCHEMA),
            "forbidden_columns": [
                "future_return",
                "forward_return",
                "long_only_net_value_proxy",
                "long_only_gross_value_proxy",
                "target_derived_features",
                "barrier_metadata",
            ],
            "notes": (
                "Use only safe keys, metadata, target columns for evaluation, "
                "and prediction outputs."
            ),
        },
    )
    write_csv_artifact(Path(output_files["target_slices_csv"]), targets)
    write_csv_artifact(Path(output_files["blockers_csv"]), blockers)
    Path(output_files["manual_commands_md"]).write_text(
        _manual_commands(base_dir, fitted_dir, previous_dir),
        encoding="utf-8",
    )
    Path(output_files["post_export_analysis_commands_md"]).write_text(
        _post_export_commands(base_dir),
        encoding="utf-8",
    )
    Path(output_files["plan_md"]).write_text(
        _markdown(plan, comparison, targets, blockers),
        encoding="utf-8",
    )
    return make_json_safe(plan)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _target_slices(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows
        if row.get("classification") == "KEEP_CONDITIONAL_RESEARCH_CANDIDATE"
    ]
    candidates.sort(
        key=lambda row: (
            row.get("model_name") != PRIMARY_MODEL,
            -_to_float(row.get("top5_lift", "0")),
            row.get("slice_family", ""),
            row.get("slice_value", ""),
        )
    )
    return [
        {
            "model_name": row.get("model_name", ""),
            "slice_family": row.get("slice_family", ""),
            "slice_value": row.get("slice_value", ""),
            "row_count": row.get("row_count", ""),
            "positive_count": row.get("positive_count", ""),
            "positive_rate": row.get("positive_rate", ""),
            "top5_lift": row.get("top5_lift", ""),
            "classification": row.get("classification", ""),
            "confirmation_priority": (
                "PRIMARY" if row.get("model_name") == PRIMARY_MODEL else "SECONDARY"
            ),
        }
        for row in candidates
    ]


def _blockers(*, fitted_dir: Path) -> list[dict[str, str]]:
    blockers = [
        {
            "blocker": "LONG_RUNS_MANUAL_ONLY",
            "severity": "required_control",
            "details": "Codex must not launch score-only or specialist export jobs.",
        },
        {
            "blocker": "PER_SPECIALIST_EXPORT_HOOK_NOT_CONFIRMED",
            "severity": "blocking_for_confirmation_export",
            "details": "A clean per-specialist confirmation export hook is still needed.",
        },
        {
            "blocker": "PATCHTST_CONFIRMATION_RUN_NOT_AVAILABLE",
            "severity": "research_blocker",
            "details": "No separate confirmation-window PatchTST specialist prediction run exists.",
        },
        {
            "blocker": "AUTOGLUON_MEMBER_PREDICTIONS_MISSING",
            "severity": "tracked_gap",
            "details": "AutoGluon member predictions remain unavailable for specialist review.",
        },
    ]
    if not fitted_dir.exists():
        blockers.append(
            {
                "blocker": "FITTED_MODELS_DIR_MISSING",
                "severity": "blocking_for_manual_export",
                "details": str(fitted_dir),
            }
        )
    return blockers


def _recommendation(
    comparison: Sequence[Mapping[str, str]],
    blockers: Sequence[Mapping[str, str]],
) -> str:
    primary = next((row for row in comparison if row.get("model_name") == PRIMARY_MODEL), {})
    has_primary_signal = _to_float(primary.get("top5_lift", "0")) >= 1.2
    has_export_blocker = any(
        row.get("blocker") == "PER_SPECIALIST_EXPORT_HOOK_NOT_CONFIRMED"
        for row in blockers
    )
    if has_primary_signal and has_export_blocker:
        return "ADD_SPECIALIST_CONFIRMATION_EXPORT_HOOK_FIRST"
    if has_primary_signal:
        return "MANUAL_CONFIRM_PATCHTST_SPECIALIST_EXPORT"
    return "KEEP_SPECIALISTS_CONDITIONALLY_UNKNOWN_OR_WEAK"


def _manual_commands(base_dir: Path, fitted_dir: Path, previous_dir: Path) -> str:
    return "\n".join(
        [
            "# Manual M20 Specialist Confirmation Commands",
            "",
            "Codex must stop here. Arash must run long/export commands manually.",
            "",
            "## Current evidence refresh (light)",
            "```powershell",
            "python scripts/analyze_m20_specialist_conditional_usefulness.py "
            f"--base-run-dir {base_dir} --previous-run-dir {previous_dir}",
            "```",
            "",
            "## Future confirmation export (manual/long, after hook exists)",
            "```powershell",
            "python -m app.training "
            "--config configs/training.m20.json "
            f"--score-only {fitted_dir} "
            "--parquet-dir exports/feature_ohlc_for_colab "
            "--confirmation-window-start <START> "
            "--confirmation-window-end <END> "
            "--confirmation-tag <TAG> "
            "--export-specialist-predictions-only",
            "```",
            "",
            "If `--export-specialist-predictions-only` is unsupported, implement the "
            "lightweight hook first; do not launch a long score-only workaround from Codex.",
            "",
        ]
    )


def _post_export_commands(base_dir: Path) -> str:
    return "\n".join(
        [
            "# Post-Export Specialist Confirmation Analysis",
            "",
            "Run after a manual confirmation-window specialist export exists.",
            "",
            "```powershell",
            "python scripts/analyze_m20_specialist_conditional_usefulness.py "
            f"--base-run-dir {base_dir} --previous-run-dir <CONFIRMATION_RUN_DIR>",
            "```",
            "",
            "Expected follow-up: compare original OOF specialist slices against the "
            "confirmation-window specialist slices before any strategy claim.",
            "",
        ]
    )


def _markdown(
    plan: Mapping[str, Any],
    comparison: Sequence[Mapping[str, str]],
    targets: Sequence[Mapping[str, Any]],
    blockers: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "# M20 Specialist Confirmation Plan",
        "",
        f"- Recommendation: `{plan['recommendation']}`",
        f"- Primary candidate: `{plan['primary_candidate']}`",
        f"- Secondary candidate: `{plan['secondary_candidate']}`",
        "- Status: research-only planning; no export or score-only rerun executed.",
        "",
        "## Candidate Evidence",
    ]
    for row in comparison:
        lines.append(
            f"- `{row.get('model_name', '')}`: top5_lift={row.get('top5_lift', '')}, "
            f"PR-AUC={row.get('pr_auc', '')}, slices={row.get('enable_slice_count', '')}"
        )
    lines.extend(["", "## Target Slices"])
    for row in targets[:20]:
        lines.append(
            f"- `{row['model_name']}` {row['slice_family']}={row['slice_value']} "
            f"top5_lift={row['top5_lift']} priority={row['confirmation_priority']}"
        )
    lines.extend(["", "## Blockers"])
    for row in blockers:
        lines.append(f"- `{row['blocker']}` ({row['severity']}): {row['details']}")
    lines.extend(
        [
            "",
            "No runtime, registry, promotion, trading/backtest, model retrain, "
            "or profit-claim behavior is added.",
            "",
        ]
    )
    return "\n".join(lines)


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "plan_json": str(output_dir / "specialist_confirmation_plan.json"),
        "plan_md": str(output_dir / "specialist_confirmation_plan.md"),
        "target_slices_csv": str(output_dir / "target_slices.csv"),
        "required_export_schema_json": str(output_dir / "required_export_schema.json"),
        "manual_commands_md": str(output_dir / "manual_commands.md"),
        "post_export_analysis_commands_md": str(output_dir / "post_export_analysis_commands.md"),
        "blockers_csv": str(output_dir / "blockers.csv"),
    }


def _to_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


__all__ = ["write_m20_specialist_confirmation_plan"]
