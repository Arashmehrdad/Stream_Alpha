"""Research-only M20 specialist row-level prediction export plan."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "specialist_prediction_export_plan"
HONESTY_FLAGS = (
    "RESEARCH_ONLY_SPECIALIST_PREDICTION_EXPORT_PLAN",
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
    {
        "column": "symbol",
        "required": True,
        "purpose": "safe row join key",
        "leakage_policy": "allowed_key",
    },
    {
        "column": "interval_begin",
        "required": True,
        "purpose": "safe row join key",
        "leakage_policy": "allowed_key",
    },
    {
        "column": "fold_index",
        "required": True,
        "purpose": "fold provenance for conditional analysis",
        "leakage_policy": "allowed_metadata",
    },
    {
        "column": "model_name",
        "required": True,
        "purpose": "specialist identity",
        "leakage_policy": "allowed_metadata",
    },
    {
        "column": "candidate_id",
        "required": True,
        "purpose": "stable specialist candidate id",
        "leakage_policy": "allowed_metadata",
    },
    {
        "column": "prediction_source",
        "required": True,
        "purpose": "OOF, validation, test, or score-only source label",
        "leakage_policy": "allowed_metadata",
    },
    {
        "column": "y_true",
        "required": False,
        "purpose": "explicit evaluation target when available",
        "leakage_policy": "allowed_target_for_evaluation_only",
    },
    {
        "column": "probability_or_score_columns",
        "required": True,
        "purpose": "row-level probability or continuous score for ranking",
        "leakage_policy": "allowed_prediction_output",
    },
)
FORBIDDEN_SCHEMA_NOTES = (
    "future_return",
    "forward_return",
    "realized_outcome",
    "barrier_metadata",
    "label_columns_as_features",
    "target_derived_features",
    "runtime_registry_fields",
)


def write_m20_specialist_prediction_export_plan(
    *,
    base_run_dir: Path,
    fitted_models_dir: Path,
    previous_run_dir: Path,
) -> dict[str, Any]:
    """Write a manual-only plan for row-level specialist prediction export."""
    # pylint: disable=too-many-locals
    base_dir = Path(base_run_dir).resolve()
    fitted_dir = Path(fitted_models_dir).resolve()
    previous_dir = Path(previous_run_dir).resolve()
    vol_dir = base_dir / "research_labels" / "vol_scaled"
    output_dir = vol_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    ledger_path = vol_dir / "model_member_audit" / "strategy_ensemble_candidate_ledger.csv"
    next_actions_path = vol_dir / "model_member_audit" / "candidate_next_actions.csv"
    adjudication_path = (
        vol_dir
        / "m20_research_path_adjudication"
        / "research_path_adjudication.json"
    )
    ledger_rows = _read_csv(ledger_path)
    next_action_rows = _read_csv(next_actions_path)
    previous_prediction_files = _previous_prediction_files(previous_dir)
    targets = _candidate_targets(
        ledger_rows=ledger_rows,
        next_action_rows=next_action_rows,
        previous_prediction_files=previous_prediction_files,
    )
    blockers = _blockers(
        targets=targets,
        fitted_models_dir=fitted_dir,
        previous_run_dir=previous_dir,
        previous_prediction_files=previous_prediction_files,
    )
    recommendation = _recommendation(targets, blockers)
    output_files = _output_files(output_dir)
    plan = {
        "recommendation": recommendation,
        "summary": (
            "Prepare manual-only row-level specialist prediction export for NHITS "
            "and PatchTST before further conditional specialist claims."
        ),
        "base_run_dir": str(base_dir),
        "fitted_models_dir": str(fitted_dir),
        "previous_run_dir": str(previous_dir),
        "source_artifacts": {
            "candidate_ledger": str(ledger_path),
            "candidate_next_actions": str(next_actions_path),
            "research_path_adjudication": str(adjudication_path),
        },
        "target_count": len(targets),
        "blocker_count": len(blockers),
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "manual_only": True,
        "codex_must_not_execute_long_exports": True,
        "output_files": output_files,
    }
    manifest = {
        "output_dir": str(output_dir),
        "source": "existing_m20_research_artifacts",
        "honesty_flags": list(HONESTY_FLAGS),
        "recommendation": recommendation,
        "output_files": output_files,
    }

    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["plan_json"]), plan)
    write_json_artifact(
        Path(output_files["required_prediction_schema_json"]),
        {
            "required_columns": list(REQUIRED_SCHEMA),
            "forbidden_columns_or_uses": list(FORBIDDEN_SCHEMA_NOTES),
            "notes": (
                "Prediction files may carry y_true for evaluation, but must not "
                "carry future returns, realized outcomes, labels, or target-derived "
                "fields as candidate features."
            ),
        },
    )
    write_csv_artifact(Path(output_files["candidate_export_targets_csv"]), targets)
    write_csv_artifact(Path(output_files["blockers_csv"]), blockers)
    Path(output_files["manual_export_commands_md"]).write_text(
        _manual_commands(base_dir, fitted_dir, previous_dir, recommendation, blockers),
        encoding="utf-8",
    )
    Path(output_files["post_export_analysis_commands_md"]).write_text(
        _post_export_commands(base_dir),
        encoding="utf-8",
    )
    Path(output_files["plan_md"]).write_text(
        _markdown(plan, targets, blockers),
        encoding="utf-8",
    )
    return make_json_safe(plan)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _previous_prediction_files(previous_run_dir: Path) -> list[dict[str, str]]:
    candidates = [
        previous_run_dir / "oof_predictions.csv",
        previous_run_dir
        / "research_labels"
        / "vol_scaled"
        / "baselines"
        / "predictions_score_only_y_pred.csv",
    ]
    rows = []
    for path in candidates:
        if not path.exists():
            continue
        header = _csv_header(path)
        rows.append(
            {
                "path": str(path),
                "has_model_name": str("model_name" in header),
                "has_probability": str(
                    any(column.startswith("prob") for column in header)
                    or "score" in ",".join(header).lower()
                ),
                "has_future_or_proxy_columns": str(
                    any(
                        token in column.lower()
                        for column in header
                        for token in ("future", "gross", "net")
                    )
                ),
            }
        )
    return rows


def _csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader, [])


def _candidate_targets(
    *,
    ledger_rows: Sequence[Mapping[str, str]],
    next_action_rows: Sequence[Mapping[str, str]],
    previous_prediction_files: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    by_id: dict[str, dict[str, str]] = {}
    for row in ledger_rows:
        candidate_id = row.get("candidate_id", "")
        model_name = row.get("model_name", "")
        if _is_specialist(model_name):
            by_id[candidate_id] = {
                "candidate_id": candidate_id,
                "model_name": model_name,
                "source_run": row.get("source_run", ""),
                "candidate_taxonomy": row.get("candidate_taxonomy", ""),
                "evidence_state": row.get("evidence_state", ""),
                "prediction_availability": row.get("prediction_availability", ""),
                "probability_availability": row.get("probability_availability", ""),
                "target_type": row.get("target_type", ""),
                "export_status": _export_status(row),
                "recommended_action": row.get(
                    "next_required_action",
                    "export row-level predictions before conditional claims",
                ),
                "manual_only": "True",
                "promotable": "False",
            }
    for row in next_action_rows:
        candidate_id = row.get("candidate_id", "")
        model_name = row.get("model_name", "")
        if _is_specialist(model_name) and candidate_id not in by_id:
            by_id[candidate_id] = {
                "candidate_id": candidate_id,
                "model_name": model_name,
                "source_run": candidate_id.split(":", maxsplit=1)[0],
                "candidate_taxonomy": "NEURALFORECAST_SPECIALIST",
                "evidence_state": "UNKNOWN_FROM_NEXT_ACTIONS",
                "prediction_availability": "",
                "probability_availability": "",
                "target_type": "direction",
                "export_status": "NEEDS_ROW_LEVEL_EXPORT",
                "recommended_action": row.get("next_required_action", ""),
                "manual_only": "True",
                "promotable": "False",
            }
    if previous_prediction_files:
        for model_name in ("neuralforecast_nhits", "neuralforecast_patchtst"):
            candidate_id = f"20260427T112021Z:{model_name}"
            if candidate_id in by_id:
                by_id[candidate_id]["export_status"] = "EXISTING_OOF_PREDICTIONS_AVAILABLE"
                by_id[candidate_id]["recommended_action"] = (
                    "sanitize existing OOF predictions and run conditional analysis"
                )
    return [by_id[key] for key in sorted(by_id)]


def _is_specialist(model_name: str) -> bool:
    return model_name in {"neuralforecast_nhits", "neuralforecast_patchtst"}


def _export_status(row: Mapping[str, str]) -> str:
    if row.get("prediction_availability", "").lower() == "true":
        return "EXISTING_ROW_LEVEL_PREDICTIONS_AVAILABLE"
    return "NEEDS_ROW_LEVEL_EXPORT"


def _blockers(
    *,
    targets: Sequence[Mapping[str, str]],
    fitted_models_dir: Path,
    previous_run_dir: Path,
    previous_prediction_files: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    blockers = [
        {
            "blocker": "LONG_RUNS_MANUAL_ONLY",
            "severity": "required_control",
            "details": "Codex must not launch score-only or specialist export jobs.",
        },
        {
            "blocker": "PER_SPECIALIST_EXPORT_HOOK_NOT_CONFIRMED",
            "severity": "blocking_for_new_exports",
            "details": (
                "Current CLI may require a score-only rerun to emit per-specialist "
                "row-level predictions; add a lightweight export hook first if needed."
            ),
        },
    ]
    if not fitted_models_dir.exists():
        blockers.append(
            {
                "blocker": "FITTED_MODELS_DIR_MISSING",
                "severity": "blocking_for_manual_export",
                "details": str(fitted_models_dir),
            }
        )
    if not previous_run_dir.exists():
        blockers.append(
            {
                "blocker": "PREVIOUS_OOF_RUN_MISSING",
                "severity": "blocking_for_existing_oof_analysis",
                "details": str(previous_run_dir),
            }
        )
    if not previous_prediction_files:
        blockers.append(
            {
                "blocker": "EXISTING_OOF_PREDICTIONS_NOT_FOUND",
                "severity": "blocking_for_existing_oof_analysis",
                "details": "No previous oof_predictions.csv or score-only predictions found.",
            }
        )
    if not any(row.get("export_status", "").startswith("EXISTING") for row in targets):
        blockers.append(
            {
                "blocker": "NO_IMMEDIATE_SPECIALIST_ROW_LEVEL_ANALYSIS",
                "severity": "research_blocker",
                "details": "No specialist target has clean immediate row-level predictions.",
            }
        )
    blockers.append(
        {
            "blocker": "AUTOGLUON_MEMBER_PREDICTIONS_MISSING",
            "severity": "tracked_gap",
            "details": "AutoGluon remains a model factory, but member predictions are absent.",
        }
    )
    return blockers


def _recommendation(
    targets: Sequence[Mapping[str, str]],
    blockers: Sequence[Mapping[str, str]],
) -> str:
    has_existing = any(
        row.get("export_status") == "EXISTING_OOF_PREDICTIONS_AVAILABLE"
        for row in targets
    )
    has_export_hook_blocker = any(
        row.get("blocker") == "PER_SPECIALIST_EXPORT_HOOK_NOT_CONFIRMED"
        for row in blockers
    )
    if has_existing and not has_export_hook_blocker:
        return "USE_EXISTING_OOF_PREDICTIONS_FOR_CONDITIONAL_ANALYSIS"
    if has_existing:
        return "ADD_LIGHTWEIGHT_PREDICTION_EXPORT_HOOK_FIRST"
    if targets:
        return "ADD_LIGHTWEIGHT_PREDICTION_EXPORT_HOOK_FIRST"
    return "BLOCKED_NEEDS_PIPELINE_SUPPORT"


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "plan_json": str(output_dir / "specialist_prediction_export_plan.json"),
        "plan_md": str(output_dir / "specialist_prediction_export_plan.md"),
        "candidate_export_targets_csv": str(output_dir / "candidate_export_targets.csv"),
        "required_prediction_schema_json": str(
            output_dir / "required_prediction_schema.json"
        ),
        "manual_export_commands_md": str(output_dir / "manual_export_commands.md"),
        "post_export_analysis_commands_md": str(
            output_dir / "post_export_analysis_commands.md"
        ),
        "blockers_csv": str(output_dir / "blockers.csv"),
    }


def _manual_commands(
    base_dir: Path,
    fitted_dir: Path,
    previous_dir: Path,
    recommendation: str,
    blockers: Sequence[Mapping[str, str]],
) -> str:
    blocker_lines = "\n".join(
        f"- `{row['blocker']}`: {row['details']}" for row in blockers
    )
    return "\n".join(
        [
            "# Manual M20 Specialist Prediction Export Commands",
            "",
            "Codex must stop here. Arash must run any long command manually.",
            "",
            f"- Recommendation: `{recommendation}`",
            "- Heaviness: planning/light inspection only inside Codex; "
            "score-only export is manual.",
            "",
            "## Existing OOF inspection (light)",
            "```powershell",
            f"Get-Content {previous_dir / 'oof_predictions.csv'} -TotalCount 5",
            "```",
            "",
            "## Manual score-only export candidate (long/manual)",
            "```powershell",
            "python -m app.training "
            "--config configs/training.m20.json "
            f"--score-only {fitted_dir} "
            "--parquet-dir exports/feature_ohlc_for_colab "
            "--export-training-frame",
            "```",
            "",
            "Expected output: a manually launched run with row-level predictions for "
            "NHITS/PatchTST that can be sanitized into the required schema.",
            "",
            "## Base run for downstream planning",
            f"- `{base_dir}`",
            "",
            "## Blockers",
            blocker_lines,
            "",
        ]
    )


def _post_export_commands(base_dir: Path) -> str:
    return "\n".join(
        [
            "# Post-Export Specialist Analysis Commands",
            "",
            "Run only after Arash manually creates row-level specialist prediction artifacts.",
            "",
            "```powershell",
            "# Placeholder until the lightweight specialist export hook defines its output dir.",
            "python scripts/plan_m20_specialist_prediction_export.py "
            f"--base-run-dir {base_dir} "
            "--fitted-models-dir artifacts/training/m20/20260405T023104Z/fitted_models "
            "--previous-run-dir artifacts/training/m20/20260427T112021Z",
            "```",
            "",
            "Next code action if no clean per-specialist predictions exist: "
            "`ADD_LIGHTWEIGHT_PREDICTION_EXPORT_HOOK_FIRST`.",
            "",
        ]
    )


def _markdown(
    plan: Mapping[str, Any],
    targets: Sequence[Mapping[str, str]],
    blockers: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "# M20 Specialist Prediction Export Plan",
        "",
        f"- Recommendation: `{plan['recommendation']}`",
        "- Status: research-only planning; no export or score-only rerun executed.",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "",
        "## Export Targets",
    ]
    for row in targets[:20]:
        lines.append(
            "- "
            f"`{row['candidate_id']}` ({row['model_name']}): "
            f"`{row['export_status']}`; {row['recommended_action']}"
        )
    lines.extend(["", "## Required Prediction Schema"])
    for row in REQUIRED_SCHEMA:
        lines.append(
            f"- `{row['column']}`: required={row['required']}; {row['purpose']}"
        )
    lines.extend(["", "## Blockers"])
    for row in blockers:
        lines.append(f"- `{row['blocker']}` ({row['severity']}): {row['details']}")
    lines.extend(
        [
            "",
            "## Honesty Flags",
            ", ".join(f"`{flag}`" for flag in HONESTY_FLAGS),
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["write_m20_specialist_prediction_export_plan"]
