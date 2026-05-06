"""Research-only M20 model/member audit and strategy candidate ledger."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


AUDIT_DIR_NAME = "model_member_audit"
HONESTY_FLAGS = (
    "RESEARCH_ONLY_MODEL_MEMBER_AUDIT",
    "AUTOGLUON_TREATED_AS_MODEL_FACTORY",
    "NO_RUNTIME_EFFECT",
    "NO_PROMOTION_EFFECT",
    "NO_REGISTRY_WRITE",
    "NOT_PROMOTABLE",
    "NO_PROFITABILITY_CLAIM",
    "STRATEGY_ENSEMBLE_DESIGN_ONLY",
    "CONDITIONAL_USEFULNESS_REQUIRES_CONFIRMATION",
    "SINGLE_RECENT_FOLD_ONLY",
    "LONG_RUNS_MANUAL_ONLY",
    "WEAK_CANDIDATES_RETAINED_FOR_CONDITIONAL_REVIEW",
    "AUTOGLUON_MEMBER_PREDICTIONS_MAY_REQUIRE_FUTURE_EXPORT",
)


def audit_m20_model_members(
    *,
    run_dir: Path,
    previous_run_dir: Path | None = None,
    fitted_models_dir: Path | None = None,
    load_autogluon_metadata: bool = False,
) -> dict[str, Any]:
    """Audit current M20 model candidates without running long jobs."""
    # pylint: disable=too-many-locals
    del load_autogluon_metadata
    resolved_run_dir = Path(run_dir).resolve()
    resolved_previous = Path(previous_run_dir).resolve() if previous_run_dir else None
    resolved_fitted = Path(fitted_models_dir).resolve() if fitted_models_dir else None
    audit_dir = resolved_run_dir / "research_labels" / "vol_scaled" / AUDIT_DIR_NAME
    audit_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    candidates.extend(_fee_baseline_candidates(resolved_run_dir))
    if resolved_previous:
        candidates.extend(_previous_run_candidates(resolved_previous))
    if resolved_fitted:
        candidates.extend(_fitted_model_candidates(resolved_fitted))
    autogluon_inventory = _autogluon_inventory(
        [resolved_run_dir, resolved_previous, resolved_fitted]
    )
    candidates.extend(_autogluon_candidates(autogluon_inventory))
    candidates = _dedupe_candidates(candidates)
    conditional = _conditional_linkage(resolved_run_dir)
    ledger = [_ledger_row(candidate, conditional) for candidate in candidates]
    evidence_rows = [_evidence_row(candidate, conditional) for candidate in candidates]
    next_actions = [_next_action_row(row) for row in ledger]
    output_files = _output_files(audit_dir, bool(autogluon_inventory))
    report = {
        "run_dir": str(resolved_run_dir),
        "previous_run_dir": str(resolved_previous) if resolved_previous else "",
        "fitted_models_dir": str(resolved_fitted) if resolved_fitted else "",
        "audit_dir": str(audit_dir),
        "candidate_count": len(candidates),
        "autogluon_member_count": len(
            [row for row in candidates if row["candidate_taxonomy"] == "AUTOGLUON_MEMBER_MODEL"]
        ),
        "honesty_flags": list(HONESTY_FLAGS),
        "recommendations": [
            "1. Confirm fee-exceedance logistic useful slices on another window/fold.",
            "2. Add/export AutoGluon member-level predictions for separate specialist review.",
            "3. Keep PatchTST/NHITS conditionally unknown until slice analysis proves otherwise.",
        ],
        "output_files": output_files,
    }
    manifest = {
        "run_dir": str(resolved_run_dir),
        "audit_dir": str(audit_dir),
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_effect": "none_research_only",
        "registry_write": False,
        "promotion_effect": False,
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["model_member_audit_manifest_json"]), manifest)
    write_json_artifact(Path(output_files["model_member_audit_report_json"]), report)
    Path(output_files["model_member_audit_report_md"]).write_text(
        _report_markdown(report, ledger),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["discovered_model_candidates_csv"]), candidates)
    write_csv_artifact(Path(output_files["strategy_ensemble_candidate_ledger_csv"]), ledger)
    write_csv_artifact(Path(output_files["candidate_evidence_state_csv"]), evidence_rows)
    write_csv_artifact(Path(output_files["candidate_next_actions_csv"]), next_actions)
    write_json_artifact(Path(output_files["conditional_evidence_linkage_json"]), conditional)
    write_csv_artifact(
        Path(output_files["autogluon_member_inventory_csv"]),
        autogluon_inventory or [_empty_autogluon_inventory_row()],
    )
    Path(output_files["manual_confirmation_commands_md"]).write_text(
        _manual_commands_markdown(resolved_run_dir, resolved_previous, resolved_fitted),
        encoding="utf-8",
    )
    return make_json_safe({**report, "manifest": manifest, "ledger": ledger})


def _fee_baseline_candidates(run_dir: Path) -> list[dict[str, Any]]:
    baseline_dir = run_dir / "research_labels" / "vol_scaled" / "fee_exceedance_baselines"
    metrics_path = baseline_dir / "fee_baseline_metrics.json"
    if not metrics_path.exists():
        return []
    metrics = _load_json(metrics_path)
    rows = []
    for baseline in metrics.get("baselines", []):
        name = str(baseline.get("baseline_name", "unknown"))
        rows.append(
            {
                "candidate_id": f"{run_dir.name}:{name}",
                "model_name": name,
                "source_run": str(run_dir),
                "source_artifact_path": str(baseline_dir),
                "candidate_taxonomy": (
                    "RESEARCH_LOGISTIC_BASELINE"
                    if name == "logistic_regression_tiny" else "SIMPLE_BASELINE"
                ),
                "target_type": "fee_exceedance",
                "has_row_level_predictions": (baseline_dir / f"predictions_{name}.csv").exists(),
                "has_probability_predictions": name != "always_negative",
                "runtime_loadable": False,
                "research_only": True,
                "best_known_global_metric": baseline.get("average_precision", ""),
            }
        )
    return rows


def _previous_run_candidates(run_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for metric in _read_csv_rows(run_dir / "fold_metrics.csv"):
        name = str(metric.get("model_name", "unknown"))
        rows.append(
            {
                "candidate_id": f"{run_dir.name}:{name}",
                "model_name": name,
                "source_run": str(run_dir),
                "source_artifact_path": str(run_dir / "oof_predictions.csv"),
                "candidate_taxonomy": _taxonomy_for_name(name),
                "target_type": "direction",
                "has_row_level_predictions": (run_dir / "oof_predictions.csv").exists(),
                "has_probability_predictions": True,
                "runtime_loadable": name.startswith("neuralforecast_"),
                "research_only": False,
                "best_known_global_metric": metric.get("mean_long_only_net_value_proxy", ""),
            }
        )
    return rows


def _fitted_model_candidates(fitted_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(fitted_dir.rglob("*.joblib")):
        name = path.stem
        rows.append(
            {
                "candidate_id": f"{fitted_dir.parent.name}:{path.parent.name}:{name}",
                "model_name": name,
                "source_run": str(fitted_dir.parent),
                "source_artifact_path": str(path),
                "candidate_taxonomy": _taxonomy_for_name(name),
                "target_type": "direction",
                "has_row_level_predictions": False,
                "has_probability_predictions": False,
                "runtime_loadable": name.startswith("neuralforecast_"),
                "research_only": False,
                "best_known_global_metric": "",
            }
        )
    return rows


def _autogluon_inventory(paths: Sequence[Path | None]) -> list[dict[str, Any]]:
    inventory = []
    for root in [path for path in paths if path is not None and path.exists()]:
        for path in sorted(root.rglob("*")):
            lower = path.name.lower()
            if AUDIT_DIR_NAME in [part.lower() for part in path.parts]:
                continue
            if lower == "autogluon_member_inventory.csv":
                continue
            if "autogluon" not in str(path).lower() and lower not in {
                "leaderboard.csv",
                "leaderboard.json",
                "predictor.pkl",
            }:
                continue
            inventory.append(
                {
                    "artifact_path": str(path),
                    "artifact_name": path.name,
                    "artifact_type": "directory" if path.is_dir() else "file",
                    "member_name": _member_name_from_path(path),
                    "metadata_available": True,
                    "member_predictions_available": False,
                    "requires_future_prediction_export": True,
                }
            )
    return inventory


def _autogluon_candidates(inventory: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not inventory:
        return [
            {
                "candidate_id": "autogluon:metadata_missing",
                "model_name": "autogluon_tabular",
                "source_run": "",
                "source_artifact_path": "",
                "candidate_taxonomy": "AUTOGLUON_ENSEMBLE",
                "target_type": "direction",
                "has_row_level_predictions": False,
                "has_probability_predictions": False,
                "runtime_loadable": False,
                "research_only": False,
                "best_known_global_metric": "",
            }
        ]
    return [
        {
            "candidate_id": f"autogluon:{row['member_name']}",
            "model_name": row["member_name"],
            "source_run": "",
            "source_artifact_path": row["artifact_path"],
            "candidate_taxonomy": (
                "AUTOGLUON_ENSEMBLE"
                if str(row["member_name"]).lower() == "autogluon_tabular"
                else "AUTOGLUON_MEMBER_MODEL"
            ),
            "target_type": "direction",
            "has_row_level_predictions": False,
            "has_probability_predictions": False,
            "runtime_loadable": False,
            "research_only": False,
            "best_known_global_metric": "",
        }
        for row in inventory
    ]


def _conditional_linkage(run_dir: Path) -> dict[str, Any]:
    report_path = (
        run_dir
        / "research_labels"
        / "vol_scaled"
        / "conditional_usefulness_full_test"
        / "conditional_usefulness_report.json"
    )
    if not report_path.exists():
        return {"available": False}
    report = _load_json(report_path)
    slice_rows = _read_csv_rows(
        report_path.parent / "conditional_usefulness_by_slice.csv"
    )
    enable = [row for row in slice_rows if row.get("classification") == "ENABLE_CANDIDATE"]
    disable = [row for row in slice_rows if row.get("classification") == "DISABLE_CANDIDATE"]
    strong_names = {"momentum=flat", "range=low", "BTC/USD", "macd=positive", "volume=low"}
    strong = [
        row for row in enable
        if row.get("slice_value") in strong_names
        or f"{row.get('slice_family')}={row.get('slice_value')}" in strong_names
    ]
    return {
        "available": True,
        "report_path": str(report_path),
        "prediction_rows_analyzed": report.get("prediction_rows_analyzed", 0),
        "search_breadth": report.get("search_breadth", {}),
        "honesty_flags": report.get("honesty_flags", []),
        "recommendation": report.get("recommendation", ""),
        "strong_slices": strong,
        "disable_slices": disable,
    }


def _ledger_row(candidate: Mapping[str, Any], conditional: Mapping[str, Any]) -> dict[str, Any]:
    is_fee_logistic = candidate["model_name"] == "logistic_regression_tiny"
    evidence = _evidence_state(candidate, conditional if is_fee_logistic else {})
    return {
        "candidate_id": candidate["candidate_id"],
        "model_name": candidate["model_name"],
        "source_run": candidate["source_run"],
        "source_artifact_path": candidate["source_artifact_path"],
        "candidate_taxonomy": candidate["candidate_taxonomy"],
        "possible_role_taxonomy": _role(candidate, evidence),
        "evidence_state": evidence,
        "target_type": candidate["target_type"],
        "prediction_availability": bool(candidate["has_row_level_predictions"]),
        "probability_availability": bool(candidate["has_probability_predictions"]),
        "condition_slice_evidence_availability": bool(
            is_fee_logistic and conditional.get("available")
        ),
        "best_known_global_metric": candidate.get("best_known_global_metric", ""),
        "best_known_slice": _best_slice(conditional) if is_fee_logistic else "",
        "disabled_slices": _disabled_slices(conditional) if is_fee_logistic else "",
        "known_blockers": _blockers(candidate, evidence),
        "next_required_action": _next_action(candidate, evidence),
        "runtime_eligible": bool(candidate["runtime_loadable"]),
        "promotable": False,
        "notes": _notes(candidate, evidence),
    }


def _evidence_row(candidate: Mapping[str, Any], conditional: Mapping[str, Any]) -> dict[str, Any]:
    state = _evidence_state(candidate, conditional)
    return {
        "candidate_id": candidate["candidate_id"],
        "model_name": candidate["model_name"],
        "evidence_state": state,
        "has_row_level_predictions": candidate["has_row_level_predictions"],
        "has_probability_predictions": candidate["has_probability_predictions"],
        "conditional_evidence_available": bool(conditional.get("available")),
    }


def _next_action_row(ledger_row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": ledger_row["candidate_id"],
        "model_name": ledger_row["model_name"],
        "next_required_action": ledger_row["next_required_action"],
        "runtime_eligible": ledger_row["runtime_eligible"],
        "promotable": ledger_row["promotable"],
    }


def _evidence_state(candidate: Mapping[str, Any], conditional: Mapping[str, Any]) -> str:
    if conditional.get("available"):
        return "HAS_FULL_TEST_CONDITIONAL_EVIDENCE"
    if candidate["has_row_level_predictions"] and candidate["has_probability_predictions"]:
        return "HAS_OOF_PREDICTIONS_ONLY"
    if candidate["has_row_level_predictions"]:
        return "HAS_OOF_PREDICTIONS_ONLY"
    if candidate["source_artifact_path"]:
        return "HAS_MODEL_ARTIFACT_ONLY"
    if candidate["candidate_taxonomy"].startswith("AUTOGLUON"):
        return "HAS_METADATA_ONLY"
    return "NOT_INSPECTABLE_CURRENTLY"


def _role(candidate: Mapping[str, Any], evidence: str) -> str:
    name = str(candidate["model_name"])
    taxonomy = str(candidate["candidate_taxonomy"])
    if name == "logistic_regression_tiny":
        return "FEE_EXCEEDANCE_FILTER"
    if taxonomy == "SIMPLE_BASELINE":
        return "DISABLE_CANDIDATE"
    if "patchtst" in name:
        return "RANGE_SIDEWAYS_SPECIALIST"
    if "nhits" in name:
        return "TREND_SPARK_SPECIALIST"
    if taxonomy.startswith("AUTOGLUON"):
        return "UNKNOWN_ROLE" if evidence == "HAS_METADATA_ONLY" else "GLOBAL_OPPORTUNITY_GATE"
    return "UNKNOWN_ROLE"


def _taxonomy_for_name(name: str) -> str:
    if name.startswith("neuralforecast_"):
        return "NEURALFORECAST_SPECIALIST"
    if name in {"dummy_most_frequent", "persistence_3", "always_negative"}:
        return "SIMPLE_BASELINE"
    if "autogluon" in name:
        return "AUTOGLUON_ENSEMBLE"
    return "UNKNOWN_MODEL"


def _blockers(candidate: Mapping[str, Any], evidence: str) -> str:
    blockers = []
    if evidence in {"HAS_MODEL_ARTIFACT_ONLY", "HAS_METADATA_ONLY", "NOT_INSPECTABLE_CURRENTLY"}:
        blockers.append("needs_row_level_prediction_export")
    if str(candidate["candidate_taxonomy"]).startswith("AUTOGLUON"):
        blockers.append("autogluon_member_predictions_may_require_future_export")
    if candidate["model_name"] in {"neuralforecast_nhits", "neuralforecast_patchtst"}:
        blockers.append("conditional_usefulness_unknown")
    return ";".join(blockers)


def _next_action(candidate: Mapping[str, Any], evidence: str) -> str:
    if candidate["model_name"] == "logistic_regression_tiny":
        return "confirm useful fee-exceedance slices on another run/window"
    if str(candidate["candidate_taxonomy"]).startswith("AUTOGLUON"):
        return "export AutoGluon member-level predictions for conditional usefulness analysis"
    if evidence == "HAS_OOF_PREDICTIONS_ONLY":
        return "run conditional analysis on existing row-level predictions"
    return "export row-level probabilities before conditional claims"


def _notes(candidate: Mapping[str, Any], evidence: str) -> str:
    if candidate["model_name"] in {"neuralforecast_nhits", "neuralforecast_patchtst"}:
        return (
            "Global direction-policy evidence is weak/negative; "
            "conditional usefulness remains unknown."
        )
    if candidate["model_name"] == "logistic_regression_tiny":
        return (
            "First full-test signal-positive fee-exceedance research candidate; "
            "single recent fold only."
        )
    return f"Evidence state: {evidence}"


def _best_slice(conditional: Mapping[str, Any]) -> str:
    rows = conditional.get("strong_slices", [])
    if not rows:
        return ""
    row = rows[0]
    return f"{row.get('slice_family')}={row.get('slice_value')}"


def _disabled_slices(conditional: Mapping[str, Any]) -> str:
    return ";".join(
        f"{row.get('slice_family')}={row.get('slice_value')}"
        for row in conditional.get("disable_slices", [])
    )


def _dedupe_candidates(candidates: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    output = []
    for candidate in candidates:
        key = str(candidate["candidate_id"])
        if key in seen:
            continue
        seen.add(key)
        output.append(dict(candidate))
    return sorted(output, key=lambda row: str(row["candidate_id"]))


def _output_files(audit_dir: Path, autogluon_exists: bool) -> dict[str, str]:
    del autogluon_exists
    names = {
        "model_member_audit_manifest_json": "model_member_audit_manifest.json",
        "model_member_audit_report_json": "model_member_audit_report.json",
        "model_member_audit_report_md": "model_member_audit_report.md",
        "discovered_model_candidates_csv": "discovered_model_candidates.csv",
        "strategy_ensemble_candidate_ledger_csv": "strategy_ensemble_candidate_ledger.csv",
        "candidate_evidence_state_csv": "candidate_evidence_state.csv",
        "autogluon_member_inventory_csv": "autogluon_member_inventory.csv",
        "candidate_next_actions_csv": "candidate_next_actions.csv",
        "conditional_evidence_linkage_json": "conditional_evidence_linkage.json",
        "manual_confirmation_commands_md": "manual_confirmation_commands.md",
    }
    return {key: str(audit_dir / name) for key, name in names.items()}


def _report_markdown(report: Mapping[str, Any], ledger: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        [
            "# M20 Model Member Audit",
            "",
            f"- Candidate count: `{report['candidate_count']}`",
            f"- AutoGluon member count: `{report['autogluon_member_count']}`",
            f"- Honesty flags: `{', '.join(report['honesty_flags'])}`",
            "",
            "## Recommendations",
            "",
            *[f"- {item}" for item in report["recommendations"]],
            "",
            "## Candidate Ledger",
            "",
            *[
                f"- `{row['model_name']}`: `{row['candidate_taxonomy']}` / "
                f"`{row['evidence_state']}` / `{row['possible_role_taxonomy']}`"
                for row in ledger[:20]
            ],
            "",
            "No runtime ensemble behavior, registry write, promotion, "
            "or profitability claim is made.",
            "",
        ]
    )


def _manual_commands_markdown(run_dir: Path, previous: Path | None, fitted: Path | None) -> str:
    previous_arg = f" --previous-run-dir {previous}" if previous else ""
    fitted_arg = f" --fitted-models-dir {fitted}" if fitted else ""
    return "\n".join(
        [
            "# M20 Manual Confirmation Commands",
            "",
            "## A. AutoGluon Member Prediction Export",
            "",
            "- Heaviness: long/manual",
            "- Command: prepare an AutoGluon/member prediction export from the "
            "original training pipeline; do not run from Codex.",
            "- Expected artifact: member-level row predictions with `symbol`, "
            "`interval_begin`, `model_name`, and probability columns.",
            "",
            "## B. M20 Score-Only Rerun With Training Frame Export",
            "",
            "- Heaviness: long/manual",
            "- Command: `python -m app.training --config configs/training.m20.json "
            "--score-only artifacts/training/m20/20260405T023104Z/fitted_models "
            "--parquet-dir exports/feature_ohlc_for_colab --export-training-frame-only`",
            "- Expected artifact: `artifacts/training/m20/<run_id>/training_frame/`.",
            "",
            "## C. Conditional Usefulness Confirmation",
            "",
            "- Heaviness: light after predictions exist",
            "- Command: `python scripts/analyze_m20_conditional_usefulness.py "
            f"--run-dir {run_dir} --prediction-source full-test`",
            "",
            "## D. Metadata-Only Model Member Audit",
            "",
            "- Heaviness: light",
            "- Command: `python scripts/audit_m20_model_members.py "
            f"--run-dir {run_dir}{previous_arg}{fitted_arg} --load-autogluon-metadata`",
            "",
        ]
    )


def _member_name_from_path(path: Path) -> str:
    if path.name.lower() in {"leaderboard.csv", "leaderboard.json", "predictor.pkl"}:
        return "autogluon_tabular"
    return path.stem or path.name


def _empty_autogluon_inventory_row() -> dict[str, Any]:
    return {
        "artifact_path": "",
        "artifact_name": "",
        "artifact_type": "",
        "member_name": "",
        "metadata_available": False,
        "member_predictions_available": False,
        "requires_future_prediction_export": True,
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}
