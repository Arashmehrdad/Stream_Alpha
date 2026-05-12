"""Research-only M20 policy input availability audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_policy_research_common import (
    FORBIDDEN_SELECTION_INPUTS,
    HONESTY_FLAGS,
    duplicate_key_count,
    matched_key_count,
    preferred_join_keys,
    read_csv_header,
    read_csv_rows,
    vol_scaled_dir,
    write_csv_artifact,
    write_json_artifact,
)


DEFAULT_OUTPUT_NAME = "policy_input_availability_audit"
DEFAULT_CANDIDATE_DIR_NAME = "strategy_candidate_v2_refined_factory"
DEFAULT_ECONOMIC_DIR_NAME = "economic_outcome_artifacts"
DEFAULT_FEATURE_DIR_NAME = "research_feature_enrichment"


def audit_m20_policy_inputs(
    *,
    source_run_dir: Path,
    prediction_run_dir: Path,
    output_name: str = DEFAULT_OUTPUT_NAME,
    candidate_dir: Path | None = None,
    economic_outcome_dir: Path | None = None,
    research_feature_dir: Path | None = None,
) -> dict[str, Any]:
    """Audit whether existing M20 artifacts can support decision-policy research."""
    # pylint: disable=too-many-arguments,too-many-locals
    source_dir = Path(source_run_dir).resolve()
    prediction_dir = Path(prediction_run_dir).resolve()
    research_dir = vol_scaled_dir(source_dir)
    output_dir = research_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = _input_paths(
        research_dir,
        prediction_dir,
        candidate_dir,
        economic_outcome_dir,
        research_feature_dir,
    )
    rows_by_name = {name: read_csv_rows(path) for name, path in paths.items()}
    headers_by_name = {name: read_csv_header(path) for name, path in paths.items()}
    datasets = {
        name: _dataset_status(name, path, rows_by_name[name], headers_by_name[name])
        for name, path in paths.items()
    }
    join_readiness = _join_readiness(rows_by_name, headers_by_name)
    missing_inputs = _missing_inputs(datasets)
    readiness = _policy_family_readiness(datasets, join_readiness)
    leakage_audit = _leakage_audit(datasets)
    ready = any(row["readiness_status"].endswith("_READY") for row in readiness)
    recommendation = _recommendation(ready, missing_inputs)
    output_files = _output_files(output_dir)
    manifest = {
        "source_run_dir": str(source_dir),
        "prediction_run_dir": str(prediction_dir),
        "input_paths": {name: str(path) for name, path in paths.items()},
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    report = {
        "summary": "M20 research-only policy input availability audit.",
        "source_run_dir": str(source_dir),
        "prediction_run_dir": str(prediction_dir),
        "policy_evaluation_ready": ready,
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "missing_input_count": len(missing_inputs),
        "overall_status": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["policy_input_availability_report_json"]), report)
    Path(output_files["policy_input_availability_report_md"]).write_text(
        _markdown(report, readiness, missing_inputs),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["available_inputs_csv"]), list(datasets.values()))
    write_csv_artifact(Path(output_files["join_readiness_csv"]), join_readiness)
    write_csv_artifact(Path(output_files["missing_inputs_csv"]), missing_inputs)
    write_csv_artifact(Path(output_files["policy_family_readiness_csv"]), readiness)
    write_csv_artifact(Path(output_files["leakage_audit_csv"]), leakage_audit)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "available_inputs": list(datasets.values()),
            "join_readiness": join_readiness,
            "missing_inputs": missing_inputs,
            "policy_family_readiness": readiness,
            "leakage_audit": leakage_audit,
            "recommendation_payload": recommendation,
        }
    )


def _input_paths(
    research_dir: Path,
    prediction_dir: Path,
    candidate_dir: Path | None,
    economic_outcome_dir: Path | None,
    research_feature_dir: Path | None,
) -> dict[str, Path]:
    candidate_path = (
        Path(candidate_dir).resolve()
        if candidate_dir
        else research_dir / DEFAULT_CANDIDATE_DIR_NAME
    )
    outcome_path = (
        Path(economic_outcome_dir).resolve()
        if economic_outcome_dir
        else research_dir / DEFAULT_ECONOMIC_DIR_NAME
    )
    feature_path = (
        Path(research_feature_dir).resolve()
        if research_feature_dir
        else research_dir / DEFAULT_FEATURE_DIR_NAME
    )
    return {
        "oof_predictions": prediction_dir / "oof_predictions.csv",
        "economic_outcomes": outcome_path / "economic_outcomes.csv",
        "strategy_candidates": candidate_path / "strategy_candidates_v2.csv",
        "research_features": feature_path / "research_features.csv",
        "fee_exceedance_labels": research_dir / "fee_exceedance_labels_vol_scaled.csv",
        "triple_barrier_labels": research_dir / "triple_barrier_labels_vol_scaled.csv",
    }


def _dataset_status(
    name: str,
    path: Path,
    rows: Sequence[Mapping[str, str]],
    columns: Sequence[str],
) -> dict[str, Any]:
    key_columns = _available_key_columns(columns)
    return {
        "input_name": name,
        "path": str(path),
        "exists": path.exists(),
        "row_count": len(rows),
        "column_count": len(columns),
        "columns": "|".join(columns),
        "key_columns": "|".join(key_columns),
        "duplicate_key_count": (
            duplicate_key_count(rows, key_columns) if rows and key_columns else ""
        ),
        "status": "AVAILABLE" if path.exists() and rows else "MISSING_OR_EMPTY",
    }


def _available_key_columns(columns: Sequence[str]) -> tuple[str, ...]:
    if all(column in columns for column in ("fold_index", "symbol", "interval_begin")):
        return ("fold_index", "symbol", "interval_begin")
    if all(column in columns for column in ("symbol", "interval_begin")):
        return ("symbol", "interval_begin")
    return ()


def _join_readiness(
    rows_by_name: Mapping[str, Sequence[Mapping[str, str]]],
    headers_by_name: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    pairs = (
        ("oof_predictions", "economic_outcomes"),
        ("oof_predictions", "strategy_candidates"),
        ("oof_predictions", "research_features"),
        ("strategy_candidates", "economic_outcomes"),
    )
    output = []
    for left_name, right_name in pairs:
        left_rows = rows_by_name[left_name]
        right_rows = rows_by_name[right_name]
        left_columns = headers_by_name[left_name]
        right_columns = headers_by_name[right_name]
        keys = preferred_join_keys(left_columns, right_columns)
        output.append(
            {
                "left_input": left_name,
                "right_input": right_name,
                "join_keys": "|".join(keys),
                "left_rows": len(left_rows),
                "right_rows": len(right_rows),
                "left_duplicate_keys": (
                    duplicate_key_count(left_rows, keys) if keys else ""
                ),
                "right_duplicate_keys": (
                    duplicate_key_count(right_rows, keys) if keys else ""
                ),
                "matched_left_rows": (
                    matched_key_count(left_rows, right_rows, keys) if keys else 0
                ),
                "join_status": "JOIN_READY" if keys and left_rows and right_rows else "BLOCKED",
            }
        )
    return output


def _missing_inputs(datasets: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    required = ("oof_predictions", "economic_outcomes", "strategy_candidates")
    return [
        {
            "input_name": name,
            "blocker": "MISSING_REQUIRED_POLICY_INPUT",
            "path": str(datasets[name]["path"]),
        }
        for name in required
        if datasets[name]["status"] != "AVAILABLE"
    ]


def _policy_family_readiness(
    datasets: Mapping[str, Mapping[str, Any]],
    joins: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    oof_columns = set(str(datasets["oof_predictions"]["columns"]).split("|"))
    candidate_ready = _join_status(joins, "oof_predictions", "strategy_candidates")
    economic_ready = _join_status(joins, "oof_predictions", "economic_outcomes")
    rows = [
        _family("OOF_PROBABILITY_THRESHOLD", "prob_up" in oof_columns and economic_ready),
        _family("OOF_CONFIDENCE_THRESHOLD", "confidence" in oof_columns and economic_ready),
        _family("REGIME_CONDITIONAL_THRESHOLD", "regime_label" in oof_columns and economic_ready),
        _family("CANDIDATE_EVENT_POLICY", candidate_ready),
        _family(
            "CANDIDATE_PLUS_SCORE_POLICY",
            candidate_ready and "prob_up" in oof_columns and economic_ready,
        ),
        _family("BASELINE_COMPARISON", "long_trade_taken" in oof_columns and economic_ready),
        _family("CALIBRATION_DIAGNOSTICS", "prob_up" in oof_columns and "y_true" in oof_columns),
        _family("TRADING_AWARE_LABELS", False),
    ]
    return rows


def _family(name: str, ready: bool) -> dict[str, str]:
    return {
        "policy_family": name,
        "readiness_status": f"{name}_READY" if ready else f"{name}_BLOCKED",
        "ready": str(ready),
        "honesty_status": "RESEARCH_ONLY",
    }


def _join_status(
    joins: Sequence[Mapping[str, Any]],
    left: str,
    right: str,
) -> bool:
    return any(
        row["left_input"] == left
        and row["right_input"] == right
        and row["join_status"] == "JOIN_READY"
        for row in joins
    )


def _leakage_audit(datasets: Mapping[str, Mapping[str, Any]]) -> list[dict[str, str]]:
    output = []
    for dataset in datasets.values():
        columns = set(str(dataset["columns"]).split("|"))
        for column in FORBIDDEN_SELECTION_INPUTS:
            if column in columns:
                output.append(
                    {
                        "input_name": str(dataset["input_name"]),
                        "column_name": column,
                        "allowed_use": "OUTCOME_OR_DIAGNOSTIC_ONLY",
                        "selection_input_allowed": "False",
                        "leakage_status": "BLOCK_FROM_POLICY_SELECTION",
                    }
                )
    return output


def _recommendation(
    policy_ready: bool,
    missing_inputs: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    recommendation = (
        "BUILD_RESEARCH_ONLY_DECISION_POLICY_EVALUATOR"
        if policy_ready and not missing_inputs
        else "BLOCKED_MISSING_POLICY_INPUTS"
    )
    return {
        "recommendation": recommendation,
        "next_required_action": recommendation,
        "missing_input_count": len(missing_inputs),
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _next_actions(recommendation: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "action": str(recommendation["next_required_action"]),
            "rationale": "Continue only through generic research-only policy tooling.",
        }
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "policy_input_availability_report_json": str(
            output_dir / "policy_input_availability_report.json"
        ),
        "policy_input_availability_report_md": str(
            output_dir / "policy_input_availability_report.md"
        ),
        "available_inputs_csv": str(output_dir / "available_inputs.csv"),
        "join_readiness_csv": str(output_dir / "join_readiness.csv"),
        "missing_inputs_csv": str(output_dir / "missing_inputs.csv"),
        "policy_family_readiness_csv": str(output_dir / "policy_family_readiness.csv"),
        "leakage_audit_csv": str(output_dir / "leakage_audit.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(
    report: Mapping[str, Any],
    readiness: Sequence[Mapping[str, str]],
    missing_inputs: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "# M20 Policy Input Availability Audit",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Policy Family Readiness",
    ]
    lines.extend(
        f"- `{row['policy_family']}`: `{row['readiness_status']}`"
        for row in readiness
    )
    lines.extend(["", "## Missing Inputs"])
    if missing_inputs:
        lines.extend(f"- `{row['input_name']}`: `{row['blocker']}`" for row in missing_inputs)
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


__all__ = ["audit_m20_policy_inputs"]
