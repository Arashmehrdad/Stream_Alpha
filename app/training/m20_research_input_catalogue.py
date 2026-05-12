"""Research-only catalogue of M20 input availability and blocked labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_policy_research_common import (
    FORBIDDEN_SELECTION_INPUTS,
    HONESTY_FLAGS,
    read_csv_header,
    read_csv_rows,
    vol_scaled_dir,
    write_csv_artifact,
    write_json_artifact,
)


DEFAULT_OUTPUT_NAME = "m20_research_input_catalogue"


def build_m20_research_input_catalogue(
    *,
    source_run_dir: Path,
    prediction_run_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Catalogue available and missing M20 research inputs."""
    source_dir = Path(source_run_dir).resolve()
    research_dir = vol_scaled_dir(source_dir)
    prediction_dir = Path(prediction_run_dir).resolve() if prediction_run_dir else None
    output_dir = research_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = _datasets(research_dir, prediction_dir)
    catalogue = [_catalogue_row(name, path) for name, path in datasets.items()]
    blocked_label_audit = _blocked_label_audit(datasets["research_features"])
    missing_columns = _missing_columns(catalogue)
    recommendation = _recommendation(blocked_label_audit)
    output_files = _output_files(output_dir)
    report = {
        "summary": "M20 research input catalogue and blocked label audit.",
        "dataset_count": len(catalogue),
        "safe_computable_blocked_labels": sum(
            1
            for row in blocked_label_audit
            if row["safe_computability"] == "SAFE_COMPUTABLE_RESEARCH_LABEL"
        ),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "overall_status": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "output_files": output_files,
    }
    manifest = {
        "source_run_dir": str(source_dir),
        "prediction_run_dir": str(prediction_dir) if prediction_dir else "",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["catalogue_json"]), report)
    Path(output_files["catalogue_md"]).write_text(_markdown(report, blocked_label_audit), "utf-8")
    write_csv_artifact(Path(output_files["input_catalogue_csv"]), catalogue)
    write_csv_artifact(Path(output_files["blocked_label_audit_csv"]), blocked_label_audit)
    write_csv_artifact(Path(output_files["missing_columns_csv"]), missing_columns)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "input_catalogue": catalogue,
            "blocked_label_audit": blocked_label_audit,
            "missing_columns": missing_columns,
            "recommendation_payload": recommendation,
        }
    )


def _datasets(research_dir: Path, prediction_dir: Path | None) -> dict[str, Path]:
    datasets = {
        "research_features": (
            research_dir / "research_feature_enrichment" / "research_features.csv"
        ),
        "economic_outcomes": research_dir / "economic_outcome_artifacts" / "economic_outcomes.csv",
        "refined_candidates": (
            research_dir / "strategy_candidate_v2_refined_factory" / "strategy_candidates_v2.csv"
        ),
        "trading_aware_labels": research_dir / "trading_aware_labels" / "trading_aware_labels.csv",
        "policy_metrics": research_dir / "trading_aware_policy_eval" / "policy_metrics.csv",
    }
    if prediction_dir is not None:
        datasets["oof_predictions"] = prediction_dir / "oof_predictions.csv"
    return datasets


def _catalogue_row(name: str, path: Path) -> dict[str, Any]:
    rows = read_csv_rows(path)
    columns = read_csv_header(path)
    return {
        "input_name": name,
        "path": str(path),
        "exists": path.exists(),
        "row_count": len(rows),
        "column_count": len(columns),
        "columns": "|".join(columns),
        "timestamp_columns": "|".join(
            column for column in columns
            if "time" in column or "interval_begin" == column
        ),
        "symbol_columns": "|".join(column for column in columns if column == "symbol"),
        "leakage_risk": _leakage_risk(columns),
        "safe_computability": _safe_computability(name, columns),
    }


def _blocked_label_audit(feature_path: Path) -> list[dict[str, str]]:
    columns = set(read_csv_header(feature_path))
    safe = {"symbol", "interval_begin", "close_price"}.issubset(columns)
    status = "SAFE_COMPUTABLE_RESEARCH_LABEL" if safe else "BLOCKED_MISSING_CLOSE_PRICE"
    missing = "" if safe else "close_price"
    return [
        {
            "label_name": "forward_return_6_candles",
            "safe_computability": status,
            "required_columns": "symbol|interval_begin|close_price",
            "missing_columns": missing,
        },
        {
            "label_name": "forward_return_12_candles",
            "safe_computability": status,
            "required_columns": "symbol|interval_begin|close_price",
            "missing_columns": missing,
        },
    ]


def _missing_columns(catalogue: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    required = {
        "research_features": ("close_price", "high_price", "low_price", "realized_vol_12"),
        "economic_outcomes": ("fee_bps", "slippage_bps"),
        "oof_predictions": ("prob_up", "confidence"),
    }
    output = []
    for row in catalogue:
        needed = required.get(str(row["input_name"]), ())
        columns = set(str(row["columns"]).split("|"))
        for column in needed:
            if column not in columns:
                output.append({"input_name": str(row["input_name"]), "missing_column": column})
    return output


def _leakage_risk(columns: Sequence[str]) -> str:
    if any(column in FORBIDDEN_SELECTION_INPUTS for column in columns):
        return "OUTCOME_COLUMNS_PRESENT_DIAGNOSTIC_ONLY"
    return "SELECTION_SAFE_OR_METADATA"


def _safe_computability(name: str, columns: Sequence[str]) -> str:
    column_set = set(columns)
    if (
        name == "research_features"
        and {"close_price", "high_price", "low_price"}.issubset(column_set)
    ):
        return "CAN_COMPUTE_MULTI_HORIZON_RESEARCH_LABELS"
    return "AVAILABLE_AS_EXISTING_ARTIFACT" if columns else "MISSING_OR_EMPTY"


def _recommendation(blocked_label_audit: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    ready = any(
        row["safe_computability"] == "SAFE_COMPUTABLE_RESEARCH_LABEL"
        for row in blocked_label_audit
    )
    recommendation = (
        "BUILD_SAFE_INPUT_REDESIGN_PLAN"
        if ready
        else "M20_BLOCKED_MISSING_SAFE_INPUTS"
    )
    return {
        "recommendation": recommendation,
        "next_required_action": recommendation,
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _next_actions(recommendation: Mapping[str, Any]) -> list[dict[str, str]]:
    return [{"priority": "1", "action": str(recommendation["next_required_action"])}]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "catalogue_json": str(output_dir / "m20_research_input_catalogue.json"),
        "catalogue_md": str(output_dir / "m20_research_input_catalogue.md"),
        "input_catalogue_csv": str(output_dir / "input_catalogue.csv"),
        "blocked_label_audit_csv": str(output_dir / "blocked_label_audit.csv"),
        "missing_columns_csv": str(output_dir / "missing_columns.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any], blocked: Sequence[Mapping[str, str]]) -> str:
    lines = [
        "# M20 Research Input Catalogue",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Dataset count: `{report['dataset_count']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Blocked Label Audit",
    ]
    lines.extend(f"- `{row['label_name']}`: `{row['safe_computability']}`" for row in blocked)
    return "\n".join(lines) + "\n"


__all__ = ["build_m20_research_input_catalogue"]
