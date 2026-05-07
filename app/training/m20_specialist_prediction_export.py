"""Research-only export of existing M20 specialist row-level predictions."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "specialist_predictions"
SPECIALIST_MODELS = ("neuralforecast_nhits", "neuralforecast_patchtst")
SAFE_OUTPUT_COLUMNS = (
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
FORBIDDEN_TOKENS = (
    "future",
    "forward",
    "gross",
    "net",
    "return",
    "barrier",
    "label",
    "target",
    "outcome",
)
HONESTY_FLAGS = (
    "RESEARCH_ONLY_SPECIALIST_PREDICTION_EXPORT",
    "EXISTING_OOF_ONLY",
    "NO_SCORE_ONLY_RERUN_EXECUTED",
    "NO_MODEL_RETRAIN",
    "NO_RUNTIME_EFFECT",
    "NO_REGISTRY_WRITE",
    "NO_PROMOTION_EFFECT",
    "NOT_BACKTEST",
    "NO_PROFIT_CLAIM",
    "NOT_PROMOTABLE",
    "LEAKAGE_COLUMNS_QUARANTINED",
)


def export_existing_m20_specialist_predictions(
    *,
    base_run_dir: Path,
    previous_run_dir: Path,
    prediction_source: str = "oof_20260427",
) -> dict[str, Any]:
    """Sanitize existing OOF specialist predictions into per-model research files."""
    # pylint: disable=too-many-locals
    base_dir = Path(base_run_dir).resolve()
    previous_dir = Path(previous_run_dir).resolve()
    oof_path = previous_dir / "oof_predictions.csv"
    output_dir = (
        base_dir / "research_labels" / "vol_scaled" / OUTPUT_DIR_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_csv(oof_path)
    source_columns = list(rows[0]) if rows else []
    forbidden_columns = _forbidden_columns(source_columns)
    safe_rows_by_model = _sanitize_rows(rows, prediction_source=prediction_source)
    output_files = _output_files(output_dir)
    per_model_outputs = _write_model_prediction_files(output_dir, safe_rows_by_model)
    audit_rows = _audit_rows(
        source_columns=source_columns,
        forbidden_columns=forbidden_columns,
        safe_rows_by_model=safe_rows_by_model,
    )
    manifest = {
        "base_run_dir": str(base_dir),
        "previous_run_dir": str(previous_dir),
        "source_oof_predictions": str(oof_path),
        "output_dir": str(output_dir),
        "prediction_source": prediction_source,
        "specialist_models": list(SPECIALIST_MODELS),
        "source_row_count": len(rows),
        "exported_row_count": sum(len(model_rows) for model_rows in safe_rows_by_model.values()),
        "forbidden_columns_quarantined": forbidden_columns,
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "output_files": output_files | per_model_outputs,
    }
    report = {
        "summary": (
            "Existing OOF NHITS/PatchTST rows were sanitized into research-only "
            "per-specialist prediction files. Future/net proxy fields were not "
            "included in prediction outputs."
        ),
        "model_row_counts": {
            model_name: len(model_rows)
            for model_name, model_rows in sorted(safe_rows_by_model.items())
        },
        "forbidden_columns_quarantined": forbidden_columns,
        "next_research_action": "RUN_SPECIALIST_CONDITIONAL_ANALYSIS_ON_SANITIZED_OOF",
        "honesty_flags": list(HONESTY_FLAGS),
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    write_csv_artifact(Path(output_files["schema_audit_csv"]), audit_rows)
    Path(output_files["report_md"]).write_text(
        _markdown(report, manifest, per_model_outputs),
        encoding="utf-8",
    )
    return make_json_safe(manifest)


def export_m20_specialist_prediction_records(
    *,
    run_dir: Path,
    prediction_rows: Sequence[Mapping[str, Any]],
    prediction_source: str,
    confirmation_window: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Export in-memory score-only specialist predictions as sanitized artifacts."""
    resolved_run_dir = Path(run_dir).resolve()
    output_dir = resolved_run_dir / "research_labels" / "vol_scaled" / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = [dict(row) for row in prediction_rows]
    forbidden_columns = _forbidden_columns(list(source_rows[0]) if source_rows else [])
    safe_rows_by_model = _sanitize_rows(
        source_rows,
        prediction_source=prediction_source,
        confirmation_window=confirmation_window,
    )
    output_files = _output_files(output_dir)
    per_model_outputs = _write_model_prediction_files(
        output_dir,
        safe_rows_by_model,
        prediction_source=prediction_source,
    )
    audit_rows = _audit_rows(
        source_columns=list(source_rows[0]) if source_rows else list(SAFE_OUTPUT_COLUMNS),
        forbidden_columns=forbidden_columns,
        safe_rows_by_model=safe_rows_by_model,
    )
    exported_row_count = sum(len(rows) for rows in safe_rows_by_model.values())
    manifest = {
        "run_dir": str(resolved_run_dir),
        "output_dir": str(output_dir),
        "prediction_source": prediction_source,
        "source": "score_only_prediction_records",
        "source_row_count": len(source_rows),
        "exported_row_count": exported_row_count,
        "specialist_models": list(SPECIALIST_MODELS),
        "forbidden_columns_quarantined": forbidden_columns,
        "confirmation_window": dict(confirmation_window or {}),
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "output_files": output_files | per_model_outputs,
    }
    report = {
        "summary": (
            "Score-only NHITS/PatchTST rows were sanitized into research-only "
            "per-specialist prediction files."
        ),
        "model_row_counts": {
            model_name: len(model_rows)
            for model_name, model_rows in sorted(safe_rows_by_model.items())
        },
        "forbidden_columns_quarantined": forbidden_columns,
        "next_research_action": "RUN_SPECIALIST_CONDITIONAL_ANALYSIS_ON_CONFIRMATION_EXPORT",
        "honesty_flags": list(HONESTY_FLAGS),
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    write_csv_artifact(Path(output_files["schema_audit_csv"]), audit_rows)
    Path(output_files["report_md"]).write_text(
        _markdown(report, manifest, per_model_outputs),
        encoding="utf-8",
    )
    return make_json_safe(manifest)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing OOF predictions file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _forbidden_columns(columns: Sequence[str]) -> list[str]:
    return [
        column
        for column in columns
        if column not in SAFE_OUTPUT_COLUMNS
        and any(token in column.lower() for token in FORBIDDEN_TOKENS)
    ]


def _sanitize_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    prediction_source: str,
    confirmation_window: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, str]]]:
    by_model: dict[str, list[dict[str, str]]] = {
        model_name: [] for model_name in SPECIALIST_MODELS
    }
    for row in rows:
        model_name = row.get("model_name", "")
        if model_name not in by_model:
            continue
        safe_row = {
            column: row.get(column, "")
            for column in SAFE_OUTPUT_COLUMNS
            if column not in {"candidate_id", "prediction_source"}
        }
        candidate_prefix = str(
            (confirmation_window or {}).get("candidate_run_id", "20260427T112021Z")
        )
        safe_row["candidate_id"] = f"{candidate_prefix}:{model_name}"
        safe_row["prediction_source"] = prediction_source
        safe_row["confirmation_window_start"] = str(
            (confirmation_window or {}).get("confirmation_window_start", "")
        )
        safe_row["confirmation_window_end"] = str(
            (confirmation_window or {}).get("confirmation_window_end", "")
        )
        safe_row["confirmation_tag"] = str(
            (confirmation_window or {}).get("confirmation_tag", "")
        )
        by_model[model_name].append(
            {column: safe_row.get(column, "") for column in SAFE_OUTPUT_COLUMNS}
        )
    return by_model


def _write_model_prediction_files(
    output_dir: Path,
    rows_by_model: Mapping[str, Sequence[Mapping[str, str]]],
    *,
    prediction_source: str = "oof",
) -> dict[str, str]:
    output_files = {}
    source_label = _safe_source_label(prediction_source)
    for model_name, rows in sorted(rows_by_model.items()):
        path = output_dir / f"predictions_{model_name}_{source_label}.csv"
        _write_prediction_csv(path, [dict(row) for row in rows])
        output_files[f"predictions_{model_name}_{source_label}_csv"] = str(path)
    return output_files


def _safe_source_label(prediction_source: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in prediction_source.lower()
    ).strip("_")
    return cleaned or "predictions"


def _write_prediction_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAFE_OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _audit_rows(
    *,
    source_columns: Sequence[str],
    forbidden_columns: Sequence[str],
    safe_rows_by_model: Mapping[str, Sequence[Mapping[str, str]]],
) -> list[dict[str, Any]]:
    safe_set = set(SAFE_OUTPUT_COLUMNS)
    forbidden_set = set(forbidden_columns)
    audit = [
        {
            "column": column,
            "source_present": True,
            "exported": column in safe_set,
            "reason": _column_reason(column, safe_set, forbidden_set),
        }
        for column in source_columns
    ]
    for model_name, rows in sorted(safe_rows_by_model.items()):
        audit.append(
            {
                "column": f"rows:{model_name}",
                "source_present": True,
                "exported": True,
                "reason": f"exported_row_count={len(rows)}",
            }
        )
    return audit


def _column_reason(column: str, safe_set: set[str], forbidden_set: set[str]) -> str:
    if column in safe_set:
        return "safe_prediction_or_key_column"
    if column in forbidden_set:
        return "quarantined_future_or_target_derived_column"
    return "not_required_for_specialist_prediction_export"


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "report.json"),
        "report_md": str(output_dir / "report.md"),
        "schema_audit_csv": str(output_dir / "schema_audit.csv"),
    }


def _markdown(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    per_model_outputs: Mapping[str, str],
) -> str:
    lines = [
        "# M20 Specialist Prediction Export",
        "",
        "- Status: `RESEARCH_ONLY_SPECIALIST_PREDICTION_EXPORT`",
        "- Source: existing OOF predictions only",
        "- Runtime status: `NO_RUNTIME_EFFECT`",
        "- Promotion status: `NOT_PROMOTABLE`",
        "",
        "## Model Row Counts",
    ]
    for model_name, row_count in report["model_row_counts"].items():
        lines.append(f"- `{model_name}`: {row_count}")
    lines.extend(["", "## Quarantined Columns"])
    for column in manifest["forbidden_columns_quarantined"]:
        lines.append(f"- `{column}`")
    lines.extend(["", "## Prediction Files"])
    for name, path in sorted(per_model_outputs.items()):
        lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "No score-only rerun, model retrain, runtime path, registry write, "
            "promotion, backtest, or profit claim was added.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "export_existing_m20_specialist_predictions",
    "export_m20_specialist_prediction_records",
]
