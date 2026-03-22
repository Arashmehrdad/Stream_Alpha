"""Artifact and index writing for M18 evaluation runs."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339
from app.evaluation.schemas import EvaluationManifest, EvaluationReport


def evaluation_root(repo_root: Path) -> Path:
    """Return the canonical M18 evaluation root."""
    return repo_root / "artifacts" / "evaluation"


def run_artifact_dir(*, repo_root: Path, evaluation_run_id: str) -> Path:
    """Return the per-run M18 artifact directory."""
    return evaluation_root(repo_root) / "m18" / evaluation_run_id


def write_evaluation_artifacts(  # pylint: disable=too-many-arguments
    *,
    repo_root: Path,
    manifest: EvaluationManifest,
    report: EvaluationReport,
    decision_opportunity_rows: list[dict[str, object]],
    performance_by_asset_rows: list[dict[str, object]],
    performance_by_regime_rows: list[dict[str, object]],
    divergence_rows: list[dict[str, object]],
    latency_rows: list[dict[str, object]],
    slippage_rows: list[dict[str, object]],
    uptime_failures_payload: dict[str, object],
    layer_comparison_payload: dict[str, object],
    paper_to_live_payload: dict[str, object],
) -> dict[str, str]:
    """Write the canonical M18 run artifact family."""
    output_dir = run_artifact_dir(
        repo_root=repo_root,
        evaluation_run_id=manifest.evaluation_run_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "evaluation_manifest": str((output_dir / "evaluation_manifest.json").resolve()),
        "evaluation_report_json": str((output_dir / "evaluation_report.json").resolve()),
        "evaluation_report_markdown": str((output_dir / "evaluation_report.md").resolve()),
        "decision_opportunities_csv": str((output_dir / "decision_opportunities.csv").resolve()),
        "performance_by_asset_csv": str((output_dir / "performance_by_asset.csv").resolve()),
        "performance_by_regime_csv": str((output_dir / "performance_by_regime.csv").resolve()),
        "divergence_events_csv": str((output_dir / "divergence_events.csv").resolve()),
        "latency_distribution_csv": str((output_dir / "latency_distribution.csv").resolve()),
        "slippage_distribution_csv": str((output_dir / "slippage_distribution.csv").resolve()),
        "uptime_and_failures_json": str((output_dir / "uptime_and_failures.json").resolve()),
        "layer_comparison_json": str((output_dir / "layer_comparison.json").resolve()),
        "paper_to_live_degradation_json": str(
            (output_dir / "paper_to_live_degradation.json").resolve()
        ),
    }
    write_json(Path(paths["evaluation_manifest"]), manifest)
    write_json(Path(paths["evaluation_report_json"]), report)
    write_text(
        Path(paths["evaluation_report_markdown"]),
        markdown_report(manifest=manifest, report=report),
    )
    write_csv(Path(paths["decision_opportunities_csv"]), decision_opportunity_rows)
    write_csv(Path(paths["performance_by_asset_csv"]), performance_by_asset_rows)
    write_csv(Path(paths["performance_by_regime_csv"]), performance_by_regime_rows)
    write_csv(Path(paths["divergence_events_csv"]), divergence_rows)
    write_csv(Path(paths["latency_distribution_csv"]), latency_rows)
    write_csv(Path(paths["slippage_distribution_csv"]), slippage_rows)
    write_json(Path(paths["uptime_and_failures_json"]), uptime_failures_payload)
    write_json(Path(paths["layer_comparison_json"]), layer_comparison_payload)
    write_json(Path(paths["paper_to_live_degradation_json"]), paper_to_live_payload)
    return paths


def append_index_entry(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSONL index entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(make_json_safe(payload), sort_keys=True))
        handle.write("\n")


def write_json(path: Path, payload: Any) -> None:
    """Write one JSON artifact deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = _to_json_safe(payload)
    path.write_text(
        json.dumps(safe_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write one CSV artifact deterministically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=_sort_key):
            writer.writerow(_normalize_csv_row(row, fieldnames))


def write_text(path: Path, payload: str) -> None:
    """Write one text artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def markdown_report(
    *,
    manifest: EvaluationManifest,
    report: EvaluationReport,
) -> str:
    """Render the canonical evaluation markdown report."""
    lines = [
        "# Stream Alpha M18 Evaluation Report",
        "",
        f"- Evaluation run: `{manifest.evaluation_run_id}`",
        f"- Service: `{manifest.service_name}`",
        f"- Window: `{manifest.window_start}` to `{manifest.window_end}`",
        f"- Modes requested: {', '.join(manifest.execution_modes_requested)}",
        f"- Modes available: {', '.join(manifest.execution_modes_available) or 'none'}",
        "",
        "## Opportunity Counts",
        "",
    ]
    for mode, count in sorted(report.opportunity_counts_by_mode.items()):
        lines.append(f"- `{mode}`: {count}")
    lines.extend(["", "## Divergence Counts", ""])
    for family, count in sorted(report.divergence_counts_by_family.items()):
        lines.append(f"- `{family}`: {count}")
    lines.extend(["", "## Cost-Aware Precision", ""])
    for mode, value in sorted(report.cost_aware_precision_by_mode.items()):
        rendered = "n/a" if value is None else f"{value:.4f}"
        lines.append(f"- `{mode}`: {rendered}")
    lines.extend(["", "## Known Limitations", ""])
    if not report.known_limitations:
        lines.append("- none")
    else:
        for item in report.known_limitations:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def _to_json_safe(payload: Any) -> Any:
    if is_dataclass(payload):
        payload = asdict(payload)
    return make_json_safe(payload)


def _fieldnames(rows: list[dict[str, object]]) -> list[str]:
    names: set[str] = set()
    for row in rows:
        names.update(str(key) for key in row.keys())
    return sorted(names)


def _sort_key(row: dict[str, object]) -> tuple[str, ...]:
    preferred = [
        row.get("execution_mode"),
        row.get("comparison_family"),
        row.get("metric_name"),
        row.get("symbol"),
        row.get("key"),
        row.get("signal_row_id"),
        row.get("event_time"),
        row.get("signal_as_of_time"),
    ]
    return tuple("" if value is None else str(value) for value in preferred)


def _normalize_csv_row(
    row: dict[str, object],
    fieldnames: list[str],
) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for fieldname in fieldnames:
        value = row.get(fieldname)
        if isinstance(value, (list, tuple)):
            normalized[fieldname] = "|".join(str(item) for item in value)
        elif hasattr(value, "isoformat"):
            normalized[fieldname] = to_rfc3339(value)
        elif isinstance(value, dict):
            normalized[fieldname] = json.dumps(make_json_safe(value), sort_keys=True)
        else:
            normalized[fieldname] = value
    return normalized
