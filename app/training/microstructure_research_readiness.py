"""Research readiness audit for microstructure data-upgrade artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic


DEFAULT_OUTPUT_DIR = "artifacts/research_data_upgrade/microstructure_research_readiness"
HONESTY_FLAGS = (
    "M20_PAUSED",
    "RESEARCH_ONLY",
    "READINESS_AUDIT_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)


def write_microstructure_research_readiness(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Write DU12 readiness audit over microstructure artifact chain."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows(root)
    readiness = _readiness(rows["artifact_readiness_csv"])
    recommendation = _recommendation(readiness)
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "microstructure_research_readiness_v1",
        "repo_root": str(root),
        "readiness_status": readiness,
        "alpha_research_reopen_ready": False,
        "artifact_count": len(rows["artifact_readiness_csv"]),
        "ready_artifact_count": sum(
            1 for row in rows["artifact_readiness_csv"] if row["status"] == "PRESENT"
        ),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    manifest = {
        "schema_version": "microstructure_research_readiness_manifest_v1",
        "repo_root": str(root),
        "output_dir": str(resolved_output_dir),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    _write_outputs(
        output_files=output_files,
        manifest=manifest,
        report=report,
        recommendation=recommendation,
        rows_by_file={**rows, "next_actions_csv": recommendation["next_actions"]},
    )
    Path(output_files["report_md"]).write_text(_markdown(report), encoding="utf-8")
    return make_json_safe(
        {**report, "manifest": manifest, "recommendation_payload": recommendation}
    )


def _rows(root: Path) -> dict[str, list[dict[str, str]]]:
    artifact_rows = _artifact_rows(root)
    return {
        "artifact_readiness_csv": artifact_rows,
        "coverage_readiness_csv": _coverage_rows(),
        "blockers_csv": _blocker_rows(),
    }


def _artifact_rows(root: Path) -> list[dict[str, str]]:
    artifacts = [
        (
            "storage_contracts",
            "microstructure_storage_contracts/microstructure_storage_contracts.json",
        ),
        (
            "capture_dry_run",
            "microstructure_capture_service_dry_run/"
            "microstructure_capture_service_dry_run.json",
        ),
        ("capture_smoke", "microstructure_capture_smoke/microstructure_capture_smoke.json"),
        ("replay", "microstructure_replay/microstructure_replay.json"),
        ("features", "microstructure_features/microstructure_features.json"),
    ]
    return [_artifact(root, name, relative) for name, relative in artifacts]


def _artifact(root: Path, name: str, relative: str) -> dict[str, str]:
    path = root / "artifacts/research_data_upgrade" / relative
    status = "PRESENT" if path.is_file() else "MISSING"
    schema_version = ""
    if path.is_file():
        schema_version = str(json.loads(path.read_text(encoding="utf-8")).get("schema_version", ""))
    return {
        "artifact": name,
        "path": str(path.relative_to(root)),
        "status": status,
        "schema_version": schema_version,
    }


def _coverage_rows() -> list[dict[str, str]]:
    return [
        {
            "check": "feature_rows",
            "status": "INSUFFICIENT_FIXTURE_ONLY",
            "observed": "2",
            "required": "operator-selected stored replay dataset",
        },
        {
            "check": "time_coverage",
            "status": "INSUFFICIENT_FIXTURE_ONLY",
            "observed": "static fixture timestamps",
            "required": "bounded multi-symbol capture window",
        },
        {
            "check": "untouched_segment",
            "status": "NOT_AVAILABLE",
            "observed": "none",
            "required": "stored data split after capture",
        },
    ]


def _blocker_rows() -> list[dict[str, str]]:
    return [
        {
            "blocker": "INSUFFICIENT_MICROSTRUCTURE_COVERAGE",
            "required_action": "COLLECT_BOUNDED_RESEARCH_MICROSTRUCTURE_DATA",
        },
        {
            "blocker": "NO_UNTOUCHED_EVALUATION_SEGMENT",
            "required_action": "CREATE_TIME_SPLIT_AFTER_CAPTURE",
        },
    ]


def _readiness(artifact_rows: list[Mapping[str, str]]) -> str:
    if not all(row["status"] == "PRESENT" for row in artifact_rows):
        return "MICROSTRUCTURE_RESEARCH_READINESS_BLOCKED_MISSING_ARTIFACTS"
    return "MICROSTRUCTURE_RESEARCH_NOT_READY_FIXTURE_ONLY"


def _recommendation(readiness: str) -> dict[str, Any]:
    if readiness.endswith("MISSING_ARTIFACTS"):
        recommendation = "RESTORE_MICROSTRUCTURE_ARTIFACT_CHAIN"
        next_action = "RE_RUN_DU7_DU11_ARTIFACTS"
    else:
        recommendation = "COLLECT_MORE_MICROSTRUCTURE_DATA_BEFORE_REOPENING_ALPHA_RESEARCH"
        next_action = "APPROVE_BOUNDED_RESEARCH_CAPTURE_OR_PAUSE_DATA_UPGRADE"
    return {
        "recommendation": recommendation,
        "next_required_action": next_action,
        "next_actions": [{"action": next_action, "runtime_effect": "NO_RUNTIME_EFFECT"}],
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
    }


def _write_outputs(
    *,
    output_files: Mapping[str, str],
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    recommendation: Mapping[str, Any],
    rows_by_file: Mapping[str, list[Mapping[str, Any]]],
) -> None:
    write_json_atomic(Path(output_files["manifest_json"]), dict(manifest))
    write_json_atomic(Path(output_files["report_json"]), dict(report))
    write_json_atomic(Path(output_files["recommendation_json"]), dict(recommendation))
    for key, rows in rows_by_file.items():
        write_csv(Path(output_files[key]), [dict(row) for row in rows])


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "microstructure_research_readiness.json"),
        "report_md": str(output_dir / "microstructure_research_readiness.md"),
        "artifact_readiness_csv": str(output_dir / "artifact_readiness.csv"),
        "coverage_readiness_csv": str(output_dir / "coverage_readiness.csv"),
        "blockers_csv": str(output_dir / "blockers.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    return (
        "# Microstructure Research Readiness\n\n"
        f"- Status: `{report['readiness_status']}`\n"
        f"- Alpha research reopen ready: `{report['alpha_research_reopen_ready']}`\n"
        f"- Recommendation: `{report['recommendation']}`\n"
        f"- Next required action: `{report['next_required_action']}`\n"
    )
