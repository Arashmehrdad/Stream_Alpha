"""Dry-run bounded smoke harness for research microstructure capture."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic
from app.training.microstructure_capture_service import build_capture_dry_run_plan


DEFAULT_OUTPUT_DIR = "artifacts/research_data_upgrade/microstructure_capture_smoke"
HONESTY_FLAGS = (
    "M20_PAUSED",
    "RESEARCH_ONLY",
    "SMOKE_DRY_RUN_ONLY",
    "NO_NETWORK_CAPTURE",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)


def write_microstructure_capture_smoke(  # pylint: disable=too-many-arguments
    *,
    repo_root: Path,
    output_dir: Path | None = None,
    symbols: tuple[str, ...] = ("BTC/USD",),
    depth: int = 10,
    duration_seconds: int = 30,
    max_events: int = 500,
    execute: bool = False,
) -> dict[str, Any]:  # pylint: disable=too-many-arguments
    """Write a dry-run bounded capture smoke artifact."""
    if execute:
        raise ValueError("Real capture smoke is blocked until separately approved")
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    plan = build_capture_dry_run_plan(
        symbols=symbols,
        depth=depth,
        duration_seconds=duration_seconds,
        max_events=max_events,
    )
    rows = _rows(plan)
    recommendation = _recommendation()
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "microstructure_capture_smoke_v1",
        "repo_root": str(root),
        "smoke_status": "BOUNDED_CAPTURE_SMOKE_DRY_RUN_READY",
        "smoke_executed": False,
        "network_capture_executed": False,
        "database_writes_executed": False,
        "symbols": list(plan.symbols),
        "depth": plan.depth,
        "duration_seconds": plan.duration_seconds,
        "max_events": plan.max_events,
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    manifest = {
        "schema_version": "microstructure_capture_smoke_manifest_v1",
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


def _rows(plan: Any) -> dict[str, list[dict[str, str]]]:
    return {
        "smoke_plan_csv": [
            {
                "symbols": ",".join(plan.symbols),
                "depth": str(plan.depth),
                "duration_seconds": str(plan.duration_seconds),
                "max_events": str(plan.max_events),
                "execute": "False",
                "runtime_effect": "NO_RUNTIME_EFFECT",
            }
        ],
        "operator_preflight_csv": [
            _preflight("schema_contracts_present", "operator must verify before real smoke"),
            _preflight("disk_budget_selected", "operator must set disk budget"),
            _preflight("stop_condition_selected", "duration and max events are mandatory"),
        ],
        "blocked_actions_csv": [
            {
                "action": "execute_smoke",
                "blocker": "SEPARATE_APPROVAL_REQUIRED",
                "required_action": "APPROVE_BOUNDED_PUBLIC_WS_CAPTURE",
            }
        ],
    }


def _preflight(name: str, detail: str) -> dict[str, str]:
    return {"check": name, "detail": detail, "status": "DRY_RUN_ONLY"}


def _recommendation() -> dict[str, Any]:
    return {
        "recommendation": "BUILD_MICROSTRUCTURE_REPLAY_AND_COVERAGE_FROM_FIXTURES",
        "next_required_action": "IMPLEMENT_MICROSTRUCTURE_REPLAY_GAP_ENGINE",
        "next_actions": [
            {
                "action": "IMPLEMENT_MICROSTRUCTURE_REPLAY_GAP_ENGINE",
                "runtime_effect": "NO_RUNTIME_EFFECT",
            }
        ],
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
        "report_json": str(output_dir / "microstructure_capture_smoke.json"),
        "report_md": str(output_dir / "microstructure_capture_smoke.md"),
        "smoke_plan_csv": str(output_dir / "smoke_plan.csv"),
        "operator_preflight_csv": str(output_dir / "operator_preflight.csv"),
        "blocked_actions_csv": str(output_dir / "blocked_actions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    return (
        "# Microstructure Capture Smoke Dry Run\n\n"
        f"- Status: `{report['smoke_status']}`\n"
        f"- Smoke executed: `{report['smoke_executed']}`\n"
        f"- Recommendation: `{report['recommendation']}`\n"
        f"- Next required action: `{report['next_required_action']}`\n"
    )
