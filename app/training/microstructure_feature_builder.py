"""Research-only microstructure feature builder from replay rows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from app.common.serialization import make_json_safe
from app.regime.artifacts import write_csv, write_json_atomic
from app.training.microstructure_replay import ReplayRow, replay_book_events
from app.training.microstructure_replay import _fixture_events as fixture_events  # pylint: disable=protected-access


DEFAULT_OUTPUT_DIR = "artifacts/research_data_upgrade/microstructure_features"
HONESTY_FLAGS = (
    "M20_PAUSED",
    "RESEARCH_ONLY",
    "FIXTURE_FEATURES_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)


@dataclass(frozen=True, slots=True)
class BuiltMicrostructureFeature:
    """One built microstructure feature row."""

    symbol: str
    event_time: str
    feature_version: str
    mid_price: float | None
    top_of_book_spread: float | None
    relative_spread: float | None
    book_gap_flag: bool

    def to_csv_row(self) -> dict[str, str]:
        """Return CSV-safe row values."""
        return {key: _csv_value(value) for key, value in asdict(self).items()}


def build_features_from_replay(
    rows: list[ReplayRow],
    *,
    feature_version: str = "microstructure_replay_v1",
) -> list[BuiltMicrostructureFeature]:
    """Build top-of-book features from replay rows."""
    built: list[BuiltMicrostructureFeature] = []
    for row in rows:
        spread = None
        mid = None
        relative = None
        if row.best_bid is not None and row.best_ask is not None:
            spread = row.best_ask - row.best_bid
            mid = (row.best_bid + row.best_ask) / 2.0
            relative = spread / mid if mid else None
        built.append(
            BuiltMicrostructureFeature(
                symbol=row.symbol,
                event_time=row.event_time,
                feature_version=feature_version,
                mid_price=mid,
                top_of_book_spread=spread,
                relative_spread=relative,
                book_gap_flag=row.book_gap_flag,
            )
        )
    return built


def write_microstructure_features(
    *,
    repo_root: Path,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Write research-only built feature artifacts."""
    root = Path(repo_root).resolve()
    resolved_output_dir = (
        root / DEFAULT_OUTPUT_DIR if output_dir is None else Path(output_dir).resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    replay_rows = replay_book_events(fixture_events())
    feature_rows = build_features_from_replay(replay_rows)
    recommendation = _recommendation()
    output_files = _output_files(resolved_output_dir)
    report = {
        "schema_version": "microstructure_features_v1",
        "repo_root": str(root),
        "feature_build_status": "MICROSTRUCTURE_FEATURE_ROWS_BUILT_FROM_FIXTURES",
        "feature_row_count": len(feature_rows),
        "input_replay_row_count": len(replay_rows),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    manifest = {
        "schema_version": "microstructure_features_manifest_v1",
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
        rows_by_file={
            "microstructure_features_csv": [row.to_csv_row() for row in feature_rows],
            "feature_lineage_csv": _lineage_rows(),
            "next_actions_csv": recommendation["next_actions"],
        },
    )
    Path(output_files["report_md"]).write_text(_markdown(report), encoding="utf-8")
    return make_json_safe(
        {**report, "manifest": manifest, "features": [asdict(row) for row in feature_rows]}
    )


def _lineage_rows() -> list[dict[str, str]]:
    return [
        {
            "feature": "top_of_book_spread",
            "inputs": "best_bid,best_ask",
            "uses_future_data": "False",
            "runtime_effect": "NO_RUNTIME_EFFECT",
        },
        {
            "feature": "relative_spread",
            "inputs": "top_of_book_spread,mid_price",
            "uses_future_data": "False",
            "runtime_effect": "NO_RUNTIME_EFFECT",
        },
    ]


def _recommendation() -> dict[str, Any]:
    return {
        "recommendation": "AUDIT_MICROSTRUCTURE_FEATURE_READINESS_FOR_ALPHA_RESEARCH",
        "next_required_action": "RUN_MICROSTRUCTURE_RESEARCH_READINESS_AUDIT",
        "next_actions": [
            {
                "action": "RUN_MICROSTRUCTURE_RESEARCH_READINESS_AUDIT",
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
        "report_json": str(output_dir / "microstructure_features.json"),
        "report_md": str(output_dir / "microstructure_features.md"),
        "microstructure_features_csv": str(output_dir / "microstructure_features.csv"),
        "feature_lineage_csv": str(output_dir / "feature_lineage.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any]) -> str:
    return (
        "# Microstructure Features\n\n"
        f"- Status: `{report['feature_build_status']}`\n"
        f"- Feature rows: `{report['feature_row_count']}`\n"
        f"- Recommendation: `{report['recommendation']}`\n"
        f"- Next required action: `{report['next_required_action']}`\n"
    )


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)
