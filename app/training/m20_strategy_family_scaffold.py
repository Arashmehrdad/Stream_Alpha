"""Research-only M20 strategy-family scaffold."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "strategy_family_scaffold"
FAMILIES = (
    "momentum_breakout",
    "range_mean_reversion",
    "volatility_expansion",
    "abstention_hold",
)
HONESTY_FLAGS = (
    "RESEARCH_ONLY_STRATEGY_FAMILY_SCAFFOLD",
    "DESIGN_ONLY",
    "NO_RUNTIME_EFFECT",
    "NO_REGISTRY_WRITE",
    "NO_PROMOTION_EFFECT",
    "NOT_BACKTEST",
    "NO_TRADING_LOGIC",
    "NO_PROFIT_CLAIM",
    "RANK_GATE_OPTIONAL_FILTER_ONLY",
)


def design_m20_strategy_families(*, base_run_dir: Path) -> dict[str, Any]:
    """Write research-only M20 strategy-family scaffold artifacts."""
    base_dir = Path(base_run_dir).resolve()
    vol_dir = base_dir / "research_labels" / "vol_scaled"
    output_dir = vol_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = _read_json(vol_dir / "m20_decision_memo" / "decision_memo.json")
    feature_columns = _read_feature_columns(base_dir)
    strategy_families = _strategy_families(decision)
    required_signals = _required_signals()
    feature_requirements = _feature_requirements(feature_columns)
    next_experiments = _next_experiments()
    output_files = _output_files(output_dir)
    report = {
        "scaffold_status": "RESEARCH_ONLY_DESIGN_READY",
        "decision_source": decision.get("decision", "UNKNOWN"),
        "family_count": len(strategy_families),
        "rank_gate_usage": "OPTIONAL_FILTER_ONLY",
        "runtime_status": "NOT_RUNTIME_READY",
        "promotion_status": "NOT_PROMOTABLE",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    manifest = {
        "base_run_dir": str(base_dir),
        "output_dir": str(output_dir),
        "source_decision_memo": str(vol_dir / "m20_decision_memo"),
        "feature_schema_path": str(
            base_dir / "training_frame" / "m20_training_frame_feature_columns.json"
        ),
        "families": list(FAMILIES),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["report_json"]), report)
    Path(output_files["report_md"]).write_text(
        _markdown(report, strategy_families, next_experiments),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["strategy_families_csv"]), strategy_families)
    write_csv_artifact(Path(output_files["required_signals_csv"]), required_signals)
    write_csv_artifact(Path(output_files["feature_requirements_csv"]), feature_requirements)
    write_csv_artifact(Path(output_files["next_experiments_csv"]), next_experiments)
    return make_json_safe(report)


def _strategy_families(decision: Mapping[str, Any]) -> list[dict[str, str]]:
    base_note = (
        "Rank gate may be used only as an optional opportunity filter; "
        "it must not decide LONG/SHORT or runtime execution."
    )
    decision_statuses = ",".join(decision.get("decision_statuses", []))
    return [
        {
            "family_id": "momentum_breakout",
            "research_role": "test directional continuation after confirmed movement potential",
            "rank_gate_role": "optional prefilter",
            "initial_status": "DESIGN_ONLY",
            "decision_context": decision_statuses,
            "notes": base_note,
        },
        {
            "family_id": "range_mean_reversion",
            "research_role": "test bounded reversal or range-bounce candidates separately",
            "rank_gate_role": "optional skip filter during low movement potential",
            "initial_status": "DESIGN_ONLY",
            "decision_context": decision_statuses,
            "notes": base_note,
        },
        {
            "family_id": "volatility_expansion",
            "research_role": (
                "test whether expanding range/volatility conditions need "
                "specialist logic"
            ),
            "rank_gate_role": "optional movement filter only",
            "initial_status": "DESIGN_ONLY",
            "decision_context": decision_statuses,
            "notes": base_note,
        },
        {
            "family_id": "abstention_hold",
            "research_role": "formalize HOLD/skip behavior for weak or unstable evidence",
            "rank_gate_role": "optional abstention input",
            "initial_status": "DESIGN_ONLY",
            "decision_context": decision_statuses,
            "notes": base_note,
        },
    ]


def _required_signals() -> list[dict[str, str]]:
    return [
        {
            "signal_id": "fee_exceedance_probability",
            "source": "logistic_regression_tiny research baseline",
            "required_for": "optional rank gate",
            "runtime_ready": "false",
        },
        {
            "signal_id": "directional_entry_signal",
            "source": "future strategy-family research",
            "required_for": "momentum_breakout,range_mean_reversion",
            "runtime_ready": "false",
        },
        {
            "signal_id": "range_or_trend_context",
            "source": "feature buckets or future regime research",
            "required_for": "all strategy families",
            "runtime_ready": "false",
        },
        {
            "signal_id": "abstention_reason",
            "source": "research-only decision rules",
            "required_for": "abstention_hold",
            "runtime_ready": "false",
        },
    ]


def _feature_requirements(feature_columns: Sequence[str]) -> list[dict[str, str]]:
    required = {
        "momentum_breakout": ("log_return_1", "macd_line_12_26", "high_price", "close_price"),
        "range_mean_reversion": ("rsi_14", "high_price", "low_price", "close_price"),
        "volatility_expansion": ("realized_vol_12", "high_price", "low_price", "volume"),
        "abstention_hold": ("realized_vol_12", "volume", "log_return_1"),
    }
    available = set(feature_columns)
    rows = []
    for family_id, columns in required.items():
        for column in columns:
            rows.append(
                {
                    "family_id": family_id,
                    "feature_name": column,
                    "available_in_training_frame": str(column in available).lower(),
                    "requirement_status": (
                        "AVAILABLE_FOR_RESEARCH"
                        if column in available else "FEATURE_EXPORT_GAP"
                    ),
                }
            )
    return rows


def _next_experiments() -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "experiment": "paper design only for momentum_breakout labels/signals",
            "allowed_now": "true",
            "blocked_actions": "runtime,promotion,trading,backtest",
        },
        {
            "priority": "2",
            "experiment": "paper design only for range_mean_reversion labels/signals",
            "allowed_now": "true",
            "blocked_actions": "runtime,promotion,trading,backtest",
        },
        {
            "priority": "3",
            "experiment": "AutoGluon member prediction export plan",
            "allowed_now": "manual/light planning only",
            "blocked_actions": "long refit,registry,promotion",
        },
        {
            "priority": "4",
            "experiment": "alternate fee-exceedance horizon/label design",
            "allowed_now": "design only",
            "blocked_actions": "model retrain without separate approval",
        },
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "report_json": str(output_dir / "report.json"),
        "report_md": str(output_dir / "report.md"),
        "strategy_families_csv": str(output_dir / "strategy_families.csv"),
        "required_signals_csv": str(output_dir / "required_signals.csv"),
        "feature_requirements_csv": str(output_dir / "feature_requirements.csv"),
        "next_experiments_csv": str(output_dir / "next_experiments.csv"),
    }


def _markdown(
    report: Mapping[str, Any],
    families: Sequence[Mapping[str, str]],
    experiments: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "# M20 Strategy-Family Research Scaffold",
        "",
        f"- Scaffold status: `{report['scaffold_status']}`",
        f"- Rank gate usage: `{report['rank_gate_usage']}`",
        f"- Runtime status: `{report['runtime_status']}`",
        f"- Promotion status: `{report['promotion_status']}`",
        f"- Honesty flags: `{', '.join(report['honesty_flags'])}`",
        "",
        "This is design-only research scaffolding. It does not implement trading logic.",
        "",
        "## Families",
        "",
    ]
    for row in families:
        lines.append(f"- `{row['family_id']}`: {row['research_role']}")
    lines.extend(["", "## Next Experiments", ""])
    for row in experiments:
        lines.append(f"- {row['priority']}. {row['experiment']}")
    lines.append("")
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_feature_columns(base_dir: Path) -> list[str]:
    path = base_dir / "training_frame" / "m20_training_frame_feature_columns.json"
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload]
    if isinstance(payload, dict):
        values = payload.get("feature_columns", payload.get("columns", []))
        return [str(item) for item in values] if isinstance(values, list) else []
    return []


def read_scaffold_csv(path: Path) -> list[dict[str, str]]:
    """Read scaffold CSV output for tests."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
