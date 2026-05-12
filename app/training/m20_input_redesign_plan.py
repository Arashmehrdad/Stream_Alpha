"""Research-only M20 input redesign planner."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.m20_policy_research_common import (
    HONESTY_FLAGS,
    read_csv_rows,
    vol_scaled_dir,
    write_csv_artifact,
    write_json_artifact,
)


DEFAULT_OUTPUT_NAME = "m20_input_redesign_plan"


def plan_m20_input_redesign(
    *,
    source_run_dir: Path,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Plan reusable safe input upgrades for M20."""
    source_dir = Path(source_run_dir).resolve()
    research_dir = vol_scaled_dir(source_dir)
    output_dir = research_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    blocked = read_csv_rows(
        research_dir / "m20_research_input_catalogue" / "blocked_label_audit.csv"
    )
    specs = _specs(blocked)
    blockers = [row for row in specs if row["spec_status"] != "READY"]
    recommendation = _recommendation(specs)
    output_files = _output_files(output_dir)
    report = {
        "summary": "Research-only M20 input redesign plan.",
        "spec_count": len(specs),
        "ready_spec_count": len([row for row in specs if row["spec_status"] == "READY"]),
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
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["plan_json"]), report)
    Path(output_files["plan_md"]).write_text(_markdown(report, specs), "utf-8")
    write_csv_artifact(Path(output_files["input_upgrade_specs_csv"]), specs)
    write_csv_artifact(Path(output_files["blocked_specs_csv"]), blockers)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "input_upgrade_specs": specs,
            "blocked_specs": blockers,
            "recommendation_payload": recommendation,
        }
    )


def _specs(blocked: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    safe_multi = any(
        row.get("safe_computability") == "SAFE_COMPUTABLE_RESEARCH_LABEL"
        for row in blocked
    )
    ready = "READY" if safe_multi else "BLOCKED"
    return [
        {
            "spec_name": "multi_horizon_forward_returns",
            "spec_family": "RICHER_MULTI_HORIZON_LABELS",
            "horizons": "6|12",
            "required_inputs": "symbol|interval_begin|close_price",
            "forbidden_selection_use": "True",
            "spec_status": ready,
        },
        {
            "spec_name": "multi_horizon_fee_exceedance",
            "spec_family": "FEE_PLUS_SLIPPAGE_EXCEEDANCE_LABELS",
            "horizons": "6|12",
            "required_inputs": "multi_horizon_forward_returns|fee_bps|slippage_bps",
            "forbidden_selection_use": "True",
            "spec_status": ready,
        },
        {
            "spec_name": "volatility_scaled_triple_barrier",
            "spec_family": "TRIPLE_BARRIER_VOLATILITY_SCALED",
            "horizons": "6|12",
            "required_inputs": "high_price|low_price|close_price|realized_vol_12",
            "forbidden_selection_use": "True",
            "spec_status": ready,
        },
        {
            "spec_name": "event_sampling_masks",
            "spec_family": "EVENT_SAMPLING",
            "horizons": "",
            "required_inputs": (
                "realized_vol_12|regime_label|adx_14|volume_zscore_12|close_zscore_12"
            ),
            "forbidden_selection_use": "False",
            "spec_status": "READY",
        },
        {
            "spec_name": "meta_label_base_signal_take",
            "spec_family": "META_LABELS",
            "horizons": "6|12",
            "required_inputs": "long_trade_taken|multi_horizon_fee_exceedance",
            "forbidden_selection_use": "True",
            "spec_status": ready,
        },
    ]


def _recommendation(specs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ready = any(row["spec_status"] == "READY" for row in specs)
    recommendation = (
        "BUILD_M20_REDESIGNED_RESEARCH_INPUTS"
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
        "plan_json": str(output_dir / "m20_input_redesign_plan.json"),
        "plan_md": str(output_dir / "m20_input_redesign_plan.md"),
        "input_upgrade_specs_csv": str(output_dir / "input_upgrade_specs.csv"),
        "blocked_specs_csv": str(output_dir / "blocked_specs.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any], specs: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# M20 Input Redesign Plan",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Ready specs: `{report['ready_spec_count']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Specs",
    ]
    lines.extend(f"- `{row['spec_name']}`: `{row['spec_status']}`" for row in specs)
    return "\n".join(lines) + "\n"


__all__ = ["plan_m20_input_redesign"]
