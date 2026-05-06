"""Research-only M20 path adjudication after strategy-family diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "m20_research_path_adjudication"
DECISION_STATUSES = (
    "RANK_GATE_SIGNAL_CONFIRMED",
    "RANK_GATE_ECONOMICS_UNSTABLE",
    "VOLATILITY_LABEL_LIFT_CONFIRMED",
    "VOLATILITY_COMBO_NOT_ECONOMICALLY_STABLE",
    "ABSTENTION_IMPLEMENTABLE_RULES_WEAK",
    "ORACLE_HOLD_RULES_NOT_IMPLEMENTABLE",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
)
HONESTY_FLAGS = (
    "RESEARCH_ONLY_PATH_ADJUDICATION",
    "EXISTING_ARTIFACTS_ONLY",
    "NO_RUNTIME_EFFECT",
    "NO_REGISTRY_WRITE",
    "NO_PROMOTION_EFFECT",
    "NOT_BACKTEST",
    "NO_PROFIT_CLAIM",
)
RECOMMENDED_NEXT_ACTION = "PLAN_ROW_LEVEL_SPECIALIST_PREDICTION_EXPORT"


def write_m20_research_path_adjudication(*, base_run_dir: Path) -> dict[str, Any]:
    """Write the current M20 research path adjudication from existing artifacts."""
    base_dir = Path(base_run_dir).resolve()
    vol_dir = base_dir / "research_labels" / "vol_scaled"
    output_dir = vol_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = _evidence_rollup(vol_dir)
    decisions = _path_decisions()
    next_actions = _next_actions()
    output_files = _output_files(output_dir)
    adjudication = {
        "decision": "STOP_CURRENT_FILTER_CHAIN_AND_PLAN_SPECIALIST_EXPORT",
        "decision_statuses": list(DECISION_STATUSES),
        "recommended_next_action": RECOMMENDED_NEXT_ACTION,
        "summary": (
            "Rank-gate signal remains real, but current filter and volatility paths "
            "do not produce stable implementable economics. The next useful step is "
            "row-level prediction export for existing specialist candidates."
        ),
        "runtime_status": "NOT_RUNTIME_READY",
        "promotion_status": "NOT_PROMOTABLE",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    manifest = {
        "base_run_dir": str(base_dir),
        "output_dir": str(output_dir),
        "source": "existing_m20_research_artifacts",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["adjudication_json"]), adjudication)
    Path(output_files["adjudication_md"]).write_text(
        _markdown(adjudication, evidence, decisions, next_actions),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["evidence_rollup_csv"]), evidence)
    write_csv_artifact(Path(output_files["path_decisions_csv"]), decisions)
    write_csv_artifact(Path(output_files["next_actions_csv"]), next_actions)
    return make_json_safe(adjudication)


def _evidence_rollup(vol_dir: Path) -> list[dict[str, Any]]:
    sources = [
        (
            "prior_decision_memo",
            vol_dir / "m20_decision_memo" / "decision_memo.json",
            "decision",
            "Rank gate already paused as standalone after mixed economics.",
        ),
        (
            "strategy_family_adjudication",
            vol_dir / "strategy_family_adjudication" / "recommendation.json",
            "recommendation",
            "Volatility is primary family; momentum secondary; range watchlist.",
        ),
        (
            "volatility_deep_dive",
            vol_dir / "volatility_expansion_deep_dive" / "recommendation.json",
            "recommendation",
            "Volatility label lift is confirmed across windows.",
        ),
        (
            "volatility_combo_economics",
            vol_dir / "volatility_combo_economics" / "recommendation.json",
            "recommendation",
            "Volatility combo did not stabilize economics.",
        ),
        (
            "abstention_hold",
            vol_dir / "abstention_hold_research" / "recommendation.json",
            "recommendation",
            "Implementable abstention is watchlist; oracle rules are not usable.",
        ),
        (
            "model_member_audit",
            vol_dir / "model_member_audit" / "candidate_next_actions.csv",
            "",
            "Existing specialist candidates need row-level prediction export.",
        ),
    ]
    output = []
    for source, path, key, adjudication in sources:
        payload = _read_json(path) if path.suffix == ".json" else {}
        output.append(
            {
                "source": source,
                "artifact_path": str(path),
                "artifact_present": path.exists(),
                "status": payload.get(key, "CSV_LEDGER" if path.exists() else "MISSING"),
                "adjudication": adjudication,
            }
        )
    return output


def _path_decisions() -> list[dict[str, str]]:
    return [
        {
            "path": "rank_gate_standalone",
            "decision": "PAUSE",
            "rationale": "Signal confirmed but economics and tail filters are unstable.",
        },
        {
            "path": "volatility_combo",
            "decision": "DO_NOT_ESCALATE_TO_CONFIRMATION_YET",
            "rationale": "Label lift persists, but combo economics are not stable.",
        },
        {
            "path": "momentum_and_volatility_context",
            "decision": "KEEP_AS_CONTEXT_FEATURES_ONLY",
            "rationale": "Useful explanatory context, not a standalone tradable edge.",
        },
        {
            "path": "abstention_hold",
            "decision": "KEEP_AS_RESEARCH_FILTER_ONLY",
            "rationale": (
                "Prediction-time abstention is weak; oracle net rules are "
                "not implementable."
            ),
        },
        {
            "path": "row_level_specialist_predictions",
            "decision": "PLAN_NEXT",
            "rationale": (
                "More candidate predictions are needed to find conditionally "
                "useful specialists."
            ),
        },
    ]


def _next_actions() -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "action": RECOMMENDED_NEXT_ACTION,
            "heaviness": "planning_only",
            "rationale": (
                "Prepare manual/light commands and artifact contracts to export row-level "
                "NHITS/PatchTST predictions for conditional usefulness."
            ),
        },
        {
            "priority": "2",
            "action": "DESIGN_ALTERNATE_HORIZON_OR_LABEL",
            "heaviness": "research_design_only",
            "rationale": "Current fee-exceedance target finds signal but not stable economics.",
        },
        {
            "priority": "3",
            "action": "KEEP_CURRENT_FILTER_CHAIN_PAUSED",
            "heaviness": "no_action",
            "rationale": (
                "Additional filters on the same logistic gate are unlikely "
                "to reveal edge."
            ),
        },
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "adjudication_json": str(output_dir / "research_path_adjudication.json"),
        "adjudication_md": str(output_dir / "research_path_adjudication.md"),
        "evidence_rollup_csv": str(output_dir / "evidence_rollup.csv"),
        "path_decisions_csv": str(output_dir / "path_decisions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
    }


def _markdown(
    report: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, str]],
    next_actions: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "# M20 Research Path Adjudication",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Next action: `{report['recommended_next_action']}`",
        f"- Statuses: `{', '.join(report['decision_statuses'])}`",
        f"- Honesty flags: `{', '.join(report['honesty_flags'])}`",
        "",
        report["summary"],
        "",
        "## Evidence",
    ]
    lines.extend(
        f"- `{row['source']}`: `{row['status']}` - {row['adjudication']}"
        for row in evidence
    )
    lines.append("")
    lines.append("## Path Decisions")
    lines.extend(
        f"- `{row['path']}` -> `{row['decision']}`: {row['rationale']}"
        for row in decisions
    )
    lines.append("")
    lines.append("## Next Actions")
    lines.extend(
        f"- `{row['priority']}` `{row['action']}`: {row['rationale']}"
        for row in next_actions
    )
    lines.append("")
    lines.append("Research-only. No runtime, registry, promotion, backtest, or profit claim.")
    lines.append("")
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_research_path_csv(path: Path) -> list[dict[str, str]]:
    """Read research path adjudication CSV for tests."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
