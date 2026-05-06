"""Research-only M20 decision memo and evidence adjudication."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "m20_decision_memo"
DECISION_STATUSES = (
    "RESEARCH_SIGNAL_CONFIRMED",
    "ECONOMICS_NOT_STABLE",
    "NO_STABLE_TAIL_FILTER",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
)
HONESTY_FLAGS = (
    "RESEARCH_ONLY_DECISION_MEMO",
    "NO_RUNTIME_EFFECT",
    "NO_REGISTRY_WRITE",
    "NO_PROMOTION_EFFECT",
    "NOT_BACKTEST",
    "NO_PROFIT_CLAIM",
)


def write_m20_decision_memo(*, base_run_dir: Path) -> dict[str, Any]:
    """Write the M20 research-only decision memo from existing artifacts."""
    base_dir = Path(base_run_dir).resolve()
    vol_dir = base_dir / "research_labels" / "vol_scaled"
    output_dir = vol_dir / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence = _evidence_table(vol_dir)
    forks = _next_forks()
    decision = {
        "decision": "PAUSE_RANK_GATE_AS_STANDALONE_PATH",
        "decision_statuses": list(DECISION_STATUSES),
        "summary": (
            "Fee-exceedance logistic ranking is confirmed as a research signal, "
            "but sparse rank-gate economics are not stable and no stable "
            "tail-risk filter was found."
        ),
        "recommended_forks": [row["fork"] for row in forks],
        "honesty_flags": list(HONESTY_FLAGS),
        "runtime_status": "NOT_RUNTIME_READY",
        "promotion_status": "NOT_PROMOTABLE",
    }
    output_files = _output_files(output_dir)
    manifest = {
        "base_run_dir": str(base_dir),
        "output_dir": str(output_dir),
        "decision_statuses": list(DECISION_STATUSES),
        "honesty_flags": list(HONESTY_FLAGS),
        "source_artifacts": [row["artifact_path"] for row in evidence],
        "output_files": output_files,
    }
    report = {
        **decision,
        "evidence_count": len(evidence),
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["decision_json"]), report)
    Path(output_files["decision_md"]).write_text(
        _markdown(report, evidence, forks),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["evidence_table_csv"]), evidence)
    write_csv_artifact(Path(output_files["next_forks_csv"]), forks)
    return make_json_safe(report)


def _evidence_table(vol_dir: Path) -> list[dict[str, Any]]:
    packet = _read_json(vol_dir / "rank_gate_evidence_packet" / "rank_gate_evidence_packet.json")
    economics = _read_json(vol_dir / "rank_gate_economics" / "recommendation.json")
    net_diag = _read_json(vol_dir / "rank_gate_net_diagnostics" / "recommendation.json")
    tail = _read_json(vol_dir / "rank_gate_tail_analysis" / "recommendation.json")
    tail_filter = _read_json(vol_dir / "rank_gate_tail_filter" / "recommendation.json")
    model_audit = _read_json(
        vol_dir / "model_member_audit" / "model_member_audit_report.json"
    )
    selector = _read_json(
        vol_dir / "strategy_selector_design" / "strategy_selector_candidate_spec.json"
    )
    return [
        _evidence_row(
            "rank_gate_evidence_packet",
            vol_dir / "rank_gate_evidence_packet",
            packet.get("evidence_status", "RESEARCH_CONFIRMED_RANK_GATE"),
            "Signal confirmed across original, prior-year, and prev-prev-year windows.",
        ),
        _evidence_row(
            "rank_gate_economics",
            vol_dir / "rank_gate_economics",
            economics.get("recommendation", "UNKNOWN"),
            "Lift persists, but net return proxies are mixed across windows.",
        ),
        _evidence_row(
            "rank_gate_net_diagnostics",
            vol_dir / "rank_gate_net_diagnostics",
            net_diag.get("recommendation", "UNKNOWN"),
            "Net-proxy decomposition preserves NET_PROXY_MIXED and NOT_PNL.",
        ),
        _evidence_row(
            "rank_gate_tail_analysis",
            vol_dir / "rank_gate_tail_analysis",
            tail.get("recommendation", "UNKNOWN"),
            "Tail diagnostics found negative windows and unstable concentration slices.",
        ),
        _evidence_row(
            "rank_gate_tail_filter",
            vol_dir / "rank_gate_tail_filter",
            tail_filter.get("recommendation", "UNKNOWN"),
            "Exploratory tail filters did not find a stable risk-reduction rule.",
        ),
        _evidence_row(
            "model_member_audit",
            vol_dir / "model_member_audit",
            model_audit.get("recommendation", "UNKNOWN"),
            (
                "AutoGluon member predictions remain missing; "
                "PatchTST/NHITS stay conditionally unknown."
            ),
        ),
        _evidence_row(
            "strategy_selector_design",
            vol_dir / "strategy_selector_design",
            selector.get("evidence_status", "RESEARCH_ONLY"),
            "Selector design remains a non-runtime opportunity-gate design input only.",
        ),
    ]


def _evidence_row(
    source: str,
    path: Path,
    status: str,
    adjudication: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "artifact_path": str(path),
        "artifact_present": path.exists(),
        "status": status,
        "adjudication": adjudication,
    }


def _next_forks() -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "fork": "pause rank-gate path",
            "rationale": "Standalone rank gate has mixed economics and no stable tail filter.",
        },
        {
            "priority": "2",
            "fork": "try richer strategy-family modules",
            "rationale": (
                "Use the fee-exceedance gate only as an optional filter for "
                "future momentum/breakout/range modules."
            ),
        },
        {
            "priority": "3",
            "fork": "export AutoGluon member predictions",
            "rationale": (
                "AutoGluon is a model factory, but member-level predictions "
                "are still missing."
            ),
        },
        {
            "priority": "4",
            "fork": "try different horizon or label",
            "rationale": "Current fee-exceedance label confirms signal but not stable net proxy.",
        },
        {
            "priority": "5",
            "fork": (
                "package M20 as research-negative for profitability but "
                "infrastructure-positive"
            ),
            "rationale": (
                "Infrastructure works; current path is not promotion-ready "
                "or profit evidence."
            ),
        },
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "decision_md": str(output_dir / "decision_memo.md"),
        "decision_json": str(output_dir / "decision_memo.json"),
        "evidence_table_csv": str(output_dir / "evidence_table.csv"),
        "next_forks_csv": str(output_dir / "next_forks.csv"),
    }


def _markdown(
    report: Mapping[str, Any],
    evidence: Sequence[Mapping[str, Any]],
    forks: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "# M20 Research Decision Memo",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Statuses: `{', '.join(report['decision_statuses'])}`",
        f"- Runtime status: `{report['runtime_status']}`",
        f"- Promotion status: `{report['promotion_status']}`",
        f"- Honesty flags: `{', '.join(report['honesty_flags'])}`",
        "",
        report["summary"],
        "",
        "## Evidence",
        "",
    ]
    for row in evidence:
        lines.append(
            f"- `{row['source']}`: `{row['status']}` - {row['adjudication']}"
        )
    lines.extend(["", "## Recommended Forks", ""])
    for row in forks:
        lines.append(f"- {row['priority']}. {row['fork']}: {row['rationale']}")
    lines.extend(
        [
            "",
            "This memo does not change runtime, registry, promotion, trading, "
            "backtest, model-retrain, long-run, or profit-claim workflows.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "MISSING_ARTIFACT"}
    return json.loads(path.read_text(encoding="utf-8"))


def read_decision_csv(path: Path) -> list[dict[str, str]]:
    """Read decision memo CSV output for tests."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]
