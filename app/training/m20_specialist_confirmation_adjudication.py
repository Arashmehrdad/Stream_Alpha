"""Research-only M20 specialist confirmation adjudication from existing artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


OUTPUT_DIR_NAME = "specialist_confirmation_adjudication"
SPECIALIST_MODELS = ("neuralforecast_patchtst", "neuralforecast_nhits")
REQUIRED_CONFIRMATION_FILES = (
    "report.json",
    "recommendation.json",
    "comparison.csv",
    "topk_metrics.csv",
    "model_metrics.csv",
)
HONESTY_FLAGS = (
    "RESEARCH_ONLY_SPECIALIST_CONFIRMATION_ADJUDICATION",
    "EXISTING_ARTIFACTS_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
    "NO_MODEL_RETRAIN",
    "NO_REGISTRY_WRITE",
    "NO_PROMOTION_EFFECT",
    "NOT_BACKTEST",
)
BLOCKERS = (
    "ECONOMIC_POLICY_EVALUATION_REQUIRED",
    "GLOBAL_CLASSIFIER_QUALITY_WEAK",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
)
NEXT_ACTION = "DESIGN_COST_AWARE_SPECIALIST_POLICY_EVALUATOR"


def write_m20_specialist_confirmation_adjudication(
    *,
    confirmation_run_dir: Path,
    original_run_dir: Path | None = None,
) -> dict[str, Any]:
    """Write conservative specialist confirmation adjudication from saved artifacts."""
    # pylint: disable=too-many-locals
    confirmation_dir = Path(confirmation_run_dir).resolve()
    confirmation_source_dir = _source_dir(confirmation_dir)
    _assert_required_sources(confirmation_source_dir, run_label="confirmation")

    output_dir = (
        confirmation_dir
        / "research_labels"
        / "vol_scaled"
        / OUTPUT_DIR_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report = _read_json(confirmation_source_dir / "report.json")
    recommendation = _read_json(confirmation_source_dir / "recommendation.json")
    comparison_rows = _read_csv(confirmation_source_dir / "comparison.csv")
    topk_rows = _read_csv(confirmation_source_dir / "topk_metrics.csv")
    model_rows = _read_csv(confirmation_source_dir / "model_metrics.csv")

    original_context: dict[str, Any] = {"provided": False}
    if original_run_dir is not None:
        original_dir = Path(original_run_dir).resolve()
        original_source_dir = _source_dir(original_dir)
        if all((original_source_dir / name).exists() for name in REQUIRED_CONFIRMATION_FILES):
            original_context = {
                "provided": True,
                "used_for_context_only": True,
                "original_run_dir": str(original_dir),
                "source_dir": str(original_source_dir),
                "report": _read_json(original_source_dir / "report.json"),
                "recommendation": _read_json(original_source_dir / "recommendation.json"),
                "comparison_rows": _read_csv(original_source_dir / "comparison.csv"),
                "topk_rows": _read_csv(original_source_dir / "topk_metrics.csv"),
                "model_rows": _read_csv(original_source_dir / "model_metrics.csv"),
            }
        else:
            original_context = {
                "provided": True,
                "used_for_context_only": False,
                "original_run_dir": str(original_dir),
                "source_dir": str(original_source_dir),
                "missing_required_artifacts": True,
            }

    evidence_metrics = _build_evidence_metrics(
        comparison_rows=comparison_rows,
        topk_rows=topk_rows,
        model_rows=model_rows,
    )
    candidate_decisions = _candidate_decisions(evidence_metrics)
    decision_by_model = {
        row["model_name"]: row["candidate_decision"] for row in candidate_decisions
    }

    patchtst_decision = decision_by_model.get(
        "neuralforecast_patchtst",
        "CONFIRMED_SELECTIVE_RANK_SLICE_RESEARCH_CANDIDATE",
    )
    nhits_decision = decision_by_model.get(
        "neuralforecast_nhits",
        "SECONDARY_WATCHLIST_OR_WEAKER_CANDIDATE",
    )
    output_files = _output_files(output_dir)
    adjudication = {
        "confirmation_run_dir": str(confirmation_dir),
        "original_run_dir": (
            None if original_run_dir is None else str(Path(original_run_dir).resolve())
        ),
        "overall_status": "RESEARCH_ONLY_NOT_PROMOTABLE",
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "patchtst_decision": patchtst_decision,
        "nhits_decision": nhits_decision,
        "joined_rows": int(report.get("joined_rows", 0)),
        "best_candidate": str(report.get("best_candidate", "")),
        "recommendation": str(report.get("recommendation", "")),
        "required_blockers": list(BLOCKERS),
        "required_next_action": NEXT_ACTION,
        "interpretation": (
            "PatchTST remains a selective rank/top-k and slice research candidate, "
            "not a globally strong classifier and not runtime-ready. "
            "This is not promotion or profitability evidence."
        ),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    manifest = {
        "confirmation_run_dir": str(confirmation_dir),
        "confirmation_source_dir": str(confirmation_source_dir),
        "original_context": original_context,
        "source_report": report,
        "source_recommendation": recommendation,
        "overall_status": adjudication["overall_status"],
        "runtime_status": adjudication["runtime_status"],
        "promotion_status": adjudication["promotion_status"],
        "profitability_status": adjudication["profitability_status"],
        "required_blockers": list(BLOCKERS),
        "required_next_action": NEXT_ACTION,
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    next_actions = _next_actions()

    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(
        Path(output_files["specialist_confirmation_adjudication_json"]),
        adjudication,
    )
    Path(output_files["specialist_confirmation_adjudication_md"]).write_text(
        _markdown(adjudication, candidate_decisions, evidence_metrics, next_actions),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["candidate_decisions_csv"]), candidate_decisions)
    write_csv_artifact(Path(output_files["evidence_metrics_csv"]), evidence_metrics)
    write_csv_artifact(Path(output_files["next_actions_csv"]), next_actions)
    return make_json_safe(adjudication)


def _build_evidence_metrics(
    *,
    comparison_rows: Sequence[Mapping[str, str]],
    topk_rows: Sequence[Mapping[str, str]],
    model_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    comparison_by_model = {
        str(row.get("model_name", "")): row for row in comparison_rows
    }
    model_metrics_by_model = {
        str(row.get("model_name", "")): row for row in model_rows
    }
    topk_by_model_fraction = {
        (str(row.get("model_name", "")), str(row.get("top_k_fraction", ""))): row
        for row in topk_rows
    }
    output: list[dict[str, Any]] = []
    for model_name in SPECIALIST_MODELS:
        comparison = comparison_by_model.get(model_name, {})
        model_metric = model_metrics_by_model.get(model_name, {})
        top1 = topk_by_model_fraction.get((model_name, "0.01"), {})
        top2 = topk_by_model_fraction.get((model_name, "0.02"), {})
        top5 = topk_by_model_fraction.get((model_name, "0.05"), {})
        top10 = topk_by_model_fraction.get((model_name, "0.1"), {})
        output.append(
            {
                "model_name": model_name,
                "top1_precision": _to_float(top1.get("precision")),
                "top1_lift": _to_float(top1.get("lift")),
                "top2_precision": _to_float(top2.get("precision")),
                "top2_lift": _to_float(top2.get("lift")),
                "top5_precision": _to_float(top5.get("precision")),
                "top5_lift": _to_float(top5.get("lift")),
                "top10_precision": _to_float(top10.get("precision")),
                "top10_lift": _to_float(top10.get("lift")),
                "pr_auc": _to_float(comparison.get("pr_auc")),
                "roc_auc": _to_float(comparison.get("roc_auc")),
                "enable_slice_count": _to_int(comparison.get("enable_slice_count")),
                "best_slice": str(comparison.get("best_slice", "")),
                "positive_rate": _to_float(model_metric.get("positive_rate")),
                "balanced_accuracy": _to_float(model_metric.get("balanced_accuracy")),
                "overall_precision": _to_float(model_metric.get("precision")),
                "recall": _to_float(model_metric.get("recall")),
                "f1": _to_float(model_metric.get("f1")),
            }
        )
    return output


def _candidate_decisions(
    evidence_metrics: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    evidence_by_model = {
        str(row["model_name"]): row for row in evidence_metrics
    }
    patch = evidence_by_model.get("neuralforecast_patchtst", {})
    nhits = evidence_by_model.get("neuralforecast_nhits", {})
    patch_is_selective = (
        _to_float(patch.get("top5_lift")) >= 1.2
        and _to_int(patch.get("enable_slice_count")) > 0
    )
    patch_is_globally_weak = (
        _to_float(patch.get("roc_auc")) < 0.52
        or _to_float(patch.get("balanced_accuracy")) < 0.55
    )
    nhits_stronger = (
        _to_float(nhits.get("top5_lift")) > _to_float(patch.get("top5_lift"))
        and _to_float(nhits.get("pr_auc")) >= _to_float(patch.get("pr_auc"))
        and _to_int(nhits.get("enable_slice_count")) >= _to_int(patch.get("enable_slice_count"))
    )
    patch_decision = (
        "CONFIRMED_SELECTIVE_RANK_SLICE_RESEARCH_CANDIDATE"
        if patch_is_selective and patch_is_globally_weak
        else "SECONDARY_WATCHLIST_OR_WEAKER_CANDIDATE"
    )
    nhits_decision = (
        "POTENTIALLY_STRONGER_THAN_PATCHTST_REVIEW_REQUIRED"
        if nhits_stronger
        else "SECONDARY_WATCHLIST_OR_WEAKER_CANDIDATE"
    )
    return [
        {
            "model_name": "neuralforecast_patchtst",
            "candidate_decision": patch_decision,
            "decision_rationale": (
                "Strong top-k and slice signal but weak global classifier quality."
            ),
            "runtime_status": "NO_RUNTIME_EFFECT",
            "promotion_status": "NOT_PROMOTABLE",
            "profitability_status": "NO_PROFIT_CLAIM",
        },
        {
            "model_name": "neuralforecast_nhits",
            "candidate_decision": nhits_decision,
            "decision_rationale": (
                "Remains secondary/watchlist unless confirmation artifacts show stronger evidence."
            ),
            "runtime_status": "NO_RUNTIME_EFFECT",
            "promotion_status": "NOT_PROMOTABLE",
            "profitability_status": "NO_PROFIT_CLAIM",
        },
    ]


def _next_actions() -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "action": NEXT_ACTION,
            "rationale": (
                "Design cost-aware specialist policy evaluation before any runtime "
                "or promotion claim."
            ),
        },
        {
            "priority": "2",
            "action": "KEEP_PATCHTST_AS_RESEARCH_ONLY",
            "rationale": (
                "Do not use as runtime/global classifier without "
                "policy economics evidence."
            ),
        },
        {
            "priority": "3",
            "action": "KEEP_NHITS_AS_SECONDARY_WATCHLIST",
            "rationale": "Retain only as secondary context unless stronger artifacts appear.",
        },
    ]


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "specialist_confirmation_adjudication_json": str(
            output_dir / "specialist_confirmation_adjudication.json"
        ),
        "specialist_confirmation_adjudication_md": str(
            output_dir / "specialist_confirmation_adjudication.md"
        ),
        "candidate_decisions_csv": str(output_dir / "candidate_decisions.csv"),
        "evidence_metrics_csv": str(output_dir / "evidence_metrics.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
    }


def _source_dir(run_dir: Path) -> Path:
    return (
        run_dir
        / "research_labels"
        / "vol_scaled"
        / "specialist_conditional_usefulness"
    )


def _assert_required_sources(source_dir: Path, *, run_label: str) -> None:
    missing = [
        str(source_dir / name)
        for name in REQUIRED_CONFIRMATION_FILES
        if not (source_dir / name).exists()
    ]
    if missing:
        raise ValueError(
            f"Missing {run_label} specialist conditional usefulness artifacts: "
            + "; ".join(missing)
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Missing artifact file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"Missing artifact file: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _markdown(
    adjudication: Mapping[str, Any],
    candidate_decisions: Sequence[Mapping[str, Any]],
    evidence_metrics: Sequence[Mapping[str, Any]],
    next_actions: Sequence[Mapping[str, str]],
) -> str:
    lines = [
        "# M20 Specialist Confirmation Adjudication",
        "",
        f"- Overall status: `{adjudication['overall_status']}`",
        f"- PatchTST: `{adjudication['patchtst_decision']}`",
        f"- NHITS: `{adjudication['nhits_decision']}`",
        f"- Runtime status: `{adjudication['runtime_status']}`",
        f"- Promotion status: `{adjudication['promotion_status']}`",
        f"- Profitability status: `{adjudication['profitability_status']}`",
        "",
        adjudication["interpretation"],
        "",
        "## Candidate Decisions",
    ]
    lines.extend(
        f"- `{row['model_name']}` -> `{row['candidate_decision']}`: {row['decision_rationale']}"
        for row in candidate_decisions
    )
    lines.append("")
    lines.append("## Key Evidence Metrics")
    lines.extend(
        (
            f"- `{row['model_name']}`: top1_lift={float(row['top1_lift']):.6f}, "
            f"top5_lift={float(row['top5_lift']):.6f}, pr_auc={float(row['pr_auc']):.6f}, "
            f"roc_auc={float(row['roc_auc']):.6f}, "
            f"enable_slice_count={int(row['enable_slice_count'])}"
        )
        for row in evidence_metrics
    )
    lines.append("")
    lines.append("## Required Blockers")
    lines.extend(f"- `{blocker}`" for blocker in adjudication["required_blockers"])
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


__all__ = ["write_m20_specialist_confirmation_adjudication"]
