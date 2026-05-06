"""Research-only M20 confirmation-window planning helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


PLAN_DIR_NAME = "confirmation_plan"
CONDITIONAL_DIR_NAME = "conditional_usefulness_full_test"
PRIMARY_CONFIRMATION_SLICES = (
    ("momentum", "flat"),
    ("range", "low"),
    ("symbol", "BTC/USD"),
    ("macd", "positive"),
    ("volume", "low"),
)
NEGATIVE_CONFIRMATION_SLICES = (
    ("month", "2026-04"),
    ("quarter", "2026Q2"),
)
HONESTY_FLAGS = (
    "RESEARCH_ONLY_CONFIRMATION_PLAN",
    "NO_RUNTIME_EFFECT",
    "NO_PROMOTION_EFFECT",
    "NO_REGISTRY_WRITE",
    "NOT_PROMOTABLE",
    "NO_PROFITABILITY_CLAIM",
    "LONG_RUNS_MANUAL_ONLY",
    "SINGLE_RECENT_FOLD_SIGNAL_REQUIRES_CONFIRMATION",
    "STRATEGY_ENSEMBLE_INPUT_ONLY",
    "CONFIRMATION_REQUIRED_BEFORE_POLICY_EVALUATION",
    "AUTOGLUON_MEMBER_AUDIT_BLOCKED_BY_MISSING_MEMBER_PREDICTIONS",
    "WEAK_CANDIDATES_RETAINED_FOR_CONDITIONAL_REVIEW",
)
BLOCKER_FLAGS = (
    "CONFIRMATION_RUN_NOT_AVAILABLE",
    "CONFIRMATION_COMPARISON_PENDING",
)


def plan_m20_confirmation_window(
    *,
    run_dir: Path,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Write a deterministic manual-only confirmation plan for the best M20 filter."""
    # pylint: disable=too-many-locals
    resolved_run_dir = Path(run_dir).resolve()
    resolved_config = (
        Path(config_path).resolve()
        if config_path is not None
        else (Path.cwd() / "configs" / "training.m20.json").resolve()
    )
    output_dir = resolved_run_dir / "research_labels" / "vol_scaled" / PLAN_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    conditional_dir = (
        resolved_run_dir / "research_labels" / "vol_scaled" / CONDITIONAL_DIR_NAME
    )
    slice_rows = _read_csv_rows(conditional_dir / "conditional_usefulness_by_slice.csv")
    conditional_report = _load_json(conditional_dir / "conditional_usefulness_report.json")
    baseline_dir = resolved_run_dir / "research_labels" / "vol_scaled" / "fee_exceedance_baselines"
    baseline_manifest = _load_json(baseline_dir / "fee_baseline_manifest.json")
    baseline_metrics = _load_json(baseline_dir / "fee_baseline_metrics.json")
    _load_json(resolved_run_dir / "training_frame" / "m20_training_frame_export_manifest.json")
    ledger_rows = _read_csv_rows(
        resolved_run_dir
        / "research_labels"
        / "vol_scaled"
        / "model_member_audit"
        / "strategy_ensemble_candidate_ledger.csv"
    )
    window_support = _window_override_support(resolved_config)
    flags = [*HONESTY_FLAGS, *BLOCKER_FLAGS]
    flags.append(
        "CONFIRMATION_WINDOW_OVERRIDE_SUPPORTED"
        if window_support["supported"]
        else "CONFIRMATION_WINDOW_OVERRIDE_NOT_SUPPORTED"
    )
    flags = sorted(dict.fromkeys(flags))
    targets = _confirmation_targets(slice_rows)
    success_rules = _success_rules()
    expected_artifacts = _expected_artifacts()
    comparison_schema = _comparison_schema()
    manual_commands = _manual_commands_markdown(
        resolved_run_dir=resolved_run_dir,
        config_path=resolved_config,
        window_support=window_support,
    )
    output_files = _output_files(output_dir)
    plan = {
        "run_dir": str(resolved_run_dir),
        "confirmation_plan_dir": str(output_dir),
        "candidate_id": _fee_candidate_id(ledger_rows),
        "baseline_name": _best_baseline_name(baseline_manifest, baseline_metrics),
        "target_type": "fee_exceedance",
        "source_evidence_dir": str(conditional_dir),
        "prediction_rows_analyzed": conditional_report.get("prediction_rows_analyzed", ""),
        "search_breadth": conditional_report.get("search_breadth", {}),
        "target_slice_count": len(targets),
        "primary_slices": [_slice_id(*item) for item in PRIMARY_CONFIRMATION_SLICES],
        "negative_slices": [_slice_id(*item) for item in NEGATIVE_CONFIRMATION_SLICES],
        "window_override_support": window_support,
        "confirmation_stages": list(expected_artifacts["stages"].keys()),
        "honesty_flags": flags,
        "recommendation": _recommendations(window_support),
        "input_files": _input_files(
            resolved_run_dir,
            resolved_config,
            conditional_dir,
            baseline_dir,
        ),
        "output_files": output_files,
    }
    manifest = {
        "run_dir": str(resolved_run_dir),
        "confirmation_plan_dir": str(output_dir),
        "runtime_effect": "none_research_only",
        "registry_write": False,
        "promotion_effect": False,
        "long_runs_executed": False,
        "honesty_flags": flags,
        "output_files": output_files,
    }

    write_json_artifact(Path(output_files["confirmation_plan_manifest_json"]), manifest)
    write_json_artifact(Path(output_files["confirmation_plan_json"]), plan)
    Path(output_files["confirmation_plan_md"]).write_text(
        _plan_markdown(plan, targets, window_support),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["confirmation_slice_targets_csv"]), targets)
    write_json_artifact(Path(output_files["confirmation_success_rules_json"]), success_rules)
    Path(output_files["confirmation_manual_commands_md"]).write_text(
        manual_commands,
        encoding="utf-8",
    )
    write_json_artifact(
        Path(output_files["confirmation_expected_artifacts_json"]),
        expected_artifacts,
    )
    write_json_artifact(
        Path(output_files["confirmation_comparison_schema_json"]),
        comparison_schema,
    )
    return make_json_safe(
        {
            **plan,
            "manifest": manifest,
            "slice_targets": targets,
            "success_rules": success_rules,
            "expected_artifacts": expected_artifacts,
        }
    )


def _confirmation_targets(
    slice_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    by_id = {
        _slice_id(row.get("slice_family", ""), row.get("slice_value", "")): row
        for row in slice_rows
    }
    target_ids: dict[str, str] = {}
    for family, value in PRIMARY_CONFIRMATION_SLICES:
        target_ids[_slice_id(family, value)] = "PRIMARY_ENABLE_SLICE"
    for family, value in NEGATIVE_CONFIRMATION_SLICES:
        target_ids[_slice_id(family, value)] = "NEGATIVE_DISABLE_SLICE"
    for row in slice_rows:
        classification = row.get("classification", "")
        if classification in {"ENABLE_CANDIDATE", "WATCHLIST_CANDIDATE"}:
            target_ids.setdefault(
                _slice_id(row.get("slice_family", ""), row.get("slice_value", "")),
                classification.replace("_CANDIDATE", "_SLICE"),
            )
        if classification == "DISABLE_CANDIDATE":
            target_ids.setdefault(
                _slice_id(row.get("slice_family", ""), row.get("slice_value", "")),
                "DISABLE_SLICE",
            )
    rows = []
    for slice_id in sorted(target_ids):
        source = by_id.get(slice_id, {})
        family, value = _split_slice_id(slice_id)
        rows.append(
            {
                "slice_family": family,
                "slice_value": value,
                "slice_id": slice_id,
                "target_type": target_ids[slice_id],
                "original_classification": source.get("classification", "MISSING_IN_SOURCE"),
                "row_count": _to_int(source.get("row_count")),
                "positive_count": _to_int(source.get("positive_count")),
                "positive_rate": _to_float(source.get("positive_rate")),
                "average_precision": _to_float(source.get("average_precision")),
                "pr_auc_lift_over_base": _metric_lift(
                    source.get("average_precision"),
                    source.get("positive_rate"),
                ),
                "top_5_precision": _to_float(source.get("top_5_precision")),
                "top_5_lift": _to_float(source.get("top_5_lift")),
                "confirmation_priority": _priority(target_ids[slice_id]),
            }
        )
    return rows


def _success_rules() -> dict[str, Any]:
    return {
        "minimum_rows": 1000,
        "minimum_positives": 50,
        "CONFIRMED": {
            "minimum_rows": 1000,
            "minimum_positives": 50,
            "minimum_top_5_lift": 1.2,
            "minimum_pr_auc_lift_over_base": 0.015,
            "requires_top_5_precision_above_base_rate": True,
        },
        "STRONGLY_CONFIRMED": {
            "minimum_rows": 1000,
            "minimum_positives": 50,
            "minimum_top_5_lift": 1.5,
            "minimum_pr_auc_lift_over_base": 0.03,
            "requires_top_5_precision_above_base_rate": True,
        },
        "NOT_CONFIRMED": {
            "requires_enough_sample": True,
            "top_5_lift_at_or_below": 1.0,
            "or_pr_auc_at_or_below_base_rate": True,
        },
        "INCONCLUSIVE": {
            "low_rows_or_positives": True,
            "or_metrics_undefined": True,
        },
    }


def _recommendations(window_support: Mapping[str, Any]) -> list[str]:
    recommendations = [
        "Confirm fee-exceedance logistic useful slices on another window/fold.",
        "Do not run policy evaluation until confirmation artifacts exist.",
    ]
    if window_support["supported"]:
        recommendations.append(
            "Arash can manually launch export-only confirmation with start/end/tag."
        )
    else:
        recommendations.append(
            "Add a safe confirmation window override before launching a different window."
        )
    return recommendations


def _expected_artifacts() -> dict[str, Any]:
    return {
        "stages": {
            "A_export_market_feature_frame": {
                "manual": True,
                "expected_files": ["training_frame/m20_training_frame_features.csv"],
            },
            "B_generate_training_frame_labels": {
                "manual": True,
                "expected_files": [
                    "research_labels/vol_scaled/fee_exceedance_labels_vol_scaled.csv"
                ],
            },
            "C_analyze_label_readiness": {
                "manual": True,
                "expected_files": ["research_labels/readiness/label_readiness_report.json"],
            },
            "D_train_fee_exceedance_baseline": {
                "manual": True,
                "expected_files": [
                    "research_labels/vol_scaled/fee_exceedance_baselines/fee_baseline_metrics.json"
                ],
            },
            "E_export_full_predictions": {
                "manual": True,
                "expected_files": [
                    (
                        "research_labels/vol_scaled/fee_exceedance_baselines/"
                        "predictions_logistic_regression_tiny_test_full.csv"
                    )
                ],
            },
            "F_full_test_conditional_usefulness": {
                "manual": True,
                "expected_files": [
                    (
                        "research_labels/vol_scaled/conditional_usefulness_full_test/"
                        "conditional_usefulness_by_slice.csv"
                    )
                ],
            },
            "G_compare_original_vs_confirmation": {
                "manual": False,
                "expected_files": ["confirmation comparison report from comparison script"],
            },
        }
    }


def _comparison_schema() -> dict[str, Any]:
    return {
        "key_columns": ["slice_family", "slice_value"],
        "metric_columns": [
            "row_count",
            "positive_count",
            "positive_rate",
            "average_precision",
            "pr_auc_lift_over_base",
            "top_5_precision",
            "top_5_lift",
        ],
        "status_values": [
            "STRONGLY_CONFIRMED",
            "CONFIRMED",
            "NOT_CONFIRMED",
            "INCONCLUSIVE",
            "MISSING_IN_CONFIRMATION",
        ],
        "recommendation_values": [
            "CONFIRM_FEE_GATE_FOR_RESEARCH_POLICY",
            "NEED_MORE_WINDOWS",
            "REJECT_CONDITIONAL_SLICE",
            "FEATURE_OR_LABEL_INSTABILITY",
            "INCONCLUSIVE_LOW_SAMPLE",
        ],
    }


def _window_override_support(config_path: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    entrypoint = Path(__file__).resolve().with_name("__main__.py")
    service = Path(__file__).resolve().with_name("service.py")
    source = (
        entrypoint.read_text(encoding="utf-8")
        + "\n"
        + service.read_text(encoding="utf-8")
    )
    supported = all(
        token in source
        for token in (
            "confirmation-window-start",
            "confirmation-window-end",
            "confirmation-tag",
            "confirmation_window_start",
            "confirmation_window_end",
        )
    )
    return {
        "supported": supported,
        "config_path": str(config_path),
        "configured_recent_scoring_window_days": config.get("recent_scoring_window_days", ""),
        "configured_test_folds": config.get("test_folds", ""),
        "reason": (
            "The training entrypoint exposes manual-only confirmation window "
            "start/end/tag flags for score-only/export-only runs."
            if supported
            else (
                "The current training entrypoint exposes score-only and export "
                "flags, but no safe CLI/config override for selecting a distinct "
                "confirmation window or fold from the operator command."
            )
        ),
        "required_future_patch": "" if supported else "ADD_CONFIRMATION_WINDOW_OVERRIDE",
        "blocker_flag": "" if supported else "CONFIRMATION_WINDOW_OVERRIDE_NOT_SUPPORTED",
    }


def _manual_commands_markdown(
    *,
    resolved_run_dir: Path,
    config_path: Path,
    window_support: Mapping[str, Any],
) -> str:
    lines = [
        "# M20 Confirmation Manual Commands",
        "",
        (
            "These commands are prepared for Arash to run manually. Codex must "
            "not launch the long confirmation run."
        ),
        "",
        "## Current Support",
        "",
        f"- Window override supported: `{window_support['supported']}`",
        f"- Blocker: `{window_support['blocker_flag'] or 'none'}`",
        (
            "- Next code action if another window cannot be selected from config: "
            "`ADD_CONFIRMATION_WINDOW_OVERRIDE`"
            if not window_support["supported"]
            else "- Confirmation window CLI override is supported for manual runs."
        ),
        "",
        "## A. Export-Only Feature Frame For Confirmation Run",
        "",
        "- Heaviness: long/manual",
        "- Expected output: `<CONFIRMATION_RUN_DIR>/training_frame/`",
        "```powershell",
        (
            "python -m app.training --config <CONFIRMATION_CONFIG> "
            "--score-only <FITTED_MODELS_DIR> --parquet-dir <PARQUET_DIR> "
            "--export-training-frame-only --confirmation-window-start <START> "
            "--confirmation-window-end <END> --confirmation-tag <TAG>"
        ),
        "```",
        "",
        (
            "Use `<CONFIRMATION_CONFIG>` as a copy of the M20 config with an "
            "explicitly reviewed confirmation window. Do not invent unsupported "
            "CLI window flags."
        ),
        "",
        "## B. Generate Labels From Training Frame",
        "",
        "- Heaviness: light/medium",
        "- Expected output: `<CONFIRMATION_RUN_DIR>/research_labels/vol_scaled/`",
        "```powershell",
        (
            "python scripts/generate_m20_research_labels.py --run-dir "
            "<CONFIRMATION_RUN_DIR> --source training-frame --use-volatility"
        ),
        "```",
        "",
        "## C. Analyze Label Readiness",
        "",
        "- Heaviness: light",
        "```powershell",
        "python scripts/analyze_m20_label_readiness.py --run-dir <CONFIRMATION_RUN_DIR>",
        "```",
        "",
        "## D. Train Tiny Fee-Exceedance Baseline",
        "",
        "- Heaviness: medium",
        "```powershell",
        (
            "python scripts/train_m20_fee_exceedance_baseline.py --run-dir "
            "<CONFIRMATION_RUN_DIR> --export-full-predictions"
        ),
        "```",
        "",
        "## E. Run Full-Test Conditional Usefulness",
        "",
        "- Heaviness: light/medium",
        "```powershell",
        (
            "python scripts/analyze_m20_conditional_usefulness.py --run-dir "
            "<CONFIRMATION_RUN_DIR> --prediction-source full-test"
        ),
        "```",
        "",
        "## F. Compare Confirmation Against Original",
        "",
        "- Heaviness: light",
        "```powershell",
        (
            f"python scripts/compare_m20_confirmation_slices.py --original-run-dir "
            f"{resolved_run_dir} --confirmation-run-dir <CONFIRMATION_RUN_DIR>"
        ),
        "```",
        "",
        "## Reference Paths",
        "",
        f"- Original run: `{resolved_run_dir}`",
        f"- Base config seen by planner: `{config_path}`",
        (
            "- Placeholders required: `<CONFIRMATION_RUN_DIR>`, "
            "`<CONFIRMATION_CONFIG>`, `<FITTED_MODELS_DIR>`, `<PARQUET_DIR>`, "
            "`<START>`, `<END>`, `<TAG>`"
        ),
        "",
    ]
    return "\n".join(lines)


def _plan_markdown(
    plan: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    window_support: Mapping[str, Any],
) -> str:
    enabled = [row for row in targets if row["original_classification"] == "ENABLE_CANDIDATE"]
    disabled = [row for row in targets if row["original_classification"] == "DISABLE_CANDIDATE"]
    return "\n".join(
        [
            "# M20 Confirmation Plan",
            "",
            f"- Candidate: `{plan['candidate_id']}`",
            f"- Baseline: `{plan['baseline_name']}`",
            f"- Target slices: `{plan['target_slice_count']}`",
            f"- Window override supported: `{window_support['supported']}`",
            f"- Honesty flags: `{', '.join(plan['honesty_flags'])}`",
            "",
            "## Required Confirmation",
            "",
            "- The fee-exceedance logistic baseline has one recent-fold full-test evidence only.",
            "- Confirmation must run on another window/fold before policy evaluation.",
            "- No runtime, registry, promotion, or execution behavior changes are made.",
            "",
            "## Primary Slices",
            "",
            *[f"- `{item}`" for item in plan["primary_slices"]],
            "",
            "## Negative Slices",
            "",
            *[f"- `{item}`" for item in plan["negative_slices"]],
            "",
            "## Current Full-Test Slice Counts",
            "",
            f"- Enable targets in plan: `{len(enabled)}`",
            f"- Disable targets in plan: `{len(disabled)}`",
            "",
            "## Blocker",
            "",
            (
                f"- `{window_support['blocker_flag']}`: {window_support['reason']}"
                if window_support["blocker_flag"]
                else (
                    "- No override blocker remains; manual confirmation runs can "
                    "pass start/end/tag."
                )
            ),
            "",
            "## Recommendation",
            "",
            *[f"- {item}" for item in plan["recommendation"]],
            "",
        ]
    )


def _input_files(
    run_dir: Path,
    config_path: Path,
    conditional_dir: Path,
    baseline_dir: Path,
) -> dict[str, str]:
    return {
        "strategy_ensemble_candidate_ledger_csv": str(
            run_dir
            / "research_labels"
            / "vol_scaled"
            / "model_member_audit"
            / "strategy_ensemble_candidate_ledger.csv"
        ),
        "conditional_usefulness_report_json": str(
            conditional_dir / "conditional_usefulness_report.json"
        ),
        "conditional_enable_disable_summary_csv": str(
            conditional_dir / "conditional_enable_disable_summary.csv"
        ),
        "conditional_usefulness_by_slice_csv": str(
            conditional_dir / "conditional_usefulness_by_slice.csv"
        ),
        "fee_baseline_manifest_json": str(baseline_dir / "fee_baseline_manifest.json"),
        "fee_baseline_metrics_json": str(baseline_dir / "fee_baseline_metrics.json"),
        "training_frame_manifest_json": str(
            run_dir / "training_frame" / "m20_training_frame_export_manifest.json"
        ),
        "config_path": str(config_path),
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    names = {
        "confirmation_plan_manifest_json": "confirmation_plan_manifest.json",
        "confirmation_plan_json": "confirmation_plan.json",
        "confirmation_plan_md": "confirmation_plan.md",
        "confirmation_slice_targets_csv": "confirmation_slice_targets.csv",
        "confirmation_success_rules_json": "confirmation_success_rules.json",
        "confirmation_manual_commands_md": "confirmation_manual_commands.md",
        "confirmation_expected_artifacts_json": "confirmation_expected_artifacts.json",
        "confirmation_comparison_schema_json": "confirmation_comparison_schema.json",
    }
    return {key: str(output_dir / name) for key, name in names.items()}


def _fee_candidate_id(ledger_rows: Sequence[Mapping[str, str]]) -> str:
    for row in ledger_rows:
        if row.get("model_name") == "logistic_regression_tiny":
            return row.get("candidate_id", "logistic_regression_tiny")
    return "logistic_regression_tiny"


def _best_baseline_name(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> str:
    for key in ("best_baseline_name", "baseline_name"):
        if manifest.get(key):
            return str(manifest[key])
    best = max(
        metrics.get("baselines", []),
        key=lambda row: (row.get("average_precision") or 0.0, row.get("balanced_accuracy") or 0.0),
        default={},
    )
    return str(best.get("baseline_name") or "logistic_regression_tiny")


def _priority(target_type: str) -> int:
    order = {
        "PRIMARY_ENABLE_SLICE": 1,
        "NEGATIVE_DISABLE_SLICE": 2,
        "ENABLE_SLICE": 3,
        "WATCHLIST_SLICE": 4,
        "DISABLE_SLICE": 5,
    }
    return order.get(target_type, 9)


def _metric_lift(metric: Any, base: Any) -> float:
    metric_value = _to_float(metric)
    base_value = _to_float(base)
    return metric_value - base_value


def _slice_id(family: str, value: str) -> str:
    return f"{family}={value}"


def _split_slice_id(slice_id: str) -> tuple[str, str]:
    if "=" not in slice_id:
        return slice_id, ""
    family, value = slice_id.split("=", 1)
    return family, value


def _to_int(value: Any) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}
