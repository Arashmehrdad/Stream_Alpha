"""Challenger-versus-champion comparison helpers for M7."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.common.time import to_rfc3339, utc_now
from app.training.dataset import (
    LEGACY_ARCHIVED_MODEL_NAMES,
    TrainingConfig,
    load_training_config,
)
from app.training.registry import (
    build_run_manifest,
    load_current_registry_entry,
    read_json,
    write_json_atomic,
)


_EPSILON = 1e-12
_PRIMARY_METRIC = "mean_long_only_net_value_proxy"
_REQUIRED_PROMOTION_BASELINES = ("persistence_3", "dummy_most_frequent")


@dataclass(frozen=True, slots=True)
class ComparisonPolicy:
    """Deterministic promotion guardrails loaded from the M7 config."""

    primary_metric: str
    max_directional_accuracy_regression: float
    max_brier_score_worsening: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation for artifacts."""
        return {
            "primary_metric": self.primary_metric,
            "max_directional_accuracy_regression": self.max_directional_accuracy_regression,
            "max_brier_score_worsening": self.max_brier_score_worsening,
        }


@dataclass(frozen=True, slots=True)
class WorkflowConfig:
    """M7 config wrapper over the existing M3 training configuration."""

    training: TrainingConfig
    comparison_policy: ComparisonPolicy


def load_workflow_config(config_path: Path) -> WorkflowConfig:
    """Load the M7 JSON config plus explicit comparison policy fields."""
    raw_config = json.loads(Path(config_path).read_text(encoding="utf-8-sig"))
    try:
        policy_block = dict(raw_config["comparison_policy"])
        policy = ComparisonPolicy(
            primary_metric=str(policy_block["primary_metric"]),
            max_directional_accuracy_regression=float(
                policy_block["max_directional_accuracy_regression"],
            ),
            max_brier_score_worsening=float(policy_block["max_brier_score_worsening"]),
        )
    except KeyError as error:
        raise ValueError(
            "configs/training.m7.json is missing required comparison_policy fields",
        ) from error
    if policy.primary_metric != _PRIMARY_METRIC:
        raise ValueError(
            "M7 promotion primary_metric must be mean_long_only_net_value_proxy"
        )

    return WorkflowConfig(
        training=load_training_config(Path(config_path)),
        comparison_policy=policy,
    )


# pylint: disable=too-many-locals
def compare_run_to_current(
    *,
    run_dir: Path,
    config_path: Path,
    registry_root: Path | None = None,
) -> dict[str, Any]:
    """Compare one challenger run to the current promoted champion, when present."""
    workflow = load_workflow_config(config_path)
    challenger_manifest = build_run_manifest(run_dir)
    challenger_summary = _load_summary_from_manifest(challenger_manifest)
    challenger_metrics = dict(challenger_manifest["winner"]["metrics"])
    baseline_checks = _baseline_checks(
        challenger_summary,
        challenger_metrics=challenger_metrics,
        primary_metric=workflow.comparison_policy.primary_metric,
    )
    current_entry = load_current_registry_entry(registry_root)
    if current_entry is not None and _is_legacy_archived_current_entry(current_entry):
        current_entry = None
    if current_entry is None:
        return _no_champion_payload(
            challenger_manifest,
            workflow.comparison_policy,
            baseline_checks=baseline_checks,
        )

    champion_manifest_path = Path(str(current_entry["run_manifest_path"])).resolve()
    if not champion_manifest_path.is_file():
        raise ValueError(
            "Current registry champion is missing run_manifest.json at "
            f"{champion_manifest_path}",
        )
    champion_manifest = read_json(champion_manifest_path)

    compatibility_checks = _compatibility_checks(challenger_manifest, champion_manifest)
    champion_metrics = dict(champion_manifest["winner"]["metrics"])

    reasons: list[str] = []
    if not all(compatibility_checks.values()):
        reasons.extend(_compatibility_failure_reasons(compatibility_checks))
    reasons.extend(_baseline_failure_reasons(baseline_checks))

    if not _is_primary_metric_improved(challenger_metrics, champion_metrics):
        reasons.append(
            "Primary metric mean_long_only_net_value_proxy did not beat the current "
            "champion under the deterministic tie-break rules",
        )

    accuracy_delta = (
        float(challenger_metrics["directional_accuracy"])
        - float(champion_metrics["directional_accuracy"])
    )
    if accuracy_delta < -workflow.comparison_policy.max_directional_accuracy_regression:
        reasons.append(
            "Directional accuracy regressed beyond the configured tolerance "
            f"({accuracy_delta:.6f})",
        )

    brier_delta = float(challenger_metrics["brier_score"]) - float(
        champion_metrics["brier_score"],
    )
    if brier_delta > workflow.comparison_policy.max_brier_score_worsening:
        reasons.append(
            "Brier score worsened beyond the configured tolerance "
            f"({brier_delta:.6f})",
        )

    passed = not reasons
    return {
        "generated_at": to_rfc3339(utc_now()),
        "passed": passed,
        "decision": "promote" if passed else "reject",
        "policy": workflow.comparison_policy.to_dict(),
        "challenger": _comparison_side(challenger_manifest, model_version=None),
        "champion": _comparison_side(
            champion_manifest,
            model_version=str(current_entry["model_version"]),
        ),
        "compatibility_checks": compatibility_checks,
        "baseline_checks": baseline_checks,
        "metric_deltas": {
            _PRIMARY_METRIC: float(challenger_metrics[_PRIMARY_METRIC])
            - float(champion_metrics[_PRIMARY_METRIC]),
            "directional_accuracy": accuracy_delta,
            "brier_score": brier_delta,
        },
        "reasons": reasons,
    }


def write_comparison_artifact(run_dir: Path, payload: dict[str, Any]) -> Path:
    """Persist the comparison payload inside the run directory."""
    comparison_path = Path(run_dir).resolve() / "comparison_vs_champion.json"
    write_json_atomic(comparison_path, payload)
    return comparison_path


def main() -> None:
    """Compare one run directory against the current champion and save the result."""
    parser = argparse.ArgumentParser(description="Compare an M7 challenger to the champion")
    parser.add_argument("--config", required=True, help="Path to configs/training.m7.json")
    parser.add_argument("--run-dir", required=True, help="Path to the challenger run directory")
    arguments = parser.parse_args()

    try:
        payload = compare_run_to_current(
            run_dir=Path(arguments.run_dir),
            config_path=Path(arguments.config),
        )
        comparison_path = write_comparison_artifact(Path(arguments.run_dir), payload)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    print(str(comparison_path))


def _no_champion_payload(
    challenger_manifest: dict[str, Any],
    policy: ComparisonPolicy,
    *,
    baseline_checks: dict[str, Any],
) -> dict[str, Any]:
    """Return the explicit bootstrap comparison payload when no champion exists yet."""
    reasons = _baseline_failure_reasons(baseline_checks)
    if not reasons:
        reasons = [
            "No current champion is registered; bootstrap is allowed because the "
            "challenger is positive after costs and beats the required baselines.",
        ]
    passed = not _baseline_failure_reasons(baseline_checks)
    return {
        "generated_at": to_rfc3339(utc_now()),
        "passed": passed,
        "decision": "bootstrap_allowed" if passed else "reject",
        "policy": policy.to_dict(),
        "challenger": _comparison_side(challenger_manifest, model_version=None),
        "champion": None,
        "compatibility_checks": {
            "feature_schema_match": True,
            "evaluation_protocol_match": True,
            "selection_rule_match": True,
        },
        "baseline_checks": baseline_checks,
        "metric_deltas": None,
        "reasons": reasons,
    }


def _comparison_side(
    manifest: dict[str, Any],
    *,
    model_version: str | None,
) -> dict[str, Any]:
    """Return the comparison metadata for one model side."""
    return {
        "model_version": model_version,
        "run_id": manifest["run_id"],
        "run_dir": manifest["run_dir"],
        "model_name": manifest["winner"]["model_name"],
        "trained_at": manifest["winner"]["trained_at"],
        "training_config": manifest["winner"].get("training_config"),
        "economics_contract": dict(manifest.get("economics_contract", {})),
        "acceptance": dict(manifest.get("acceptance", {})),
        "metrics": manifest["winner"]["metrics"],
    }


def _compatibility_checks(
    challenger_manifest: dict[str, Any],
    champion_manifest: dict[str, Any],
) -> dict[str, bool]:
    """Return explicit feature-schema and evaluation-protocol compatibility checks."""
    return {
        "feature_schema_match": challenger_manifest["feature_schema"]
        == champion_manifest["feature_schema"],
        "evaluation_protocol_match": challenger_manifest["evaluation_protocol"]
        == champion_manifest["evaluation_protocol"],
        "selection_rule_match": challenger_manifest["winner"]["selection_rule"]
        == champion_manifest["winner"]["selection_rule"],
    }


def _compatibility_failure_reasons(checks: dict[str, bool]) -> list[str]:
    """Translate compatibility failures into readable rejection reasons."""
    reasons: list[str] = []
    if not checks["feature_schema_match"]:
        reasons.append("Feature schema assumptions differ between challenger and champion")
    if not checks["evaluation_protocol_match"]:
        reasons.append("Evaluation protocol differs between challenger and champion")
    if not checks["selection_rule_match"]:
        reasons.append("Winner selection rule differs between challenger and champion")
    return reasons


def _is_primary_metric_improved(
    challenger_metrics: dict[str, Any],
    champion_metrics: dict[str, Any],
) -> bool:
    """Apply the deterministic primary metric plus tie-break ordering."""
    challenger_net = float(challenger_metrics[_PRIMARY_METRIC])
    champion_net = float(champion_metrics[_PRIMARY_METRIC])
    if challenger_net > champion_net + _EPSILON:
        return True
    if challenger_net < champion_net - _EPSILON:
        return False

    challenger_accuracy = float(challenger_metrics["directional_accuracy"])
    champion_accuracy = float(champion_metrics["directional_accuracy"])
    if challenger_accuracy > champion_accuracy + _EPSILON:
        return True
    if challenger_accuracy < champion_accuracy - _EPSILON:
        return False

    challenger_brier = float(challenger_metrics["brier_score"])
    champion_brier = float(champion_metrics["brier_score"])
    return challenger_brier < champion_brier - _EPSILON


def _load_summary_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    summary_path = Path(str(manifest["artifact_files"]["summary.json"])).resolve()
    if not summary_path.is_file():
        raise ValueError(f"summary.json is missing from {summary_path}")
    return read_json(summary_path)


def _baseline_checks(
    challenger_summary: dict[str, Any],
    *,
    challenger_metrics: dict[str, Any],
    primary_metric: str,
) -> dict[str, Any]:
    baseline_results: dict[str, dict[str, Any]] = {}
    challenger_value = float(challenger_metrics[primary_metric])
    for baseline_name in _REQUIRED_PROMOTION_BASELINES:
        try:
            baseline_value = float(challenger_summary["models"][baseline_name][primary_metric])
        except KeyError as error:
            raise ValueError(
                f"Training summary is missing required baseline metric {primary_metric} "
                f"for {baseline_name}"
            ) from error
        baseline_results[baseline_name] = {
            "metric_name": primary_metric,
            "challenger_value": challenger_value,
            "baseline_value": baseline_value,
            "passed": challenger_value > baseline_value + _EPSILON,
        }
    return {
        "winner_after_cost_positive": challenger_value > _EPSILON,
        "required_baselines": baseline_results,
    }


def _baseline_failure_reasons(baseline_checks: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not bool(baseline_checks["winner_after_cost_positive"]):
        reasons.append(
            "Challenger mean_long_only_net_value_proxy is not positive after costs",
        )
    for baseline_name, details in baseline_checks["required_baselines"].items():
        if bool(details["passed"]):
            continue
        reasons.append(
            "Challenger mean_long_only_net_value_proxy did not beat baseline "
            f"{baseline_name} after costs",
        )
    return reasons


def _is_legacy_archived_current_entry(entry: dict[str, Any]) -> bool:
    """Treat archived sklearn champions as non-authoritative bootstrap state."""
    return str(entry.get("model_name", "")).strip() in LEGACY_ARCHIVED_MODEL_NAMES


if __name__ == "__main__":
    main()
