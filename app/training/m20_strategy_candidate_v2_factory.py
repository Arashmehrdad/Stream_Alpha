"""Generic research-only M20 v2 strategy candidate factory."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_OUTPUT_NAME = "strategy_candidate_v2_factory"
DEFAULT_REDESIGN_PLAN_NAME = "strategy_candidate_redesign_plan"
DEFAULT_ECONOMIC_OUTCOME_NAME = "economic_outcome_artifacts"
DEFAULT_TRAINING_FRAME_DIR = "training_frame"
FEATURE_FILE = "m20_training_frame_features.csv"
FEATURE_COLUMNS_FILE = "m20_training_frame_feature_columns.json"
RESEARCH_FEATURE_FILE = "research_features.csv"
RESEARCH_FEATURE_COLUMNS_FILE = "research_feature_columns.json"
RESEARCH_BLOCKED_FEATURES_FILE = "blocked_features.csv"
LABEL_FILE = "research_labels/vol_scaled/fee_exceedance_labels_vol_scaled.csv"
TRIPLE_BARRIER_FILE = "research_labels/vol_scaled/triple_barrier_labels_vol_scaled.csv"
ECONOMIC_OUTCOME_FILE = "economic_outcomes.csv"
DEFINITION_SPECS_FILE = "candidate_definition_specs.csv"
HONESTY_FLAGS = (
    "RESEARCH_ONLY",
    "NO_RUNTIME_EFFECT",
    "NOT_BACKTEST",
    "NOT_RUNTIME_READY",
    "NOT_PROMOTABLE",
    "NO_PROFIT_CLAIM",
)
READY_STATUS = "READY_FOR_V2_FACTORY"
BLOCKED_STATUS = "BLOCKED_MISSING_FEATURES"
LOW_SAMPLE_MIN_ROWS = 100


def build_m20_strategy_candidates_v2(
    *,
    source_run_dir: Path,
    redesign_plan_dir: Path | None = None,
    economic_outcome_dir: Path | None = None,
    training_frame_dir: Path | None = None,
    research_feature_dir: Path | None = None,
    label_source_run_dir: Path | None = None,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    """Build v2 strategy candidate artifacts from redesign definitions."""
    # pylint: disable=too-many-arguments,too-many-locals,too-many-statements
    source_dir = Path(source_run_dir).resolve()
    vol_scaled_dir = source_dir / "research_labels" / "vol_scaled"
    label_dir = Path(label_source_run_dir).resolve() if label_source_run_dir else source_dir
    frame_dir = (
        Path(training_frame_dir).resolve()
        if training_frame_dir is not None
        else source_dir / DEFAULT_TRAINING_FRAME_DIR
    )
    feature_source = _feature_source(frame_dir, research_feature_dir)
    plan_dir = (
        Path(redesign_plan_dir).resolve()
        if redesign_plan_dir is not None
        else vol_scaled_dir / DEFAULT_REDESIGN_PLAN_NAME
    )
    outcome_dir = (
        Path(economic_outcome_dir).resolve()
        if economic_outcome_dir is not None
        else vol_scaled_dir / DEFAULT_ECONOMIC_OUTCOME_NAME
    )
    output_dir = vol_scaled_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = _output_files(output_dir)

    feature_path = Path(feature_source["feature_path"])
    definition_path = plan_dir / DEFINITION_SPECS_FILE
    if not feature_path.exists():
        raise ValueError(f"Missing training frame features: {feature_path}")
    if not definition_path.exists():
        raise ValueError(f"Missing v2 candidate definitions: {definition_path}")

    feature_rows = _read_csv(feature_path)
    feature_columns = _feature_columns(
        Path(feature_source["feature_columns_path"]),
        feature_rows,
    )
    definitions = _read_csv(definition_path)
    unavailable_features = _blocked_research_features(feature_source["blocked_features_path"])
    contexts = _contexts(feature_rows)
    outcome_index = _outcome_index(outcome_dir / ECONOMIC_OUTCOME_FILE)
    label_index = _label_index(label_dir / LABEL_FILE)
    triple_barrier_index = _triple_barrier_index(label_dir / TRIPLE_BARRIER_FILE)
    economics_available = bool(outcome_index)
    labels_available = bool(label_index)
    base_positive_rate = _base_positive_rate(feature_rows, label_index)

    definition_audit = _definition_audit(definitions, feature_columns, unavailable_features)
    blocked_definitions = [
        row for row in definition_audit if row["definition_status"] == BLOCKED_STATUS
    ]
    candidate_rows: list[dict[str, Any]] = []
    candidate_metrics: list[dict[str, Any]] = []
    by_symbol: list[dict[str, Any]] = []
    by_time: list[dict[str, Any]] = []

    for definition in definition_audit:
        if definition["definition_status"] != READY_STATUS:
            candidate_metrics.append(_blocked_metric(definition))
            continue
        predicate = _predicate_for(definition)
        selected = [
            _candidate_row(
                feature_row,
                definition,
                contexts,
                outcome_index,
                label_index,
                triple_barrier_index,
            )
            for feature_row in feature_rows
            if predicate(feature_row, contexts[_key(feature_row)])
        ]
        candidate_rows.extend(selected)
        metric = _candidate_metric(
            definition=definition,
            selected_rows=selected,
            total_rows=len(feature_rows),
            labels_available=labels_available,
            economics_available=economics_available,
            base_positive_rate=base_positive_rate,
        )
        candidate_metrics.append(metric)
        by_symbol.extend(
            _slice_metrics(
                metric,
                selected,
                "symbol",
                labels_available,
                economics_available,
                base_positive_rate,
            )
        )
        by_time.extend(
            _slice_metrics(
                metric,
                selected,
                "month",
                labels_available,
                economics_available,
                base_positive_rate,
            )
        )
        by_time.extend(
            _slice_metrics(
                metric,
                selected,
                "quarter",
                labels_available,
                economics_available,
                base_positive_rate,
            )
        )

    decisions = _candidate_decisions(candidate_metrics)
    recommendation = _recommendation(decisions)
    manifest = {
        "source_run_dir": str(source_dir),
        "training_frame_dir": str(frame_dir),
        "research_feature_dir": feature_source["research_feature_dir"],
        "source_feature_mode": feature_source["source_feature_mode"],
        "redesign_plan_dir": str(plan_dir),
        "economic_outcome_dir": str(outcome_dir),
        "label_source_run_dir": str(label_dir),
        "feature_path": str(feature_path),
        "feature_columns_path": feature_source["feature_columns_path"],
        "blocked_features_path": feature_source["blocked_features_path"],
        "definition_path": str(definition_path),
        "feature_columns": list(feature_columns),
        "economics_available": economics_available,
        "labels_available": labels_available,
        "candidate_count": len(candidate_metrics),
        "candidate_event_rows": len(candidate_rows),
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }
    report = {
        "summary": "Generic M20 v2 strategy candidate factory.",
        "source_feature_mode": feature_source["source_feature_mode"],
        "feature_path": str(feature_path),
        "research_feature_dir": feature_source["research_feature_dir"],
        "candidate_count": len(candidate_metrics),
        "ready_definition_count": sum(
            1 for row in definition_audit if row["definition_status"] == READY_STATUS
        ),
        "blocked_definition_count": len(blocked_definitions),
        "candidate_event_rows": len(candidate_rows),
        "recommendation": recommendation["recommendation"],
        "next_required_action": recommendation["next_required_action"],
        "overall_status": list(HONESTY_FLAGS),
        "runtime_status": "NO_RUNTIME_EFFECT",
        "promotion_status": "NOT_PROMOTABLE",
        "profitability_status": "NO_PROFIT_CLAIM",
        "honesty_flags": list(HONESTY_FLAGS),
        "output_files": output_files,
    }

    write_json_artifact(Path(output_files["manifest_json"]), manifest)
    write_json_artifact(Path(output_files["strategy_candidate_v2_report_json"]), report)
    Path(output_files["strategy_candidate_v2_report_md"]).write_text(
        _markdown(report, decisions),
        encoding="utf-8",
    )
    write_csv_artifact(Path(output_files["strategy_candidates_v2_csv"]), candidate_rows)
    write_csv_artifact(Path(output_files["candidate_metrics_csv"]), candidate_metrics)
    write_csv_artifact(Path(output_files["by_symbol_csv"]), by_symbol)
    write_csv_artifact(Path(output_files["by_time_csv"]), by_time)
    write_csv_artifact(Path(output_files["definition_audit_csv"]), definition_audit)
    write_csv_artifact(Path(output_files["blocked_definitions_csv"]), blocked_definitions)
    write_csv_artifact(Path(output_files["candidate_decisions_csv"]), decisions)
    write_csv_artifact(Path(output_files["next_actions_csv"]), _next_actions(recommendation))
    write_json_artifact(Path(output_files["recommendation_json"]), recommendation)
    return make_json_safe(
        {
            **report,
            "manifest": manifest,
            "definition_audit": definition_audit,
            "candidate_metrics": candidate_metrics,
            "candidate_decisions": decisions,
            "recommendation_payload": recommendation,
        }
    )


def _contexts(rows: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, str], dict[str, float]]:
    return {
        _key(row): {
            "range_ratio": _range_ratio(row),
            "abs_close_zscore": abs(_to_float(row.get("close_zscore_12"))),
            "abs_momentum": abs(_to_float(row.get("momentum_3"))),
            "abs_macd": abs(_to_float(row.get("macd_line_12_26"))),
        }
        for row in rows
    }


def _feature_source(
    frame_dir: Path,
    research_feature_dir: Path | None,
) -> dict[str, str]:
    if research_feature_dir is None:
        return {
            "source_feature_mode": "training_frame",
            "research_feature_dir": "",
            "feature_path": str(frame_dir / FEATURE_FILE),
            "feature_columns_path": str(frame_dir / FEATURE_COLUMNS_FILE),
            "blocked_features_path": "",
        }
    feature_dir = Path(research_feature_dir).resolve()
    return {
        "source_feature_mode": "research_feature_enrichment",
        "research_feature_dir": str(feature_dir),
        "feature_path": str(feature_dir / RESEARCH_FEATURE_FILE),
        "feature_columns_path": str(feature_dir / RESEARCH_FEATURE_COLUMNS_FILE),
        "blocked_features_path": str(feature_dir / RESEARCH_BLOCKED_FEATURES_FILE),
    }


def _definition_audit(
    definitions: Sequence[Mapping[str, str]],
    feature_columns: Sequence[str],
    unavailable_features: set[str],
) -> list[dict[str, Any]]:
    columns = set(feature_columns)
    output = []
    for definition in definitions:
        required = _split_pipe(definition.get("required_features", ""))
        missing = [
            column for column in required
            if column not in columns or column in unavailable_features
        ]
        source_status = definition.get("definition_status", "")
        status = READY_STATUS if not missing else BLOCKED_STATUS
        output.append(
            {
                "redesign_family": definition.get("redesign_family", ""),
                "candidate_name": definition.get("candidate_name", ""),
                "candidate_version": definition.get("candidate_version", "v2_design"),
                "required_features": "|".join(required),
                "missing_features": "|".join(missing),
                "source_definition_status": source_status,
                "rechecked_with_active_feature_source": True,
                "definition_status": status,
            }
        )
    return output


def _predicate_for(
    definition: Mapping[str, Any],
) -> Callable[[Mapping[str, str], Mapping[str, float]], bool]:
    predicates = {
        "momentum_volume_confirmed": _momentum_volume_confirmed,
        "macd_volatility_adjusted_direction": _macd_volatility_adjusted_direction,
        "range_expansion_volume_confirmed": _range_expansion_volume_confirmed,
        "high_agreement_low_turnover_setup": _high_agreement_low_turnover_setup,
        "tail_risk_avoidance_context": _tail_risk_avoidance_context,
        "non_extreme_volatility_momentum": _non_extreme_volatility_momentum,
        "regime_conditioned_momentum": _regime_conditioned_momentum,
        "trend_strength_filtered_momentum": _trend_strength_filtered_momentum,
    }
    return predicates.get(str(definition["candidate_name"]), lambda _row, _context: False)


def _momentum_volume_confirmed(
    row: Mapping[str, str],
    _context: Mapping[str, float],
) -> bool:
    return (
        _to_float(row.get("momentum_3")) > 0.0
        and _to_float(row.get("volume_zscore_12")) > 0.5
        and 0.003 <= _to_float(row.get("realized_vol_12")) <= 0.04
    )


def _macd_volatility_adjusted_direction(
    row: Mapping[str, str],
    context: Mapping[str, float],
) -> bool:
    close = max(abs(_to_float(row.get("close_price"))), 1.0)
    normalized_macd = context["abs_macd"] / close
    return (
        _to_float(row.get("macd_line_12_26")) > 0.0
        and normalized_macd >= _to_float(row.get("return_std_12"))
        and _to_float(row.get("realized_vol_12")) <= 0.04
    )


def _range_expansion_volume_confirmed(
    row: Mapping[str, str],
    context: Mapping[str, float],
) -> bool:
    return (
        context["range_ratio"] >= 0.004
        and _to_float(row.get("volume_zscore_12")) > 0.5
        and _to_float(row.get("log_return_1")) > 0.0
    )


def _high_agreement_low_turnover_setup(
    row: Mapping[str, str],
    _context: Mapping[str, float],
) -> bool:
    rsi = _to_float(row.get("rsi_14"))
    return (
        _to_float(row.get("macd_line_12_26")) > 0.0
        and _to_float(row.get("momentum_3")) > 0.0
        and 45.0 <= rsi <= 70.0
        and _to_float(row.get("volume_zscore_12")) > 0.5
    )


def _tail_risk_avoidance_context(
    row: Mapping[str, str],
    context: Mapping[str, float],
) -> bool:
    return (
        _to_float(row.get("realized_vol_12")) > 0.025
        or _to_float(row.get("return_std_12")) > 0.01
        or context["abs_close_zscore"] > 2.0
        or abs(_to_float(row.get("volume_zscore_12"))) > 2.0
    )


def _non_extreme_volatility_momentum(
    row: Mapping[str, str],
    context: Mapping[str, float],
) -> bool:
    return (
        _to_float(row.get("momentum_3")) > 0.0
        and 0.004 <= _to_float(row.get("realized_vol_12")) <= 0.025
        and context["abs_close_zscore"] <= 1.5
    )


def _regime_conditioned_momentum(
    row: Mapping[str, str],
    _context: Mapping[str, float],
) -> bool:
    return (
        row.get("regime_label") == "TREND_UP"
        and _to_float(row.get("momentum_3")) > 0.0
        and _to_float(row.get("realized_vol_12")) <= 0.04
    )


def _trend_strength_filtered_momentum(
    row: Mapping[str, str],
    _context: Mapping[str, float],
) -> bool:
    return (
        _to_float(row.get("adx_14")) >= 20.0
        and _to_float(row.get("momentum_3")) > 0.0
        and _to_float(row.get("realized_vol_12")) <= 0.04
    )


def _candidate_row(
    feature_row: Mapping[str, str],
    definition: Mapping[str, Any],
    contexts: Mapping[tuple[str, str, str], Mapping[str, float]],
    outcome_index: Mapping[tuple[str, str, str], Mapping[str, str]],
    label_index: Mapping[tuple[str, str, str], Mapping[str, str]],
    triple_barrier_index: Mapping[tuple[str, str, str], str],
) -> dict[str, Any]:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    key = _key(feature_row)
    outcome = outcome_index.get(key, {})
    label = label_index.get(key, {})
    feature_columns = _split_pipe(str(definition["required_features"]))
    return {
        "strategy_family": definition["redesign_family"],
        "candidate_name": definition["candidate_name"],
        "candidate_version": definition["candidate_version"],
        "symbol": feature_row.get("symbol", ""),
        "interval_begin": feature_row.get("interval_begin", ""),
        "fold_index": feature_row.get("fold_index", ""),
        "setup_passed": True,
        "feature_snapshot_json": json.dumps(
            {column: feature_row.get(column, "") for column in feature_columns},
            sort_keys=True,
        ),
        "context_snapshot_json": json.dumps(contexts.get(key, {}), sort_keys=True),
        "fee_exceedance_label": outcome.get("fee_exceedance_label", label.get("label", "")),
        "triple_barrier_label": outcome.get(
            "triple_barrier_label",
            triple_barrier_index.get(key, ""),
        ),
        "gross_value_proxy": outcome.get("gross_value_proxy", ""),
        "net_value_proxy": outcome.get("net_value_proxy", ""),
    }


def _candidate_metric(
    *,
    definition: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
    total_rows: int,
    labels_available: bool,
    economics_available: bool,
    base_positive_rate: float,
) -> dict[str, Any]:
    # pylint: disable=too-many-arguments
    selected_count = len(selected_rows)
    positive_count = sum(_to_int(row.get("fee_exceedance_label")) for row in selected_rows)
    selected_rate = positive_count / selected_count if selected_count else 0.0
    economics = _economic_metrics(selected_rows) if economics_available else _empty_economics()
    classification = _classify_candidate(selected_count, economics_available, economics)
    return {
        "strategy_family": definition["redesign_family"],
        "candidate_name": definition["candidate_name"],
        "candidate_version": definition["candidate_version"],
        "total_rows": total_rows,
        "selected_rows": selected_count,
        "coverage": selected_count / total_rows if total_rows else 0.0,
        "base_positive_rate": base_positive_rate if labels_available else "",
        "selected_positive_rate": selected_rate if labels_available else "",
        "lift_vs_base": (
            selected_rate / base_positive_rate
            if labels_available and base_positive_rate > 0
            else ""
        ),
        "classification": classification,
        **economics,
    }


def _slice_metrics(
    base_metric: Mapping[str, Any],
    selected_rows: Sequence[Mapping[str, Any]],
    slice_family: str,
    labels_available: bool,
    economics_available: bool,
    base_positive_rate: float,
) -> list[dict[str, Any]]:
    # pylint: disable=too-many-arguments,too-many-positional-arguments
    output = []
    for value, rows in sorted(_group_by(selected_rows, slice_family).items()):
        metric = _candidate_metric(
            definition={
                "redesign_family": base_metric["strategy_family"],
                "candidate_name": base_metric["candidate_name"],
                "candidate_version": base_metric["candidate_version"],
            },
            selected_rows=rows,
            total_rows=len(selected_rows),
            labels_available=labels_available,
            economics_available=economics_available,
            base_positive_rate=base_positive_rate,
        )
        output.append({"slice_family": slice_family, "slice_value": value, **metric})
    return output


def _candidate_decisions(metrics: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "strategy_family": row["strategy_family"],
            "candidate_name": row["candidate_name"],
            "candidate_version": row["candidate_version"],
            "candidate_decision": row["classification"],
            "selected_rows": row["selected_rows"],
            "coverage": row["coverage"],
            "mean_net_proxy": row["mean_net_proxy"],
            "cumulative_net_proxy": row["cumulative_net_proxy"],
            "runtime_status": "NO_RUNTIME_EFFECT",
            "promotion_status": "NOT_PROMOTABLE",
            "profitability_status": "NO_PROFIT_CLAIM",
        }
        for row in metrics
    ]


def _blocked_metric(definition: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "strategy_family": definition["redesign_family"],
        "candidate_name": definition["candidate_name"],
        "candidate_version": definition["candidate_version"],
        "total_rows": 0,
        "selected_rows": 0,
        "coverage": 0.0,
        "base_positive_rate": "",
        "selected_positive_rate": "",
        "lift_vs_base": "",
        "classification": "V2_STRATEGY_CANDIDATE_BLOCKED_MISSING_FEATURES",
        **_empty_economics(),
    }


def _classify_candidate(
    selected_count: int,
    economics_available: bool,
    economics: Mapping[str, Any],
) -> str:
    if selected_count == 0:
        return "V2_STRATEGY_CANDIDATE_INSUFFICIENT_EVIDENCE"
    if selected_count < LOW_SAMPLE_MIN_ROWS:
        return "V2_STRATEGY_CANDIDATE_LOW_SAMPLE"
    if not economics_available or economics.get("mean_net_proxy") in ("", None):
        return "V2_STRATEGY_CANDIDATE_INSUFFICIENT_EVIDENCE"
    if float(economics["mean_net_proxy"]) > 0.0:
        return "V2_STRATEGY_CANDIDATE_RESEARCH_WATCHLIST_POSITIVE_PROXY"
    return "V2_STRATEGY_CANDIDATE_ECONOMICS_NEGATIVE"


def _recommendation(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ready = [
        row for row in decisions
        if row["candidate_decision"] != "V2_STRATEGY_CANDIDATE_BLOCKED_MISSING_FEATURES"
    ]
    has_positive = any(
        row["candidate_decision"]
        == "V2_STRATEGY_CANDIDATE_RESEARCH_WATCHLIST_POSITIVE_PROXY"
        for row in decisions
    )
    if has_positive:
        recommendation = "EVALUATE_V2_CANDIDATES_WITH_GENERIC_REFINEMENT_PIPELINE"
    elif ready:
        recommendation = "REFINE_OR_ADD_SAFE_FEATURES_FOR_V2_CANDIDATES"
    else:
        recommendation = "ADD_REQUIRED_FEATURES_BEFORE_V2_CANDIDATES"
    return {
        "recommendation": recommendation,
        "next_required_action": recommendation,
        "runtime_ready": False,
        "promotable": False,
        "profitability_claim": False,
        "honesty_flags": list(HONESTY_FLAGS),
    }


def _next_actions(recommendation: Mapping[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "priority": "1",
            "action": str(recommendation["next_required_action"]),
            "rationale": "Continue through generic v2 research tooling only.",
        },
        {
            "priority": "2",
            "action": "KEEP_V2_CANDIDATES_RESEARCH_ONLY",
            "rationale": "No runtime, registry, promotion, backtest, trading, or profit claim.",
        },
    ]


def _economic_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    net_values = [
        _to_float(row.get("net_value_proxy")) for row in rows
        if row.get("net_value_proxy") not in ("", None)
    ]
    if not net_values:
        return _empty_economics()
    return {
        "mean_net_proxy": _mean(net_values),
        "cumulative_net_proxy": sum(net_values),
        "max_drawdown_proxy": _max_drawdown(net_values),
        "win_rate_proxy": sum(1 for value in net_values if value > 0.0) / len(net_values),
    }


def _empty_economics() -> dict[str, Any]:
    return {
        "mean_net_proxy": "",
        "cumulative_net_proxy": "",
        "max_drawdown_proxy": "",
        "win_rate_proxy": "",
    }


def _output_files(output_dir: Path) -> dict[str, str]:
    return {
        "manifest_json": str(output_dir / "manifest.json"),
        "strategy_candidate_v2_report_json": str(
            output_dir / "strategy_candidate_v2_report.json"
        ),
        "strategy_candidate_v2_report_md": str(
            output_dir / "strategy_candidate_v2_report.md"
        ),
        "strategy_candidates_v2_csv": str(output_dir / "strategy_candidates_v2.csv"),
        "candidate_metrics_csv": str(output_dir / "candidate_metrics.csv"),
        "by_symbol_csv": str(output_dir / "by_symbol.csv"),
        "by_time_csv": str(output_dir / "by_time.csv"),
        "definition_audit_csv": str(output_dir / "definition_audit.csv"),
        "blocked_definitions_csv": str(output_dir / "blocked_definitions.csv"),
        "candidate_decisions_csv": str(output_dir / "candidate_decisions.csv"),
        "next_actions_csv": str(output_dir / "next_actions.csv"),
        "recommendation_json": str(output_dir / "recommendation.json"),
    }


def _markdown(report: Mapping[str, Any], decisions: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# M20 Strategy Candidate V2 Factory",
        "",
        f"- Recommendation: `{report['recommendation']}`",
        f"- Next required action: `{report['next_required_action']}`",
        f"- Candidate count: `{report['candidate_count']}`",
        f"- Candidate event rows: `{report['candidate_event_rows']}`",
        "- Status: `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, "
        "`NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`",
        "",
        "## Candidate Decisions",
    ]
    for row in decisions:
        lines.append(
            f"- `{row['strategy_family']}:{row['candidate_name']}` -> "
            f"`{row['candidate_decision']}`"
        )
    lines.append("")
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _feature_columns(path: Path, rows: Sequence[Mapping[str, str]]) -> list[str]:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return list(payload.get("feature_columns", []))
    if not rows:
        return []
    return [
        column for column in rows[0]
        if column not in ("symbol", "interval_begin", "fold_index", "row_id")
    ]


def _blocked_research_features(path_value: str) -> set[str]:
    if not path_value:
        return set()
    path = Path(path_value)
    if not path.exists():
        return set()
    return {
        row.get("feature_name", "")
        for row in _read_csv(path)
        if row.get("feature_name", "")
    }


def _outcome_index(path: Path) -> dict[tuple[str, str, str], Mapping[str, str]]:
    if not path.exists():
        return {}
    return {_key(row): row for row in _read_csv(path)}


def _label_index(path: Path) -> dict[tuple[str, str, str], Mapping[str, str]]:
    if not path.exists():
        return {}
    return {
        _key(row): row for row in _read_csv(path)
        if row.get("scenario_name", "current_fee") == "current_fee"
    }


def _triple_barrier_index(path: Path) -> dict[tuple[str, str, str], str]:
    if not path.exists():
        return {}
    return {_key(row): row.get("label", "") for row in _read_csv(path)}


def _base_positive_rate(
    feature_rows: Sequence[Mapping[str, str]],
    label_index: Mapping[tuple[str, str, str], Mapping[str, str]],
) -> float:
    labels = []
    for row in feature_rows:
        label = label_index.get(_key(row))
        if label is not None:
            labels.append(_to_int(label.get("label")))
    return sum(labels) / len(labels) if labels else 0.0


def _group_by(
    rows: Sequence[Mapping[str, Any]],
    slice_family: str,
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if slice_family == "month":
            value = str(row.get("interval_begin", ""))[:7]
        elif slice_family == "quarter":
            value = _quarter(str(row.get("interval_begin", "")))
        else:
            value = str(row.get(slice_family, ""))
        grouped.setdefault(value, []).append(row)
    return grouped


def _key(row: Mapping[str, str]) -> tuple[str, str, str]:
    return (
        row.get("fold_index", ""),
        row.get("symbol", ""),
        row.get("interval_begin", ""),
    )


def _split_pipe(value: str) -> list[str]:
    return [part for part in value.split("|") if part]


def _range_ratio(row: Mapping[str, str]) -> float:
    close = _to_float(row.get("close_price"))
    if close == 0.0:
        return 0.0
    return (_to_float(row.get("high_price")) - _to_float(row.get("low_price"))) / close


def _quarter(timestamp: str) -> str:
    if len(timestamp) < 7:
        return ""
    try:
        month = int(timestamp[5:7])
    except ValueError:
        return ""
    return f"{timestamp[:4]}Q{((month - 1) // 3) + 1}"


def _max_drawdown(values: Sequence[float]) -> float:
    cumulative = 0.0
    peak = 0.0
    drawdown = 0.0
    for value in values:
        cumulative += value
        peak = max(peak, cumulative)
        drawdown = min(drawdown, cumulative - peak)
    return drawdown


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


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


__all__ = ["build_m20_strategy_candidates_v2"]
