"""Offline Packet 2 research and roster selection for the Stream Alpha ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.common.serialization import make_json_safe
from app.common.time import to_rfc3339, utc_now
from app.ensemble.config import default_ensemble_config_path, load_ensemble_config
from app.ensemble.schemas import (
    EnsembleEvaluationSliceMetrics,
    EnsembleResearchCandidate,
    EnsembleResearchResult,
    EnsembleRosterSelection,
)
from app.inference.service import load_model_artifact
from app.regime.config import load_regime_config
from app.regime.dataset import RegimeSourceRow
from app.regime.service import classify_row, fit_symbol_thresholds
from app.training.dataset import (
    DatasetSample,
    TrainingConfig,
    TrainingDataset,
    load_training_config,
    load_training_dataset,
)
from app.training.registry import (
    list_registry_entries,
    load_current_registry_entry,
    repo_root,
    write_json_atomic,
)


GENERALIST = "GENERALIST"
TREND_SPECIALIST = "TREND_SPECIALIST"
RANGE_SPECIALIST = "RANGE_SPECIALIST"

SLICE_ALL = "ALL"
SLICE_TREND_COMBINED = "TREND_COMBINED"
SLICE_RANGE = "RANGE"
SLICE_HIGH_VOL = "HIGH_VOL"

DEFAULT_SCOPE_REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL")
TREND_SCOPE_REGIMES = ("TREND_UP", "TREND_DOWN")
RANGE_SCOPE_REGIMES = ("RANGE",)
SPECIALIST_FAMILIES = {
    "NEURALFORECAST_NHITS",
    "NEURALFORECAST_NBEATSX",
    "NEURALFORECAST_TFT",
    "NEURALFORECAST_PATCHTST",
}


@dataclass(frozen=True, slots=True)
class EnsembleResearchRunOutput:
    """Written Packet 2 research bundle plus the selected runtime roster."""

    run_id: str
    artifact_dir: str
    report_path: str
    candidate_results: tuple[EnsembleResearchResult, ...]
    selection: EnsembleRosterSelection


@dataclass(frozen=True, slots=True)
class _EvaluatedRow:
    regime_label: str
    trade_taken: bool
    predicted_up: bool
    y_true: int
    net_return: float


def load_registry_research_candidates(
    *,
    registry_root: Path | None = None,
    include_current_generalist: bool = True,
) -> list[EnsembleResearchCandidate]:
    """Load Packet 2 research candidates from registry metadata and current baseline."""
    candidates: dict[tuple[str, str], EnsembleResearchCandidate] = {}

    for entry in list_registry_entries(registry_root=registry_root):
        metadata = dict(entry.get("metadata") or {})
        model_family = str(metadata.get("model_family", "")).strip()
        candidate_role = str(metadata.get("candidate_role", "")).strip()
        if not model_family or not candidate_role:
            continue
        resolved_scope_regimes = metadata.get("scope_regimes") or _default_scope_regimes(
            candidate_role
        )
        candidate = EnsembleResearchCandidate(
            model_version=str(entry["model_version"]),
            model_name=str(entry["model_name"]),
            model_family=model_family,
            candidate_role=candidate_role,
            artifact_path=str(entry["model_artifact_path"]),
            trained_at=str(entry["trained_at"]),
            scope_regimes=list(resolved_scope_regimes),
            entry_metadata=metadata,
        )
        candidates[(candidate.model_version, candidate.model_family)] = candidate

    if include_current_generalist:
        current_entry = load_current_registry_entry(registry_root)
        if current_entry is not None:
            metadata = dict(current_entry.get("metadata") or {})
            current_candidate = EnsembleResearchCandidate(
                model_version=str(current_entry["model_version"]),
                model_name=str(current_entry["model_name"]),
                model_family="REGISTRY_CHAMPION_BASELINE",
                candidate_role=str(
                    metadata.get("candidate_role", GENERALIST)
                ),
                artifact_path=str(current_entry["model_artifact_path"]),
                trained_at=str(current_entry["trained_at"]),
                scope_regimes=list(
                    metadata.get("scope_regimes") or DEFAULT_SCOPE_REGIMES
                ),
                entry_metadata={
                    **metadata,
                    "source": "current_registry",
                },
            )
            candidates[(
                current_candidate.model_version,
                current_candidate.model_family,
            )] = current_candidate

    return sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate.candidate_role,
            candidate.model_family,
            candidate.model_version,
        ),
    )


def evaluate_registry_candidates(
    *,
    candidates: list[EnsembleResearchCandidate],
    dataset: TrainingDataset,
    regime_labels_by_row_id: dict[str, str],
    training_config: TrainingConfig,
    slippage_bps: float = 0.0,
) -> list[EnsembleResearchResult]:
    """Evaluate registry-backed candidates honestly across required Packet 2 slices."""
    results: list[EnsembleResearchResult] = []
    for candidate in candidates:
        results.append(
            _evaluate_one_candidate(
                candidate=candidate,
                dataset=dataset,
                regime_labels_by_row_id=regime_labels_by_row_id,
                training_config=training_config,
                slippage_bps=slippage_bps,
            )
        )
    return results


def select_runtime_roster(
    candidate_results: list[EnsembleResearchResult],
) -> EnsembleRosterSelection:
    """Select the canonical 3-role Packet 2 runtime roster."""
    generalists = [
        result
        for result in candidate_results
        if result.candidate.candidate_role == GENERALIST
        and result.candidate.model_family in {"AUTOGLUON", "REGISTRY_CHAMPION_BASELINE"}
    ]
    trend_specialists = [
        result
        for result in candidate_results
        if result.candidate.candidate_role == TREND_SPECIALIST
        and result.candidate.model_family in SPECIALIST_FAMILIES
    ]
    range_specialists = [
        result
        for result in candidate_results
        if result.candidate.candidate_role == RANGE_SPECIALIST
        and result.candidate.model_family in SPECIALIST_FAMILIES
    ]

    if not generalists:
        raise ValueError(
            "Packet 2 research did not find any eligible GENERALIST candidates"
        )
    if not trend_specialists:
        raise ValueError(
            "Packet 2 research did not find any eligible TREND_SPECIALIST candidates"
        )
    if not range_specialists:
        raise ValueError(
            "Packet 2 research did not find any eligible RANGE_SPECIALIST candidates"
        )

    incumbent_generalist = next(
        (
            result
            for result in generalists
            if result.candidate.model_family == "REGISTRY_CHAMPION_BASELINE"
        ),
        None,
    )
    autogluon_generalists = [
        result
        for result in generalists
        if result.candidate.model_family == "AUTOGLUON"
    ]
    if incumbent_generalist is not None and autogluon_generalists:
        best_autogluon = _select_best_result(autogluon_generalists, SLICE_ALL)
        selected_generalist = (
            best_autogluon
            if _result_key(best_autogluon, SLICE_ALL)
            > _result_key(incumbent_generalist, SLICE_ALL)
            else incumbent_generalist
        )
    else:
        selected_generalist = _select_best_result(generalists, SLICE_ALL)

    selected_trend = _select_best_result(trend_specialists, SLICE_TREND_COMBINED)
    selected_range = _select_best_result(range_specialists, SLICE_RANGE)
    evidence_summary_json = _build_evidence_summary(
        selected_generalist=selected_generalist,
        selected_trend=selected_trend,
        selected_range=selected_range,
        candidate_results=candidate_results,
    )
    return EnsembleRosterSelection(
        generalist=selected_generalist,
        trend_specialist=selected_trend,
        range_specialist=selected_range,
        evidence_summary_json=evidence_summary_json,
    )


def run_packet2_research(  # pylint: disable=too-many-locals
    *,
    registry_root: Path | None = None,
    training_config_path: Path | None = None,
    slippage_bps: float = 0.0,
) -> EnsembleResearchRunOutput:
    """Run Packet 2 research over local registry candidates and write report artifacts."""
    resolved_training_config_path = (
        Path("configs/training.m7.json")
        if training_config_path is None
        else Path(training_config_path)
    )
    training_config = load_training_config(resolved_training_config_path)
    dataset = load_training_dataset(training_config)
    regime_labels_by_row_id = build_regime_labels_by_row_id(
        samples=dataset.samples,
        symbols=training_config.symbols,
    )
    candidates = load_registry_research_candidates(registry_root=registry_root)
    candidate_results = evaluate_registry_candidates(
        candidates=candidates,
        dataset=dataset,
        regime_labels_by_row_id=regime_labels_by_row_id,
        training_config=training_config,
        slippage_bps=slippage_bps,
    )
    selection = select_runtime_roster(candidate_results)

    run_id = utc_now().strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = _research_artifact_root() / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    report_path = artifact_dir / "research_report.json"
    write_json_atomic(
        report_path,
        {
            "schema_version": "m20_packet2_research_v1",
            "generated_at": to_rfc3339(utc_now()),
            "run_id": run_id,
            "training_config_path": str(resolved_training_config_path.resolve()),
            "registry_root": str((repo_root() / "artifacts" / "registry").resolve())
            if registry_root is None
            else str(Path(registry_root).resolve()),
            "candidate_results": [
                result.model_dump(mode="json") for result in candidate_results
            ],
            "selection": selection.model_dump(mode="json"),
            "deferred_items": [
                (
                    "Weighted aggregate ensemble explainability remains deferred "
                    "when Packet 2 research runs offline."
                ),
            ],
        },
    )
    return EnsembleResearchRunOutput(
        run_id=run_id,
        artifact_dir=str(artifact_dir.resolve()),
        report_path=str(report_path.resolve()),
        candidate_results=tuple(candidate_results),
        selection=selection,
    )


def build_regime_labels_by_row_id(
    *,
    samples: tuple[DatasetSample, ...],
    symbols: tuple[str, ...],
) -> dict[str, str]:
    """Classify training samples into deterministic M8 regime slices."""
    regime_config = load_regime_config(Path("configs/regime.m8.json"))
    rows = [_sample_to_regime_row(sample) for sample in samples]
    thresholds_by_symbol = fit_symbol_thresholds(rows, regime_config)
    return {
        sample.row_id: classify_row(
            _sample_to_regime_row(sample),
            thresholds_by_symbol,
        )
        for sample in samples
        if sample.symbol in symbols
    }


def _evaluate_one_candidate(  # pylint: disable=too-many-locals
    *,
    candidate: EnsembleResearchCandidate,
    dataset: TrainingDataset,
    regime_labels_by_row_id: dict[str, str],
    training_config: TrainingConfig,
    slippage_bps: float,
) -> EnsembleResearchResult:
    model_artifact = load_model_artifact(candidate.artifact_path)
    cost_rate = training_config.round_trip_fee_rate + (slippage_bps / 10_000.0)
    evaluated_rows: list[_EvaluatedRow] = []

    for sample in dataset.samples:
        regime_label = regime_labels_by_row_id[sample.row_id]
        feature_input = _sample_feature_input(sample, model_artifact.feature_columns)
        probabilities = model_artifact.model.predict_proba([feature_input])
        if len(probabilities) != 1 or len(probabilities[0]) != 2:
            raise ValueError(
                f"Candidate {candidate.model_version} must return binary probabilities",
            )
        prob_up = float(probabilities[0][1])
        predicted_up = prob_up >= 0.5
        trade_taken = predicted_up
        net_return = 0.0
        if trade_taken:
            net_return = float(sample.future_return_3) - cost_rate
        evaluated_rows.append(
            _EvaluatedRow(
                regime_label=regime_label,
                trade_taken=trade_taken,
                predicted_up=predicted_up,
                y_true=int(sample.label),
                net_return=net_return,
            )
        )

    metrics_by_slice = {
        SLICE_ALL: _compute_slice_metrics(SLICE_ALL, evaluated_rows),
        SLICE_TREND_COMBINED: _compute_slice_metrics(
            SLICE_TREND_COMBINED,
            [row for row in evaluated_rows if row.regime_label in TREND_SCOPE_REGIMES],
        ),
        SLICE_RANGE: _compute_slice_metrics(
            SLICE_RANGE,
            [row for row in evaluated_rows if row.regime_label == "RANGE"],
        ),
        SLICE_HIGH_VOL: _compute_slice_metrics(
            SLICE_HIGH_VOL,
            [row for row in evaluated_rows if row.regime_label == "HIGH_VOL"],
        ),
    }
    primary_slice = _primary_slice_for_role(candidate.candidate_role)
    return EnsembleResearchResult(
        candidate=candidate,
        metrics_by_slice=metrics_by_slice,
        primary_slice=primary_slice,
        primary_metric_value=metrics_by_slice[primary_slice].net_pnl_after_fees_slippage,
    )


def _compute_slice_metrics(
    slice_label: str,
    rows: list[_EvaluatedRow],
) -> EnsembleEvaluationSliceMetrics:
    if not rows:
        return EnsembleEvaluationSliceMetrics(
            slice_label=slice_label,
            net_pnl_after_fees_slippage=0.0,
            max_drawdown=0.0,
            calmar_ratio=None,
            profit_factor=None,
            signal_precision=0.0,
            trade_count=0,
            blocked_trade_rate=0.0,
            shadow_divergence=None,
        )

    net_pnl = sum(row.net_return for row in rows)
    equity = 0.0
    peak_equity = 0.0
    max_drawdown = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    trade_count = 0
    true_positive_trades = 0

    for row in rows:
        equity += row.net_return
        peak_equity = max(peak_equity, equity)
        max_drawdown = max(max_drawdown, peak_equity - equity)
        if row.net_return > 0.0:
            gross_profit += row.net_return
        elif row.net_return < 0.0:
            gross_loss += abs(row.net_return)
        if row.trade_taken:
            trade_count += 1
            if row.y_true == 1:
                true_positive_trades += 1

    signal_precision = 0.0 if trade_count == 0 else true_positive_trades / trade_count
    calmar_ratio = None if max_drawdown <= 1e-12 else net_pnl / max_drawdown
    profit_factor = None if gross_loss <= 1e-12 else gross_profit / gross_loss
    return EnsembleEvaluationSliceMetrics(
        slice_label=slice_label,
        net_pnl_after_fees_slippage=net_pnl,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        profit_factor=profit_factor,
        signal_precision=signal_precision,
        trade_count=trade_count,
        blocked_trade_rate=0.0,
        shadow_divergence=None,
    )


def _sample_feature_input(
    sample: DatasetSample,
    feature_columns: tuple[str, ...],
) -> dict[str, Any]:
    feature_input: dict[str, Any] = {}
    for column in feature_columns:
        if column == "symbol":
            feature_input[column] = sample.symbol
            continue
        if column not in sample.features:
            raise ValueError(
                f"Training sample {sample.row_id} is missing feature {column}"
            )
        feature_input[column] = sample.features[column]
    return feature_input


def _result_key(result: EnsembleResearchResult, slice_label: str) -> tuple[Any, ...]:
    metrics = result.metrics_by_slice[slice_label]
    calmar_ratio = (
        float("-inf") if metrics.calmar_ratio is None else metrics.calmar_ratio
    )
    profit_factor = (
        float("-inf") if metrics.profit_factor is None else metrics.profit_factor
    )
    return (
        metrics.net_pnl_after_fees_slippage,
        calmar_ratio,
        profit_factor,
        metrics.signal_precision,
        metrics.trade_count,
        result.candidate.model_version,
    )


def _select_best_result(
    results: list[EnsembleResearchResult],
    slice_label: str,
) -> EnsembleResearchResult:
    return max(results, key=lambda result: _result_key(result, slice_label))


def _build_evidence_summary(
    *,
    selected_generalist: EnsembleResearchResult,
    selected_trend: EnsembleResearchResult,
    selected_range: EnsembleResearchResult,
    candidate_results: list[EnsembleResearchResult],
) -> dict[str, Any]:
    return make_json_safe(
        {
            "packet": "M20_PACKET_2",
            "top_level_model_identity": {
                "model_name": "dynamic_ensemble",
                "model_version_pattern": "ensemble_profile:<profile_id>",
            },
            "required_slices": [
                SLICE_ALL,
                SLICE_TREND_COMBINED,
                SLICE_RANGE,
                SLICE_HIGH_VOL,
            ],
            "selected_roster": {
                GENERALIST: selected_generalist.model_dump(mode="json"),
                TREND_SPECIALIST: selected_trend.model_dump(mode="json"),
                RANGE_SPECIALIST: selected_range.model_dump(mode="json"),
            },
            "candidate_count": len(candidate_results),
            "deferred_items": [
                (
                    "Weighted aggregate ensemble explainability remains deferred "
                    "beyond Packet 2."
                ),
            ],
        }
    )


def _primary_slice_for_role(candidate_role: str) -> str:
    if candidate_role == GENERALIST:
        return SLICE_ALL
    if candidate_role == TREND_SPECIALIST:
        return SLICE_TREND_COMBINED
    if candidate_role == RANGE_SPECIALIST:
        return SLICE_RANGE
    raise ValueError(
        f"Unsupported candidate_role for Packet 2 selection: {candidate_role}"
    )



def _default_scope_regimes(candidate_role: str) -> tuple[str, ...]:
    if candidate_role == GENERALIST:
        return DEFAULT_SCOPE_REGIMES
    if candidate_role == TREND_SPECIALIST:
        return TREND_SCOPE_REGIMES
    if candidate_role == RANGE_SPECIALIST:
        return RANGE_SCOPE_REGIMES
    return DEFAULT_SCOPE_REGIMES


def _sample_to_regime_row(sample: DatasetSample) -> RegimeSourceRow:
    return RegimeSourceRow(
        symbol=sample.symbol,
        interval_begin=sample.interval_begin,
        as_of_time=sample.as_of_time,
        realized_vol_12=float(sample.features["realized_vol_12"]),
        momentum_3=float(sample.features["momentum_3"]),
        macd_line_12_26=float(sample.features["macd_line_12_26"]),
    )


def _research_artifact_root() -> Path:
    config = load_ensemble_config(default_ensemble_config_path())
    artifact_root = Path(config.artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = repo_root() / artifact_root
    return artifact_root / "research"
