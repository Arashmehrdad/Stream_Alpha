"""Deterministic M4-side explainability helpers for M14."""

# pylint: disable=too-many-arguments

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from app.common.time import to_rfc3339, utc_now
from app.explainability.config import ExplainabilityConfig
from app.explainability.schemas import (
    PredictionExplanation,
    ReferenceVectorArtifact,
    RegimeObservedMetrics,
    RegimeReason,
    RegimeThresholds,
    ResolvedReferenceVector,
    SignalExplanation,
    ThresholdSnapshot,
    TopFeatureContribution,
)
from app.reliability.artifacts import write_json_artifact


REFERENCE_ABLATION_METHOD = "ONE_AT_A_TIME_REFERENCE_ABLATION"
REFERENCE_VECTOR_SCHEMA_VERSION = "m14_reference_vector_v1"
REFERENCE_VECTOR_SOURCE_FEATURE_MEDIANS = "FEATURE_OHLC_MEDIANS"
REFERENCE_VECTOR_SOURCE_PERSISTED_ARTIFACT = "PERSISTED_ARTIFACT"
EXPLAINABILITY_AVAILABLE = "EXPLAINABILITY_AVAILABLE"
EXPLAINABILITY_NO_NUMERIC_FEATURES = "EXPLAINABILITY_NO_NUMERIC_FEATURES"
EXPLAINABILITY_REFERENCE_UNAVAILABLE = "EXPLAINABILITY_REFERENCE_UNAVAILABLE"
SIGNAL_EXPLANATION_MODEL_DECISION = "SIGNAL_MODEL_DECISION"
SIGNAL_EXPLANATION_RELIABILITY_HOLD = "SIGNAL_RELIABILITY_HOLD"
REGIME_HIGH_VOL = "REGIME_HIGH_VOL"
REGIME_TREND_UP = "REGIME_TREND_UP"
REGIME_TREND_DOWN = "REGIME_TREND_DOWN"
REGIME_RANGE = "REGIME_RANGE"


class ExplainabilityService:
    """Resolve persisted references and build additive M4 explainability payloads."""

    def __init__(self, config: ExplainabilityConfig) -> None:
        self.config = config
        self._reference_cache: dict[str, ResolvedReferenceVector] = {}

    async def build_prediction_details(
        self,
        *,
        model_artifact: Any,
        database: Any,
        feature_input: dict[str, Any],
        interval_minutes: int,
        source_table: str,
        prob_up: float,
    ) -> tuple[list[TopFeatureContribution], PredictionExplanation]:
        """Build additive top-feature contributions for one prediction response."""
        explainable_feature_names = self._resolve_explainable_feature_names(
            model_artifact.feature_columns,
            feature_input,
        )
        if not explainable_feature_names:
            return (
                [],
                PredictionExplanation(
                    method=REFERENCE_ABLATION_METHOD,
                    available=False,
                    reason_code=EXPLAINABILITY_NO_NUMERIC_FEATURES,
                    summary_text=(
                        "No configured numeric market features overlapped the active "
                        "model feature columns."
                    ),
                    explainable_feature_count=0,
                    top_feature_count=0,
                ),
            )

        reference_vector = await self._resolve_reference_vector(
            model_artifact=model_artifact,
            database=database,
            explainable_feature_names=explainable_feature_names,
            interval_minutes=interval_minutes,
            source_table=source_table,
        )
        if reference_vector is None:
            return (
                [],
                PredictionExplanation(
                    method=REFERENCE_ABLATION_METHOD,
                    available=False,
                    reason_code=EXPLAINABILITY_REFERENCE_UNAVAILABLE,
                    summary_text=(
                        "Explainability reference values were unavailable, so no "
                        "top-feature contributions were produced."
                    ),
                    explainable_feature_count=len(explainable_feature_names),
                    top_feature_count=0,
                ),
            )

        top_features = compute_top_feature_contributions(
            model=model_artifact.model,
            feature_input=feature_input,
            base_prob_up=prob_up,
            reference_values=reference_vector.reference_values,
            explainable_feature_names=explainable_feature_names,
            top_feature_count=self.config.reference.top_feature_count,
        )
        return (
            top_features,
            PredictionExplanation(
                method=REFERENCE_ABLATION_METHOD,
                available=True,
                reason_code=EXPLAINABILITY_AVAILABLE,
                summary_text=(
                    "Top features use one-at-a-time reference ablation against the "
                    "persisted reference vector for the active model version."
                ),
                reference_vector_path=reference_vector.reference_vector_path,
                reference_vector_source=reference_vector.reference_vector_source,
                explainable_feature_count=len(explainable_feature_names),
                top_feature_count=len(top_features),
            ),
        )

    def build_prediction_unavailable(
        self,
        *,
        summary_text: str,
    ) -> PredictionExplanation:
        """Return a stable fallback when explainability should not block M4 responses."""
        return PredictionExplanation(
            method=REFERENCE_ABLATION_METHOD,
            available=False,
            reason_code=EXPLAINABILITY_REFERENCE_UNAVAILABLE,
            summary_text=summary_text,
            explainable_feature_count=0,
            top_feature_count=0,
        )

    def build_threshold_snapshot(
        self,
        *,
        buy_prob_up: float,
        sell_prob_up: float,
        allow_new_long_entries: bool,
        resolved_regime: Any | None,
    ) -> ThresholdSnapshot:
        """Build the explicit threshold snapshot attached to `/signal`."""
        if resolved_regime is None:
            return ThresholdSnapshot(
                buy_prob_up=buy_prob_up,
                sell_prob_up=sell_prob_up,
                allow_new_long_entries=allow_new_long_entries,
            )
        return ThresholdSnapshot(
            buy_prob_up=buy_prob_up,
            sell_prob_up=sell_prob_up,
            allow_new_long_entries=allow_new_long_entries,
            regime_label=resolved_regime.regime_label,
            regime_run_id=resolved_regime.regime_run_id,
            regime_artifact_path=resolved_regime.regime_artifact_path,
            high_vol_threshold=resolved_regime.high_vol_threshold,
            trend_abs_threshold=resolved_regime.trend_abs_threshold,
        )

    def build_signal_explanation(
        self,
        *,
        signal: str,
        decision_source: str,
        reason: str,
        trade_allowed: bool,
        regime_reason: RegimeReason | None,
    ) -> SignalExplanation:
        """Build the additive operator-facing signal explanation summary."""
        if decision_source == "reliability":
            reason_code = SIGNAL_EXPLANATION_RELIABILITY_HOLD
            summary_text = f"Reliability forced HOLD: {reason}"
        else:
            reason_code = SIGNAL_EXPLANATION_MODEL_DECISION
            regime_text = (
                ""
                if regime_reason is None
                else (
                    f" Regime {regime_reason.regime_label} is explained by "
                    f"{regime_reason.reason_code}."
                )
            )
            summary_text = f"{signal} came from the M4 model path: {reason}.{regime_text}"
        return SignalExplanation(
            signal=signal,
            available=True,
            reason_code=reason_code,
            summary_text=summary_text,
            trade_allowed=trade_allowed,
            decision_source=decision_source,
        )

    def _resolve_explainable_feature_names(
        self,
        model_feature_columns: Sequence[str],
        feature_input: dict[str, Any],
    ) -> tuple[str, ...]:
        configured_features = self.config.reference.explainable_numeric_features
        return tuple(
            feature_name
            for feature_name in configured_features
            if feature_name in model_feature_columns
            and feature_name in feature_input
            and _is_numeric(feature_input[feature_name])
        )

    async def _resolve_reference_vector(
        self,
        *,
        model_artifact: Any,
        database: Any,
        explainable_feature_names: Sequence[str],
        interval_minutes: int,
        source_table: str,
    ) -> ResolvedReferenceVector | None:
        cached_reference = self._reference_cache.get(model_artifact.model_version)
        if cached_reference is not None:
            return cached_reference

        artifact_path = self._reference_artifact_path(model_artifact.model_version)
        if artifact_path.is_file():
            loaded_reference = self._load_reference_vector_artifact(
                artifact_path,
                explainable_feature_names=explainable_feature_names,
            )
            self._reference_cache[model_artifact.model_version] = loaded_reference
            return loaded_reference

        reference_values = await database.fetch_feature_reference_vector(
            feature_names=tuple(explainable_feature_names),
            interval_minutes=interval_minutes,
        )
        numeric_reference_values = {
            feature_name: float(reference_values[feature_name])
            for feature_name in explainable_feature_names
            if feature_name in reference_values and _is_numeric(reference_values[feature_name])
        }
        if not numeric_reference_values:
            return None

        artifact_payload = ReferenceVectorArtifact(
            schema_version=REFERENCE_VECTOR_SCHEMA_VERSION,
            model_version=model_artifact.model_version,
            generated_at=to_rfc3339(utc_now()),
            source_table=source_table,
            interval_minutes=interval_minutes,
            reference_vector_source=REFERENCE_VECTOR_SOURCE_FEATURE_MEDIANS,
            explainable_numeric_features=list(explainable_feature_names),
            reference_values=numeric_reference_values,
        )
        write_json_artifact(
            artifact_path,
            artifact_payload.model_dump(mode="json"),
        )
        resolved_reference = ResolvedReferenceVector(
            reference_vector_path=str(artifact_path.resolve()),
            reference_vector_source=REFERENCE_VECTOR_SOURCE_FEATURE_MEDIANS,
            reference_values=numeric_reference_values,
        )
        self._reference_cache[model_artifact.model_version] = resolved_reference
        return resolved_reference

    def _load_reference_vector_artifact(
        self,
        artifact_path: Path,
        *,
        explainable_feature_names: Sequence[str],
    ) -> ResolvedReferenceVector:
        payload = ReferenceVectorArtifact.model_validate(
            json.loads(artifact_path.read_text(encoding="utf-8"))
        )
        reference_values = {
            feature_name: float(payload.reference_values[feature_name])
            for feature_name in explainable_feature_names
            if feature_name in payload.reference_values
        }
        return ResolvedReferenceVector(
            reference_vector_path=str(artifact_path.resolve()),
            reference_vector_source=REFERENCE_VECTOR_SOURCE_PERSISTED_ARTIFACT,
            reference_values=reference_values,
        )

    def _reference_artifact_path(self, model_version: str) -> Path:
        artifact_root = Path(self.config.reference.artifact_root)
        if not artifact_root.is_absolute():
            artifact_root = Path(__file__).resolve().parents[2] / artifact_root
        return artifact_root / model_version / self.config.reference.reference_filename


def compute_top_feature_contributions(
    *,
    model: Any,
    feature_input: dict[str, Any],
    base_prob_up: float,
    reference_values: dict[str, float],
    explainable_feature_names: Sequence[str],
    top_feature_count: int,
) -> list[TopFeatureContribution]:
    """Compute deterministic signed contributions via one-at-a-time ablation."""
    contributions: list[TopFeatureContribution] = []
    for feature_name in explainable_feature_names:
        if feature_name not in reference_values:
            continue
        ablated_input = dict(feature_input)
        reference_value = float(reference_values[feature_name])
        ablated_input[feature_name] = reference_value
        ablated_prob_up = _score_prob_up(model, ablated_input)
        signed_contribution = float(base_prob_up - ablated_prob_up)
        contributions.append(
            TopFeatureContribution(
                feature_name=feature_name,
                feature_value=float(feature_input[feature_name]),
                reference_value=reference_value,
                signed_contribution=signed_contribution,
                direction=_contribution_direction(signed_contribution),
            )
        )

    ordered_contributions = sorted(
        contributions,
        key=lambda contribution: (
            -abs(contribution.signed_contribution),
            contribution.feature_name,
        ),
    )
    return ordered_contributions[:top_feature_count]


def build_regime_reason(
    *,
    resolved_regime: Any,
    trade_allowed: bool,
) -> RegimeReason:
    """Build one explicit regime reason from the exact-row regime metrics."""
    realized_vol_12 = float(resolved_regime.realized_vol_12)
    momentum_3 = float(resolved_regime.momentum_3)
    macd_line_12_26 = float(resolved_regime.macd_line_12_26)
    high_vol_threshold = float(resolved_regime.high_vol_threshold)
    trend_abs_threshold = float(resolved_regime.trend_abs_threshold)

    if realized_vol_12 >= high_vol_threshold:
        reason_code = REGIME_HIGH_VOL
        reason_text = (
            f"realized_vol_12 {realized_vol_12:.6f} >= "
            f"high_vol_threshold {high_vol_threshold:.6f}"
        )
    elif momentum_3 >= trend_abs_threshold and macd_line_12_26 > 0.0:
        reason_code = REGIME_TREND_UP
        reason_text = (
            f"momentum_3 {momentum_3:.6f} >= trend_abs_threshold "
            f"{trend_abs_threshold:.6f} and macd_line_12_26 {macd_line_12_26:.6f} > 0"
        )
    elif momentum_3 <= -trend_abs_threshold and macd_line_12_26 < 0.0:
        reason_code = REGIME_TREND_DOWN
        reason_text = (
            f"momentum_3 {momentum_3:.6f} <= -trend_abs_threshold "
            f"{trend_abs_threshold:.6f} and macd_line_12_26 {macd_line_12_26:.6f} < 0"
        )
    else:
        reason_code = REGIME_RANGE
        reason_text = (
            f"realized_vol_12 {realized_vol_12:.6f} < high_vol_threshold "
            f"{high_vol_threshold:.6f} and trend conditions were not met"
        )

    return RegimeReason(
        regime_label=resolved_regime.regime_label,
        regime_run_id=resolved_regime.regime_run_id,
        reason_code=reason_code,
        reason_text=reason_text,
        observed_metrics=RegimeObservedMetrics(
            realized_vol_12=realized_vol_12,
            momentum_3=momentum_3,
            macd_line_12_26=macd_line_12_26,
        ),
        thresholds=RegimeThresholds(
            high_vol_threshold=high_vol_threshold,
            trend_abs_threshold=trend_abs_threshold,
        ),
        trade_allowed=trade_allowed,
    )


def _score_prob_up(model: Any, feature_input: dict[str, Any]) -> float:
    probabilities = model.predict_proba([feature_input])
    if len(probabilities) != 1 or len(probabilities[0]) != 2:
        raise ValueError("Explainability scoring expects binary predict_proba output")
    return float(probabilities[0][1])


def _contribution_direction(signed_contribution: float) -> str:
    if signed_contribution > 0.0:
        return "UP"
    if signed_contribution < 0.0:
        return "DOWN"
    return "NEUTRAL"


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
