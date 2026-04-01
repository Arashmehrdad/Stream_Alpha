"""Read and apply bounded M19 adaptation without changing core authorities."""

# pylint: disable=too-many-arguments,too-many-locals,too-many-lines

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from app.adaptation.artifacts import (
    append_jsonl_artifact,
    write_json_artifact,
    write_markdown_artifact,
)
from app.adaptation.calibration import apply_calibration
from app.adaptation.config import (
    AdaptationConfig,
    default_adaptation_config_path,
    load_adaptation_config,
)
from app.adaptation.drift import classify_drift, population_stability_index
from app.adaptation.performance import build_rolling_performance_windows
from app.adaptation.schemas import (
    AdaptationContextPayload,
    AdaptationDriftResponse,
    AdaptationPerformanceResponse,
    AdaptationProfilesResponse,
    AdaptationPromotionsResponse,
    AdaptationSummaryResponse,
    AdaptiveDriftRecord,
    AdaptivePerformanceWindow,
    AdaptiveProfileRecord,
    AdaptivePromotionDecisionRecord,
    AdaptiveRecentPerformanceSummary,
    AppliedAdaptation,
    EffectiveThresholds,
)
from app.adaptation.sizing import bounded_size_multiplier
from app.adaptation.thresholds import bounded_effective_thresholds
from app.common.time import to_rfc3339, utc_now
from app.trading.repository import TradingRepository
from app.trading.schemas import DecisionTraceRecord, PaperPosition, TradeLedgerEntry


class AdaptationService:
    """Bounded M19 adaptation service with read-only APIs and additive runtime hooks."""

    def __init__(
        self,
        *,
        repository: TradingRepository | None = None,
        config: AdaptationConfig | None = None,
    ) -> None:
        self.config = config or load_adaptation_config(default_adaptation_config_path())
        self.repository = repository
        self._connected = False

    async def startup(self) -> None:
        """Open the additive repository connection when configured."""
        await self._ensure_repository_ready()

    async def shutdown(self) -> None:
        """Close the additive repository connection when configured."""
        if self.repository is None or not self._connected:
            return
        await self.repository.close()
        self._connected = False

    async def _ensure_repository_ready(self) -> bool:
        """Best-effort repository initialization for additive adaptation reads."""
        if self.repository is None:
            return False
        if self._connected:
            return True
        try:
            await self.repository.connect()
        except Exception:  # pylint: disable=broad-exception-caught
            return False
        self._connected = True
        return True

    async def resolve_applied_adaptation(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        base_buy_prob_up: float,
        base_sell_prob_up: float,
        confidence: float,
        health_overall_status: str | None,
        freshness_status: str | None,
    ) -> AppliedAdaptation:
        """Resolve the active bounded adaptation to apply to one M4/M10 path."""
        if not self.config.enabled:
            return AppliedAdaptation(adaptation_reason_codes=("ADAPTATION_DISABLED",))
        if not await self._ensure_repository_ready():
            return AppliedAdaptation(
                adaptation_reason_codes=("ADAPTATION_REPOSITORY_UNAVAILABLE",)
            )
        profile = await self.repository.load_active_adaptive_profile(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )
        if profile is None:
            return AppliedAdaptation(
                adaptation_reason_codes=("NO_ACTIVE_ADAPTATION_PROFILE",)
            )
        drift_state = await self.repository.load_latest_adaptive_drift_state(
            symbol=symbol,
            regime_label=regime_label,
        )
        performance = await self.repository.load_latest_adaptive_performance_window(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )
        self._write_drift_summary_artifact(
            symbol=symbol,
            regime_label=regime_label,
            items=[] if drift_state is None else [drift_state],
        )
        self._write_performance_summary_artifact(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
            items=[] if performance is None else [performance],
        )
        frozen_by_health_gate = self._is_frozen_by_health_gate(
            health_overall_status=health_overall_status,
            freshness_status=freshness_status,
            drift_state=drift_state,
        )
        calibrated_confidence = apply_calibration(
            profile.calibration_profile_json,
            confidence,
        )
        effective_thresholds = bounded_effective_thresholds(
            base_buy_prob_up=base_buy_prob_up,
            base_sell_prob_up=base_sell_prob_up,
            calibrated_confidence=calibrated_confidence,
            performance=performance,
            configured_delta=profile.threshold_policy_json.buy_threshold_delta,
            bounds=self.config.threshold_bounds,
        )
        size_multiplier = bounded_size_multiplier(
            configured_multiplier=profile.sizing_policy_json.size_multiplier,
            calibrated_confidence=calibrated_confidence,
            performance=performance,
            bounds=self.config.sizing_bounds,
        )
        reason_codes = ["ADAPTATION_PROFILE_ACTIVE"]
        if frozen_by_health_gate:
            effective_thresholds = EffectiveThresholds(
                buy_prob_up=base_buy_prob_up,
                sell_prob_up=base_sell_prob_up,
            )
            size_multiplier = 1.0
            reason_codes.append("ADAPTATION_FROZEN_BY_HEALTH_GATE")
        return AppliedAdaptation(
            profile_id=profile.profile_id,
            calibrated_confidence=calibrated_confidence,
            effective_thresholds=effective_thresholds,
            adaptive_size_multiplier=size_multiplier,
            drift_status=None if drift_state is None else drift_state.status,
            recent_performance_summary=(
                None
                if performance is None
                else performance_to_summary(performance)
            ),
            adaptation_reason_codes=tuple(reason_codes),
            frozen_by_health_gate=frozen_by_health_gate,
        )

    async def summary(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
    ) -> AdaptationSummaryResponse:
        """Return the read-only M19 adaptation summary payload."""
        if not await self._ensure_repository_ready():
            return AdaptationSummaryResponse(
                enabled=self.config.enabled,
                active_profile_count=0,
                adaptation_status="UNAVAILABLE",
                reason_codes=["ADAPTATION_REPOSITORY_UNAVAILABLE"],
            )
        profiles = await self.repository.load_adaptive_profiles(limit=50)
        promotions = await self.repository.load_adaptive_promotion_decisions(limit=1)
        drift = await self.repository.load_adaptive_drift_states(
            symbol=symbol,
            regime_label=regime_label,
            limit=1,
        )
        performance = await self.repository.load_adaptive_performance_windows(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
            limit=1,
        )
        active_profile = await self.repository.load_active_adaptive_profile(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )
        evidence_backed = bool(drift or performance)
        reason_codes = (
            ["ACTIVE_PROFILE_PRESENT"]
            if active_profile is not None
            else ["NO_ACTIVE_PROFILE"]
        )
        if evidence_backed:
            reason_codes.append("RUNTIME_EVIDENCE_PRESENT")
        else:
            reason_codes.append("NO_RUNTIME_EVIDENCE")
        return AdaptationSummaryResponse(
            enabled=self.config.enabled,
            active_profile_count=sum(1 for item in profiles if item.status == "ACTIVE"),
            active_profile_id=(
                None if active_profile is None else active_profile.profile_id
            ),
            adaptation_status=("ACTIVE" if active_profile is not None else "IDLE"),
            evidence_backed=evidence_backed,
            latest_drift_status=None if not drift else drift[0].status,
            latest_drift_updated_at=None if not drift else drift[0].updated_at,
            latest_promotion_decision=(
                None if not promotions else promotions[0].decision
            ),
            latest_performance_window_id=(
                None if not performance else performance[0].window_id
            ),
            latest_performance_trade_count=(
                None if not performance else performance[0].trade_count
            ),
            latest_performance_created_at=(
                None if not performance else performance[0].created_at
            ),
            reason_codes=reason_codes,
        )

    async def drift(
        self,
        *,
        symbol: str,
        regime_label: str,
        limit: int = 50,
    ) -> AdaptationDriftResponse:
        """Return the read-only drift collection."""
        if not await self._ensure_repository_ready():
            return AdaptationDriftResponse()
        items = await self.repository.load_adaptive_drift_states(
            symbol=symbol,
            regime_label=regime_label,
            limit=limit,
        )
        self._write_drift_summary_artifact(
            symbol=symbol,
            regime_label=regime_label,
            items=items,
        )
        return AdaptationDriftResponse(items=items)

    async def performance(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        limit: int = 50,
    ) -> AdaptationPerformanceResponse:
        """Return the read-only rolling performance collection."""
        if not await self._ensure_repository_ready():
            return AdaptationPerformanceResponse()
        items = await self.repository.load_adaptive_performance_windows(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
            limit=limit,
        )
        self._write_performance_summary_artifact(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
            items=items,
        )
        return AdaptationPerformanceResponse(items=items)

    async def profiles(self, *, limit: int = 50) -> AdaptationProfilesResponse:
        """Return the read-only adaptive profile collection."""
        if not await self._ensure_repository_ready():
            return AdaptationProfilesResponse()
        return AdaptationProfilesResponse(
            items=await self.repository.load_adaptive_profiles(limit=limit)
        )

    async def promotions(self, *, limit: int = 50) -> AdaptationPromotionsResponse:
        """Return the read-only adaptive promotion collection."""
        if not await self._ensure_repository_ready():
            return AdaptationPromotionsResponse()
        return AdaptationPromotionsResponse(
            items=await self.repository.load_adaptive_promotion_decisions(limit=limit)
        )

    async def rollback_active_profile(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        decision_id: str,
        summary_text: str,
    ) -> AdaptivePromotionDecisionRecord:
        """Persist and apply an explicit runtime rollback to the configured target profile."""
        if not await self._ensure_repository_ready():
            raise RuntimeError("Adaptation repository unavailable for rollback")
        active_profile = await self.repository.load_active_adaptive_profile(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )
        if active_profile is None:
            raise RuntimeError("No active adaptive profile is available for rollback")
        if active_profile.rollback_target_profile_id is None:
            raise RuntimeError("Active adaptive profile has no rollback target")
        rollback_target = await self.repository.load_adaptive_profile(
            profile_id=active_profile.rollback_target_profile_id,
        )
        if rollback_target is None:
            raise RuntimeError("Rollback target adaptive profile was not found")
        decided_at = utc_now()
        decision = AdaptivePromotionDecisionRecord(
            decision_id=decision_id,
            target_type="PROFILE",
            target_id=rollback_target.profile_id,
            incumbent_id=active_profile.profile_id,
            decision="ROLLBACK",
            metrics_delta_json={
                "rolled_back_profile_id": active_profile.profile_id,
                "restored_profile_id": rollback_target.profile_id,
            },
            safety_checks_json={"runtime_rollback": True},
            research_integrity_json={
                "execution_mode": execution_mode,
                "symbol": symbol,
                "regime_label": regime_label,
            },
            reason_codes=["ROLLBACK_TARGET_ACTIVATED"],
            summary_text=summary_text,
            decided_at=decided_at,
        )
        await self.repository.rollback_adaptive_profile(
            active_profile_id=active_profile.profile_id,
            rollback_target_profile_id=rollback_target.profile_id,
            changed_at=decided_at,
        )
        await self.repository.save_adaptive_promotion_decision(decision)
        restored_profile = await self.repository.load_adaptive_profile(
            profile_id=rollback_target.profile_id,
        )
        if restored_profile is None:
            raise RuntimeError("Rollback target could not be reloaded after rollback")
        self.write_profile_artifacts(
            profile=restored_profile,
            latest_promotion=decision,
        )
        drift_state = await self.repository.load_latest_adaptive_drift_state(
            symbol=symbol,
            regime_label=regime_label,
        )
        performance = await self.repository.load_latest_adaptive_performance_window(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )
        self._write_drift_summary_artifact(
            symbol=symbol,
            regime_label=regime_label,
            items=[] if drift_state is None else [drift_state],
        )
        self._write_performance_summary_artifact(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
            items=[] if performance is None else [performance],
        )
        return decision

    async def write_runtime_persisted_truth(
        self,
        *,
        service_name: str,
        execution_mode: str,
        source_exchange: str,
        interval_minutes: int,
        symbols: tuple[str, ...],
        system_reliability=None,
        ) -> None:
        """Persist additive M19 runtime truth from existing feature, trade, and trace evidence."""
        if not self.config.enabled or not await self._ensure_repository_ready():
            return
        if self.repository is None:
            return
        if not all(
            hasattr(self.repository, attribute)
            for attribute in (
                "load_feature_rows_for_adaptation",
                "save_adaptive_drift_state",
                "load_positions",
                "load_trade_ledger_entries",
                "load_decision_traces_since",
                "save_adaptive_performance_window",
            )
        ):
            return
        evaluated_at = utc_now()
        drift_items = await self._persist_runtime_drift_states(
            source_exchange=source_exchange,
            interval_minutes=interval_minutes,
            symbols=symbols,
        )
        performance_items = await self._persist_runtime_performance_windows(
            service_name=service_name,
            execution_mode=execution_mode,
            system_reliability=system_reliability,
            evaluated_at=evaluated_at,
        )
        self._write_drift_summary_artifact(
            symbol="ALL",
            regime_label="ALL",
            items=drift_items,
        )
        self._write_performance_summary_artifact(
            execution_mode=execution_mode,
            symbol="ALL",
            regime_label="ALL",
            items=performance_items,
        )

    def write_profile_artifacts(
        self,
        *,
        profile: AdaptiveProfileRecord,
        latest_promotion: AdaptivePromotionDecisionRecord | None = None,
    ) -> None:
        """Write deterministic profile and promotion artifacts."""
        write_json_artifact(
            self.config.artifacts.current_profile_path,
            profile.model_dump(mode="json"),
        )
        report_payload = {
            "generated_at": to_rfc3339(utc_now()),
            "profile": profile.model_dump(mode="json"),
            "latest_promotion": (
                None if latest_promotion is None else latest_promotion.model_dump(mode="json")
            ),
        }
        report_json_path = Path(self.config.artifacts.reports_dir) / f"{profile.profile_id}.json"
        report_md_path = Path(self.config.artifacts.reports_dir) / f"{profile.profile_id}.md"
        write_json_artifact(report_json_path, report_payload)
        write_markdown_artifact(
            report_md_path,
            [
                f"# Adaptive Profile {profile.profile_id}",
                "",
                f"- status: {profile.status}",
                f"- execution_mode_scope: {profile.execution_mode_scope}",
                f"- symbol_scope: {profile.symbol_scope}",
                f"- regime_scope: {profile.regime_scope}",
                f"- rollback_target_profile_id: {profile.rollback_target_profile_id}",
            ],
        )
        if latest_promotion is not None:
            append_jsonl_artifact(
                self.config.artifacts.promotions_history_path,
                latest_promotion.model_dump(mode="json"),
            )

    def _write_drift_summary_artifact(
        self,
        *,
        symbol: str,
        regime_label: str,
        items: list[AdaptiveDriftRecord],
    ) -> None:
        """Write the configured latest drift summary artifact."""
        write_json_artifact(
            self.config.artifacts.drift_summary_path,
            {
                "generated_at": to_rfc3339(utc_now()),
                "symbol": symbol,
                "regime_label": regime_label,
                "item_count": len(items),
                "items": [item.model_dump(mode="json") for item in items],
            },
        )

    def _write_performance_summary_artifact(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        items: list[AdaptivePerformanceWindow],
    ) -> None:
        """Write the configured latest performance summary artifact."""
        write_json_artifact(
            self.config.artifacts.performance_summary_path,
            {
                "generated_at": to_rfc3339(utc_now()),
                "execution_mode": execution_mode,
                "symbol": symbol,
                "regime_label": regime_label,
                "item_count": len(items),
                "items": [item.model_dump(mode="json") for item in items],
            },
        )

    def to_trace_payload(
        self,
        applied: AppliedAdaptation,
    ) -> AdaptationContextPayload | None:
        """Convert one resolved adaptation into canonical trace payload form."""
        if applied.profile_id is None and not applied.adaptation_reason_codes:
            return None
        return AdaptationContextPayload(
            adaptation_profile_id=applied.profile_id,
            threshold_policy_id=applied.profile_id,
            sizing_policy_id=applied.profile_id,
            calibration_profile_id=applied.profile_id,
            drift_status=applied.drift_status,
            recent_performance_summary=applied.recent_performance_summary,
            adaptation_reason_codes=list(applied.adaptation_reason_codes),
            frozen_by_health_gate=applied.frozen_by_health_gate,
            calibrated_confidence=applied.calibrated_confidence,
            adaptive_size_multiplier=applied.adaptive_size_multiplier,
            effective_thresholds=applied.effective_thresholds,
        )

    async def _persist_runtime_drift_states(
        self,
        *,
        source_exchange: str,
        interval_minutes: int,
        symbols: tuple[str, ...],
    ) -> list[AdaptiveDriftRecord]:
        reference_count = self.config.drift.minimum_reference_samples
        live_count = self.config.drift.minimum_live_samples
        required_rows = reference_count + live_count
        aggregate_reference: dict[str, list[float]] = defaultdict(list)
        aggregate_live: dict[str, list[float]] = defaultdict(list)
        reference_starts = []
        reference_ends = []
        live_starts = []
        live_ends = []
        included_symbols: list[str] = []
        saved_records: list[AdaptiveDriftRecord] = []

        for symbol in symbols:
            rows = await self.repository.load_feature_rows_for_adaptation(
                symbol=symbol,
                source_exchange=source_exchange,
                interval_minutes=interval_minutes,
                feature_columns=self.config.drift.features,
                limit=required_rows,
            )
            split_windows = self._split_drift_windows(
                rows=rows,
                reference_count=reference_count,
                live_count=live_count,
            )
            if split_windows is None:
                continue
            reference_rows, live_rows = split_windows
            reference_series = self._feature_series(
                rows=reference_rows,
                feature_names=self.config.drift.features,
            )
            live_series = self._feature_series(
                rows=live_rows,
                feature_names=self.config.drift.features,
            )
            drift_record = self._build_drift_record_from_series(
                symbol=symbol,
                regime_label="ALL",
                reference_window_start=reference_rows[0]["interval_begin"],
                reference_window_end=reference_rows[-1]["interval_begin"],
                live_window_start=live_rows[0]["interval_begin"],
                live_window_end=live_rows[-1]["interval_begin"],
                reference_series=reference_series,
                live_series=live_series,
                included_symbols=(symbol,),
            )
            if drift_record is None:
                continue
            await self.repository.save_adaptive_drift_state(drift_record)
            saved_records.append(drift_record)
            for feature_name, values in reference_series.items():
                aggregate_reference[feature_name].extend(values)
            for feature_name, values in live_series.items():
                aggregate_live[feature_name].extend(values)
            reference_starts.append(reference_rows[0]["interval_begin"])
            reference_ends.append(reference_rows[-1]["interval_begin"])
            live_starts.append(live_rows[0]["interval_begin"])
            live_ends.append(live_rows[-1]["interval_begin"])
            included_symbols.append(symbol)

        if not included_symbols:
            return saved_records
        aggregate_record = self._build_drift_record_from_series(
            symbol="ALL",
            regime_label="ALL",
            reference_window_start=min(reference_starts),
            reference_window_end=max(reference_ends),
            live_window_start=min(live_starts),
            live_window_end=max(live_ends),
            reference_series=dict(aggregate_reference),
            live_series=dict(aggregate_live),
            included_symbols=tuple(sorted(included_symbols)),
        )
        if aggregate_record is not None:
            await self.repository.save_adaptive_drift_state(aggregate_record)
            saved_records.append(aggregate_record)
        return saved_records

    async def _persist_runtime_performance_windows(
        self,
        *,
        service_name: str,
        execution_mode: str,
        system_reliability,
        evaluated_at,
    ) -> list[AdaptivePerformanceWindow]:
        positions = await self.repository.load_positions(
            service_name=service_name,
            execution_mode=execution_mode,
        )
        closed_positions = [
            position
            for position in positions
            if position.status == "CLOSED"
            and position.realized_pnl is not None
            and self._position_event_time(position) is not None
        ]
        if not closed_positions:
            return []
        earliest_event_time = min(
            self._position_event_time(position) for position in closed_positions
        )
        decision_traces = await self.repository.load_decision_traces_since(
            service_name=service_name,
            execution_mode=execution_mode,
            since=earliest_event_time,
        )
        ledger_entries = await self.repository.load_trade_ledger_entries(
            service_name=service_name,
            execution_mode=execution_mode,
            since=earliest_event_time,
        )
        saved_windows: list[AdaptivePerformanceWindow] = []
        for symbol_scope, regime_scope, scoped_positions in self._performance_scopes(
            closed_positions
        ):
            position_rows = self._performance_rows_from_positions(
                positions=scoped_positions,
                ledger_entries=ledger_entries,
            )
            for window in build_rolling_performance_windows(
                execution_mode=execution_mode,
                symbol=symbol_scope,
                regime_label=regime_scope,
                rows=position_rows,
                trade_counts=self.config.rolling_windows.trade_counts,
                day_windows=self.config.rolling_windows.day_windows,
                now=evaluated_at,
            ):
                scoped_traces = self._decision_traces_in_window(
                    traces=decision_traces,
                    symbol_scope=symbol_scope,
                    regime_scope=regime_scope,
                    window_start=window.window_start,
                    window_end=window.window_end,
                )
                comparable_count, positive_count = self._precision_counts(
                    traces=scoped_traces,
                    positions=scoped_positions,
                )
                blocked_count = sum(1 for trace in scoped_traces if self._trace_blocked(trace))
                decision_count = len(scoped_traces)
                slippage_sample_count = sum(
                    1
                    for row in position_rows
                    if window.window_start <= row["event_time"] <= window.window_end
                    and row["slippage_sample_count"] > 0
                )
                persisted_window = window.model_copy(
                    update={
                        "precision": (
                            0.0
                            if comparable_count == 0
                            else positive_count / comparable_count
                        ),
                        "blocked_trade_rate": (
                            0.0 if decision_count == 0 else blocked_count / decision_count
                        ),
                        "shadow_divergence_rate": 0.0,
                        "health_context": self._performance_health_context(
                            system_reliability=system_reliability,
                            decision_count=decision_count,
                            blocked_count=blocked_count,
                            comparable_count=comparable_count,
                            positive_count=positive_count,
                            slippage_sample_count=slippage_sample_count,
                        ),
                    }
                )
                await self.repository.save_adaptive_performance_window(persisted_window)
                saved_windows.append(persisted_window)
        return saved_windows

    def _split_drift_windows(
        self,
        *,
        rows: list[dict[str, object]],
        reference_count: int,
        live_count: int,
    ) -> tuple[list[dict[str, object]], list[dict[str, object]]] | None:
        if len(rows) < reference_count + live_count:
            return None
        reference_rows = rows[-(reference_count + live_count) : -live_count]
        live_rows = rows[-live_count:]
        if len(reference_rows) < reference_count or len(live_rows) < live_count:
            return None
        return reference_rows, live_rows

    def _feature_series(
        self,
        *,
        rows: list[dict[str, object]],
        feature_names: tuple[str, ...],
    ) -> dict[str, list[float]]:
        series: dict[str, list[float]] = {}
        for feature_name in feature_names:
            values = [
                float(value)
                for row in rows
                for value in (row.get(feature_name),)
                if isinstance(value, (int, float))
            ]
            series[feature_name] = values
        return series

    def _build_drift_record_from_series(
        self,
        *,
        symbol: str,
        regime_label: str,
        reference_window_start,
        reference_window_end,
        live_window_start,
        live_window_end,
        reference_series: dict[str, list[float]],
        live_series: dict[str, list[float]],
        included_symbols: tuple[str, ...],
    ) -> AdaptiveDriftRecord | None:
        scored_features: list[tuple[str, float, float, float]] = []
        unavailable_features: list[str] = []
        for feature_name in self.config.drift.features:
            reference_values = reference_series.get(feature_name, [])
            live_values = live_series.get(feature_name, [])
            if (
                len(reference_values) < self.config.drift.minimum_reference_samples
                or len(live_values) < self.config.drift.minimum_live_samples
            ):
                unavailable_features.append(feature_name)
                continue
            thresholds = self.config.drift.feature_thresholds.get(
                feature_name,
                self.config.drift.feature_default,
            )
            scored_features.append(
                (
                    feature_name,
                    population_stability_index(reference_values, live_values),
                    thresholds.warning,
                    thresholds.breach,
                )
            )
        if not scored_features:
            return None
        dominant_feature, drift_score, warning_threshold, breach_threshold = max(
            scored_features,
            key=lambda item: item[1],
        )
        status, reason_code = classify_drift(
            drift_score,
            warning_threshold=warning_threshold,
            breach_threshold=breach_threshold,
        )
        return AdaptiveDriftRecord(
            symbol=symbol,
            regime_label=regime_label,
            detector_name=self.config.drift.detector_name,
            window_id=(
                f"rolling_{self.config.drift.minimum_reference_samples}"
                f"_vs_{self.config.drift.minimum_live_samples}"
            ),
            reference_window_start=reference_window_start,
            reference_window_end=reference_window_end,
            live_window_start=live_window_start,
            live_window_end=live_window_end,
            drift_score=drift_score,
            warning_threshold=warning_threshold,
            breach_threshold=breach_threshold,
            status=status,
            reason_code=reason_code,
            detail=(
                "Feature PSI drift from persisted runtime feature rows. "
                f"dominant_feature={dominant_feature}; "
                f"scored_features={len(scored_features)}; "
                f"included_symbols={list(included_symbols)}; "
                f"unavailable_features={unavailable_features}"
            ),
        )

    def _performance_scopes(
        self,
        positions: list[PaperPosition],
    ) -> list[tuple[str, str, list[PaperPosition]]]:
        scopes: list[tuple[str, str, list[PaperPosition]]] = [("ALL", "ALL", positions)]
        symbols = sorted({position.symbol for position in positions})
        regimes = sorted({self._regime_key(position.entry_regime_label) for position in positions})
        for symbol in symbols:
            scopes.append(
                (
                    symbol,
                    "ALL",
                    [position for position in positions if position.symbol == symbol],
                )
            )
        for regime_label in regimes:
            scopes.append(
                (
                    "ALL",
                    regime_label,
                    [
                        position
                        for position in positions
                        if self._regime_key(position.entry_regime_label) == regime_label
                    ],
                )
            )
        for symbol in symbols:
            for regime_label in regimes:
                scoped_positions = [
                    position
                    for position in positions
                    if position.symbol == symbol
                    and self._regime_key(position.entry_regime_label) == regime_label
                ]
                if scoped_positions:
                    scopes.append((symbol, regime_label, scoped_positions))
        return [scope for scope in scopes if scope[2]]

    def _performance_rows_from_positions(
        self,
        *,
        positions: list[PaperPosition],
        ledger_entries: list[TradeLedgerEntry],
    ) -> list[dict[str, object]]:
        ledger_by_position_id: dict[int, list[TradeLedgerEntry]] = defaultdict(list)
        for entry in ledger_entries:
            if entry.position_id is not None:
                ledger_by_position_id[entry.position_id].append(entry)
        rows: list[dict[str, object]] = []
        for position in positions:
            event_time = self._position_event_time(position)
            if event_time is None:
                continue
            position_ledger = (
                []
                if position.position_id is None
                else ledger_by_position_id.get(position.position_id, [])
            )
            slippage_values = [entry.slippage_bps for entry in position_ledger]
            rows.append(
                {
                    "event_time": event_time,
                    "realized_pnl": position.realized_pnl or 0.0,
                    "slippage_bps": (
                        0.0
                        if not slippage_values
                        else sum(slippage_values) / len(slippage_values)
                    ),
                    "predicted_positive": True,
                    "true_positive": (position.realized_pnl or 0.0) > 0.0,
                    "blocked": False,
                    "shadow_diverged": False,
                    "health_context": {},
                    "slippage_sample_count": len(slippage_values),
                }
            )
        return rows

    def _decision_traces_in_window(
        self,
        *,
        traces: list[DecisionTraceRecord],
        symbol_scope: str,
        regime_scope: str,
        window_start,
        window_end,
    ) -> list[DecisionTraceRecord]:
        return [
            trace
            for trace in traces
            if window_start <= trace.signal_as_of_time <= window_end
            and symbol_scope in ("ALL", trace.symbol)
            and regime_scope in ("ALL", self._trace_regime_label(trace))
        ]

    def _precision_counts(
        self,
        *,
        traces: list[DecisionTraceRecord],
        positions: list[PaperPosition],
    ) -> tuple[int, int]:
        positive_by_trace_id = {
            position.entry_decision_trace_id: (position.realized_pnl or 0.0) > 0.0
            for position in positions
            if position.entry_decision_trace_id is not None
        }
        comparable = 0
        positive = 0
        for trace in traces:
            if trace.signal != "BUY":
                continue
            outcome = positive_by_trace_id.get(trace.decision_trace_id)
            if outcome is None:
                continue
            comparable += 1
            if outcome:
                positive += 1
        return comparable, positive

    def _performance_health_context(
        self,
        *,
        system_reliability,
        decision_count: int,
        blocked_count: int,
        comparable_count: int,
        positive_count: int,
        slippage_sample_count: int,
    ) -> dict[str, object]:
        return {
            "system_reliability_available": system_reliability is not None,
            "system_health_overall_status": (
                None
                if system_reliability is None
                else system_reliability.health_overall_status
            ),
            "system_reason_codes": (
                []
                if system_reliability is None
                else list(system_reliability.reason_codes)
            ),
            "decision_trace_count": decision_count,
            "blocked_decision_count": blocked_count,
            "precision_comparable_count": comparable_count,
            "precision_positive_count": positive_count,
            "slippage_sample_count": slippage_sample_count,
        }

    def _position_event_time(self, position: PaperPosition):
        return position.exit_fill_time or position.closed_at or position.entry_fill_time

    def _trace_regime_label(self, trace: DecisionTraceRecord) -> str:
        if trace.payload.regime_reason is None:
            return "UNKNOWN"
        return self._regime_key(trace.payload.regime_reason.regime_label)

    def _trace_blocked(self, trace: DecisionTraceRecord) -> bool:
        return trace.payload.blocked_trade is not None or trace.risk_outcome == "BLOCKED"

    def _regime_key(self, value: str | None) -> str:
        if value is None or not value.strip():
            return "UNKNOWN"
        return value

    def _is_frozen_by_health_gate(
        self,
        *,
        health_overall_status: str | None,
        freshness_status: str | None,
        drift_state: AdaptiveDriftRecord | None,
    ) -> bool:
        if (
            self.config.freeze_rules.freeze_on_degraded_reliability
            and health_overall_status in self.config.freeze_rules.degraded_health_statuses
        ):
            return True
        if (
            self.config.freeze_rules.freeze_on_degraded_reliability
            and freshness_status in self.config.freeze_rules.degraded_freshness_statuses
        ):
            return True
        if (
            self.config.freeze_rules.freeze_on_drift_breach
            and drift_state is not None
            and drift_state.status == "BREACHED"
        ):
            return True
        return False


def performance_to_summary(
    performance: AdaptivePerformanceWindow,
) -> AdaptiveRecentPerformanceSummary:
    """Convert one persisted performance row into the compact summary shape."""
    return AdaptiveRecentPerformanceSummary(
        window_id=performance.window_id,
        window_type=performance.window_type,
        trade_count=performance.trade_count,
        net_pnl_after_costs=performance.net_pnl_after_costs,
        max_drawdown=performance.max_drawdown,
        profit_factor=performance.profit_factor,
        win_rate=performance.win_rate,
        blocked_trade_rate=performance.blocked_trade_rate,
        shadow_divergence_rate=performance.shadow_divergence_rate,
    )
