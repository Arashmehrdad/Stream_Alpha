"""Read and apply bounded M19 adaptation without changing core authorities."""

# pylint: disable=too-many-arguments,too-many-locals

from __future__ import annotations

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
        active_profile = await self.repository.load_active_adaptive_profile(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )
        return AdaptationSummaryResponse(
            enabled=self.config.enabled,
            active_profile_count=sum(1 for item in profiles if item.status == "ACTIVE"),
            active_profile_id=(
                None if active_profile is None else active_profile.profile_id
            ),
            adaptation_status=("ACTIVE" if active_profile is not None else "IDLE"),
            latest_drift_status=None if not drift else drift[0].status,
            latest_promotion_decision=(
                None if not promotions else promotions[0].decision
            ),
            reason_codes=(
                ["ACTIVE_PROFILE_PRESENT"]
                if active_profile is not None
                else ["NO_ACTIVE_PROFILE"]
            ),
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
        return AdaptationDriftResponse(
            items=await self.repository.load_adaptive_drift_states(
                symbol=symbol,
                regime_label=regime_label,
                limit=limit,
            )
        )

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
        return AdaptationPerformanceResponse(
            items=await self.repository.load_adaptive_performance_windows(
                execution_mode=execution_mode,
                symbol=symbol,
                regime_label=regime_label,
                limit=limit,
            )
        )

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
