"""Read and manage the additive M21 continual learning data layer."""

# pylint: disable=too-many-arguments,too-many-lines

from __future__ import annotations

from pathlib import Path

from app.common.time import to_rfc3339, utc_now
from app.continual_learning.artifacts import (
    append_jsonl_artifact,
    ensure_continual_learning_artifact_root,
    write_json_artifact,
    write_markdown_artifact,
)
from app.continual_learning.config import (
    ContinualLearningConfig,
    default_continual_learning_config_path,
    load_continual_learning_config,
)
from app.continual_learning.schemas import (
    ContinualLearningContextPayload,
    ContinualLearningDriftCapRecord,
    ContinualLearningDriftCapsResponse,
    ContinualLearningEventRecord,
    ContinualLearningEventsResponse,
    ContinualLearningExperimentRecord,
    ContinualLearningExperimentsResponse,
    ContinualLearningProfileRecord,
    ContinualLearningProfilesResponse,
    ContinualLearningPromoteProfileRequest,
    ContinualLearningPromotionDecisionRecord,
    ContinualLearningRollbackRequest,
    ContinualLearningPromotionsResponse,
    ContinualLearningSummaryResponse,
    ContinualLearningWorkflowResponse,
)
from app.trading.repository import TradingRepository


CONTINUAL_LEARNING_OPERATOR_CONFIRMATION_REQUIRED = (
    "CONTINUAL_LEARNING_OPERATOR_CONFIRMATION_REQUIRED"
)
CONTINUAL_LEARNING_REPOSITORY_UNAVAILABLE = (
    "CONTINUAL_LEARNING_REPOSITORY_UNAVAILABLE"
)
CONTINUAL_LEARNING_PROFILE_NOT_FOUND = "CONTINUAL_LEARNING_PROFILE_NOT_FOUND"
CONTINUAL_LEARNING_LIVE_ELIGIBILITY_BLOCKED = (
    "CONTINUAL_LEARNING_LIVE_ELIGIBILITY_BLOCKED"
)
CONTINUAL_LEARNING_SHADOW_CHALLENGER_LIVE_BLOCKED = (
    "CONTINUAL_LEARNING_SHADOW_CHALLENGER_LIVE_BLOCKED"
)
CONTINUAL_LEARNING_DRIFT_CAP_BREACHED = "CONTINUAL_LEARNING_DRIFT_CAP_BREACHED"
CONTINUAL_LEARNING_BLOCKED_BY_HEALTH_STATUS = (
    "CONTINUAL_LEARNING_BLOCKED_BY_HEALTH_STATUS"
)
CONTINUAL_LEARNING_BLOCKED_BY_FRESHNESS_STATUS = (
    "CONTINUAL_LEARNING_BLOCKED_BY_FRESHNESS_STATUS"
)
CONTINUAL_LEARNING_PROMOTION_APPLIED = "CONTINUAL_LEARNING_PROMOTION_APPLIED"
CONTINUAL_LEARNING_NO_ACTIVE_PROFILE_FOR_ROLLBACK = (
    "CONTINUAL_LEARNING_NO_ACTIVE_PROFILE_FOR_ROLLBACK"
)
CONTINUAL_LEARNING_NO_ROLLBACK_TARGET = "CONTINUAL_LEARNING_NO_ROLLBACK_TARGET"
CONTINUAL_LEARNING_ROLLBACK_APPLIED = "CONTINUAL_LEARNING_ROLLBACK_APPLIED"


class ContinualLearningService:
    """Guarded M21 continual-learning service with read and operator workflow support."""

    def __init__(
        self,
        *,
        repository: TradingRepository | None = None,
        config: ContinualLearningConfig | None = None,
    ) -> None:
        self.config = config or load_continual_learning_config(
            default_continual_learning_config_path()
        )
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

    async def load_active_profile(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
    ) -> ContinualLearningProfileRecord | None:
        """Load the best matching active continual-learning profile by scope."""
        if not await self._ensure_repository_ready():
            return None
        return await self.repository.load_active_continual_learning_profile(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )

    async def load_profile(
        self,
        *,
        profile_id: str,
    ) -> ContinualLearningProfileRecord | None:
        """Load one persisted continual-learning profile by id."""
        if not await self._ensure_repository_ready():
            return None
        return await self.repository.load_continual_learning_profile(profile_id=profile_id)

    async def resolve_runtime_context(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        health_overall_status: str | None = None,
        freshness_status: str | None = None,
    ) -> ContinualLearningContextPayload:
        """Resolve compact read-only continual-learning runtime context for M4/M14."""
        if not self.config.enabled:
            return ContinualLearningContextPayload(
                enabled=False,
                reason_codes=["CONTINUAL_LEARNING_DISABLED"],
            )

        if not await self._ensure_repository_ready():
            return ContinualLearningContextPayload(
                enabled=True,
                active_profile_id=None,
                live_eligible=False,
                frozen_by_health_gate=False,
                reason_codes=["CONTINUAL_LEARNING_REPOSITORY_UNAVAILABLE"],
            )

        active_profile = await self.repository.load_active_continual_learning_profile(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )
        drift_cap = await self.repository.load_latest_continual_learning_drift_cap(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )
        promotions = await self.repository.load_continual_learning_promotion_decisions(
            limit=1
        )

        frozen_by_health_gate = (
            (health_overall_status is not None and health_overall_status != "HEALTHY")
            or (freshness_status is not None and freshness_status != "FRESH")
        )
        latest_decision = None if not promotions else promotions[0].decision

        if active_profile is None:
            return ContinualLearningContextPayload(
                enabled=True,
                active_profile_id=None,
                live_eligible=False,
                drift_cap_status=None if drift_cap is None else drift_cap.status,
                latest_promotion_decision=latest_decision,
                frozen_by_health_gate=frozen_by_health_gate,
                reason_codes=["NO_ACTIVE_CONTINUAL_LEARNING_PROFILE"],
            )

        return _runtime_context_from_profile(
            profile=active_profile,
            drift_cap_status=None if drift_cap is None else drift_cap.status,
            latest_promotion_decision=latest_decision,
            frozen_by_health_gate=frozen_by_health_gate,
        )

    # pylint: disable=too-many-locals
    async def summary(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
    ) -> ContinualLearningSummaryResponse:
        """Return the read-only M21 continual-learning summary payload."""
        if not await self._ensure_repository_ready():
            return ContinualLearningSummaryResponse(
                enabled=self.config.enabled,
                active_profile_count=0,
                continual_learning_status="UNAVAILABLE",
                reason_codes=["CONTINUAL_LEARNING_REPOSITORY_UNAVAILABLE"],
            )
        aggregate_scope = _is_aggregate_scope(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )
        profiles = await self.repository.load_continual_learning_profiles(limit=500)
        matching_profiles = _filter_profiles_for_scope(
            profiles,
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )
        active_profiles = [item for item in matching_profiles if item.status == "ACTIVE"]
        promotions = await self.repository.load_continual_learning_promotion_decisions(limit=1)
        events = await self.repository.load_continual_learning_events(limit=1)
        drift_status = None
        active_profile_id = None
        active_candidate_type = None

        if aggregate_scope:
            drift_caps = await self.repository.load_all_continual_learning_drift_caps(limit=500)
            matching_drift_caps = _filter_drift_caps_for_scope(
                drift_caps,
                execution_mode=execution_mode,
                symbol=symbol,
                regime_label=regime_label,
            )
            drift_status = _worst_drift_cap_status(matching_drift_caps)
            if len(active_profiles) == 1:
                active_profile_id = active_profiles[0].profile_id
                active_candidate_type = active_profiles[0].candidate_type
        else:
            active_profile = await self.repository.load_active_continual_learning_profile(
                execution_mode=execution_mode,
                symbol=symbol,
                regime_label=regime_label,
            )
            if active_profile is not None:
                active_profile_id = active_profile.profile_id
                active_candidate_type = active_profile.candidate_type
            drift_cap = await self.repository.load_latest_continual_learning_drift_cap(
                execution_mode=execution_mode,
                symbol=symbol,
                regime_label=regime_label,
            )
            drift_status = None if drift_cap is None else drift_cap.status

        reason_codes: list[str] = []
        if aggregate_scope:
            reason_codes.append("AGGREGATED_SCOPE_SUMMARY")
        reason_codes.append(
            "ACTIVE_PROFILE_PRESENT"
            if active_profiles
            else "NO_ACTIVE_CONTINUAL_LEARNING_PROFILE"
        )

        summary = ContinualLearningSummaryResponse(
            enabled=self.config.enabled,
            active_profile_count=len(active_profiles),
            active_profile_id=active_profile_id,
            continual_learning_status=("ACTIVE" if active_profiles else "IDLE"),
            active_candidate_type=active_candidate_type,
            latest_drift_cap_status=drift_status,
            latest_promotion_decision=(
                None if not promotions else promotions[0].decision
            ),
            latest_event_type=(None if not events else events[0].event_type),
            reason_codes=reason_codes,
        )
        self._write_summary_artifact(summary)
        return summary
    # pylint: enable=too-many-locals

    async def experiments(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        limit: int = 50,
    ) -> ContinualLearningExperimentsResponse:
        """Return the read-only continual-learning experiments payload."""
        if not await self._ensure_repository_ready():
            return ContinualLearningExperimentsResponse()
        items = await self.repository.load_continual_learning_experiments(limit=500)
        matching_items = _filter_experiments_for_scope(
            items,
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )
        return ContinualLearningExperimentsResponse(
            items=matching_items[:limit]
        )

    async def profiles(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        limit: int = 50,
    ) -> ContinualLearningProfilesResponse:
        """Return the read-only continual-learning profiles payload."""
        if not await self._ensure_repository_ready():
            return ContinualLearningProfilesResponse()
        items = await self.repository.load_continual_learning_profiles(limit=500)
        matching_items = _filter_profiles_for_scope(
            items,
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )
        return ContinualLearningProfilesResponse(
            items=matching_items[:limit]
        )

    async def drift_caps(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        limit: int = 50,
    ) -> ContinualLearningDriftCapsResponse:
        """Return the read-only continual-learning drift-cap payload."""
        if not await self._ensure_repository_ready():
            return ContinualLearningDriftCapsResponse()
        if _is_aggregate_scope(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        ):
            unscoped_items = await self.repository.load_all_continual_learning_drift_caps(
                limit=500
            )
            items = _filter_drift_caps_for_scope(
                unscoped_items,
                execution_mode=execution_mode,
                symbol=symbol,
                regime_label=regime_label,
            )[:limit]
        else:
            items = await self.repository.load_continual_learning_drift_caps(
                execution_mode=execution_mode,
                symbol=symbol,
                regime_label=regime_label,
                limit=limit,
            )
        self._write_drift_caps_summary_artifact(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
            items=items,
        )
        return ContinualLearningDriftCapsResponse(items=items)

    async def promotions(self, *, limit: int = 50) -> ContinualLearningPromotionsResponse:
        """Return the read-only continual-learning promotions payload."""
        if not await self._ensure_repository_ready():
            return ContinualLearningPromotionsResponse()
        return ContinualLearningPromotionsResponse(
            items=await self.repository.load_continual_learning_promotion_decisions(
                limit=limit
            )
        )

    async def events(self, *, limit: int = 50) -> ContinualLearningEventsResponse:
        """Return the read-only continual-learning events payload."""
        if not await self._ensure_repository_ready():
            return ContinualLearningEventsResponse()
        return ContinualLearningEventsResponse(
            items=await self.repository.load_continual_learning_events(limit=limit)
        )

    async def rollback_active_profile(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        decision_id: str,
        summary_text: str,
    ) -> ContinualLearningPromotionDecisionRecord:
        """Persist and apply an explicit M21 runtime rollback to the configured target."""
        if not await self._ensure_repository_ready():
            raise RuntimeError("Continual learning repository unavailable for rollback")
        active_profile = await self.repository.load_active_continual_learning_profile(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )
        if active_profile is None:
            raise RuntimeError("No active continual-learning profile is available for rollback")
        if active_profile.rollback_target_profile_id is None:
            raise RuntimeError("Active continual-learning profile has no rollback target")
        rollback_target = await self.repository.load_continual_learning_profile(
            profile_id=active_profile.rollback_target_profile_id,
        )
        if rollback_target is None:
            raise RuntimeError("Rollback target continual-learning profile was not found")

        decided_at = utc_now()
        decision = ContinualLearningPromotionDecisionRecord(
            decision_id=decision_id,
            target_type="PROFILE",
            target_id=rollback_target.profile_id,
            incumbent_id=active_profile.profile_id,
            candidate_type=rollback_target.candidate_type,
            decision="ROLLBACK",
            live_eligible_after_decision=rollback_target.live_eligible,
            metrics_delta_json={
                "rolled_back_profile_id": active_profile.profile_id,
                "restored_profile_id": rollback_target.profile_id,
            },
            safety_checks_json={"runtime_rollback": True},
            research_integrity_json={
                "rollback_target_profile_id": rollback_target.profile_id,
                "source_experiment_id": rollback_target.source_experiment_id,
            },
            reason_codes=[CONTINUAL_LEARNING_ROLLBACK_APPLIED],
            summary_text=summary_text,
            decided_at=decided_at,
        )
        await self.repository.rollback_continual_learning_profile(
            active_profile_id=active_profile.profile_id,
            rollback_target_profile_id=rollback_target.profile_id,
            changed_at=decided_at,
        )
        await self.repository.save_continual_learning_promotion_decision(decision)
        event = ContinualLearningEventRecord(
            event_id=f"event:{decision_id}",
            event_type="ROLLBACK_APPLIED",
            profile_id=rollback_target.profile_id,
            decision_id=decision_id,
            reason_code=CONTINUAL_LEARNING_ROLLBACK_APPLIED,
            payload_json={
                "execution_mode": execution_mode,
                "symbol": symbol,
                "regime_label": regime_label,
                "restored_profile_id": rollback_target.profile_id,
            },
            created_at=decided_at,
        )
        await self.repository.save_continual_learning_event(event)
        restored_profile = await self.repository.load_continual_learning_profile(
            profile_id=rollback_target.profile_id,
        )
        if restored_profile is None:
            raise RuntimeError("Rollback target could not be reloaded after rollback")
        self.write_profile_artifacts(profile=restored_profile, latest_promotion=decision)
        self._append_event_artifact(event)
        return decision

    async def promote_profile(  # pylint: disable=too-many-locals
        self,
        request: ContinualLearningPromoteProfileRequest,
        health_overall_status: str | None = None,
        freshness_status: str | None = None,
    ) -> ContinualLearningWorkflowResponse:
        """Apply one guarded M21 profile promotion against persisted profile truth only."""
        if not request.operator_confirmed:
            return _build_workflow_response(
                decision_id=request.decision_id,
                success=False,
                blocked=True,
                decision="HOLD",
                target_profile_id=request.profile_id,
                reason_codes=[CONTINUAL_LEARNING_OPERATOR_CONFIRMATION_REQUIRED],
                summary_text=request.summary_text,
                health_overall_status=health_overall_status,
                freshness_status=freshness_status,
            )
        if not await self._ensure_repository_ready():
            return _build_workflow_response(
                decision_id=request.decision_id,
                success=False,
                blocked=True,
                decision="HOLD",
                target_profile_id=request.profile_id,
                reason_codes=[CONTINUAL_LEARNING_REPOSITORY_UNAVAILABLE],
                summary_text=request.summary_text,
                health_overall_status=health_overall_status,
                freshness_status=freshness_status,
            )

        profile = await self.repository.load_continual_learning_profile(
            profile_id=request.profile_id,
        )
        if profile is None:
            return _build_workflow_response(
                decision_id=request.decision_id,
                success=False,
                blocked=True,
                decision="HOLD",
                target_profile_id=request.profile_id,
                reason_codes=[CONTINUAL_LEARNING_PROFILE_NOT_FOUND],
                summary_text=request.summary_text,
                health_overall_status=health_overall_status,
                freshness_status=freshness_status,
            )

        incumbent = await self.repository.load_active_continual_learning_profile(
            execution_mode=profile.execution_mode_scope,
            symbol=profile.symbol_scope,
            regime_label=profile.regime_scope,
        )
        drift_cap = await self.repository.load_latest_continual_learning_drift_cap(
            execution_mode=profile.execution_mode_scope,
            symbol=profile.symbol_scope,
            regime_label=profile.regime_scope,
        )
        drift_status = None if drift_cap is None else drift_cap.status

        block_reasons: list[str] = []
        if request.requested_promotion_stage == "LIVE_ELIGIBLE":
            if profile.candidate_type in self.config.shadow_only_candidate_types:
                block_reasons.append(CONTINUAL_LEARNING_SHADOW_CHALLENGER_LIVE_BLOCKED)
            if profile.candidate_type not in self.config.live_eligible_candidate_types:
                block_reasons.append(CONTINUAL_LEARNING_LIVE_ELIGIBILITY_BLOCKED)
            if not profile.live_eligible:
                block_reasons.append(CONTINUAL_LEARNING_LIVE_ELIGIBILITY_BLOCKED)
        if drift_status == "BREACHED":
            block_reasons.append(CONTINUAL_LEARNING_DRIFT_CAP_BREACHED)
        if health_overall_status is not None and health_overall_status != "HEALTHY":
            block_reasons.append(CONTINUAL_LEARNING_BLOCKED_BY_HEALTH_STATUS)
        if freshness_status is not None and freshness_status != "FRESH":
            block_reasons.append(CONTINUAL_LEARNING_BLOCKED_BY_FRESHNESS_STATUS)

        if block_reasons:
            reason_codes = _merge_reason_codes(block_reasons, request.reason_codes)
            event_id = await self._persist_blocked_workflow(
                decision_id=request.decision_id,
                target_id=profile.profile_id,
                incumbent_id=(None if incumbent is None else incumbent.profile_id),
                candidate_type=profile.candidate_type,
                event_type="PROMOTION_BLOCKED",
                reason_codes=reason_codes,
                summary_text=request.summary_text,
                payload_json={
                    "profile_id": profile.profile_id,
                    "requested_promotion_stage": request.requested_promotion_stage,
                    "health_overall_status": health_overall_status,
                    "freshness_status": freshness_status,
                    "drift_cap_status": drift_status,
                },
                live_eligible_after_decision=profile.live_eligible,
            )
            return _build_workflow_response(
                decision_id=request.decision_id,
                success=False,
                blocked=True,
                decision="HOLD",
                target_profile_id=profile.profile_id,
                incumbent_profile_id=(None if incumbent is None else incumbent.profile_id),
                promotion_stage_after=profile.promotion_stage,
                live_eligible_after=profile.live_eligible,
                drift_cap_status=drift_status,
                health_overall_status=health_overall_status,
                freshness_status=freshness_status,
                event_id=event_id,
                reason_codes=reason_codes,
                summary_text=request.summary_text,
            )

        changed_at = utc_now()
        live_eligible_after = request.requested_promotion_stage == "LIVE_ELIGIBLE"
        incumbent_profile_id = await self.repository.promote_continual_learning_profile(
            target_profile_id=profile.profile_id,
            execution_mode_scope=profile.execution_mode_scope,
            symbol_scope=profile.symbol_scope,
            regime_scope=profile.regime_scope,
            promotion_stage=request.requested_promotion_stage,
            live_eligible=live_eligible_after,
            changed_at=changed_at,
        )
        updated_profile = await self.repository.load_continual_learning_profile(
            profile_id=profile.profile_id,
        )
        if updated_profile is None:
            raise RuntimeError("Promoted continual-learning profile could not be reloaded")

        reason_codes = _merge_reason_codes(
            [CONTINUAL_LEARNING_PROMOTION_APPLIED],
            request.reason_codes,
        )
        decision = ContinualLearningPromotionDecisionRecord(
            decision_id=request.decision_id,
            target_type="PROFILE",
            target_id=updated_profile.profile_id,
            incumbent_id=incumbent_profile_id,
            candidate_type=updated_profile.candidate_type,
            decision="PROMOTE",
            live_eligible_after_decision=updated_profile.live_eligible,
            metrics_delta_json={
                "promotion_stage_after": updated_profile.promotion_stage,
            },
            safety_checks_json={
                "health_overall_status": health_overall_status,
                "freshness_status": freshness_status,
                "drift_cap_status": drift_status,
            },
            research_integrity_json={
                "source_experiment_id": updated_profile.source_experiment_id,
                "baseline_target_id": updated_profile.baseline_target_id,
            },
            reason_codes=reason_codes,
            summary_text=request.summary_text,
            decided_at=changed_at,
        )
        await self.repository.save_continual_learning_promotion_decision(decision)
        event = ContinualLearningEventRecord(
            event_id=f"event:{request.decision_id}",
            event_type="PROMOTION_APPLIED",
            profile_id=updated_profile.profile_id,
            experiment_id=updated_profile.source_experiment_id,
            decision_id=request.decision_id,
            reason_code=CONTINUAL_LEARNING_PROMOTION_APPLIED,
            payload_json={
                "requested_promotion_stage": request.requested_promotion_stage,
                "incumbent_profile_id": incumbent_profile_id,
                "health_overall_status": health_overall_status,
                "freshness_status": freshness_status,
                "drift_cap_status": drift_status,
            },
            created_at=changed_at,
        )
        await self.repository.save_continual_learning_event(event)
        self.write_profile_artifacts(profile=updated_profile, latest_promotion=decision)
        self._append_event_artifact(event)
        return _build_workflow_response(
            decision_id=request.decision_id,
            success=True,
            blocked=False,
            decision="PROMOTE",
            target_profile_id=updated_profile.profile_id,
            incumbent_profile_id=incumbent_profile_id,
            promotion_stage_after=updated_profile.promotion_stage,
            live_eligible_after=updated_profile.live_eligible,
            drift_cap_status=drift_status,
            health_overall_status=health_overall_status,
            freshness_status=freshness_status,
            event_id=event.event_id,
            reason_codes=reason_codes,
            summary_text=request.summary_text,
        )

    async def rollback_profile(
        self,
        request: ContinualLearningRollbackRequest,
        health_overall_status: str | None = None,
        freshness_status: str | None = None,
    ) -> ContinualLearningWorkflowResponse:
        """Apply one guarded M21 rollback using the accepted explicit rollback path."""
        if not request.operator_confirmed:
            return _build_workflow_response(
                decision_id=request.decision_id,
                success=False,
                blocked=True,
                decision="HOLD",
                reason_codes=[CONTINUAL_LEARNING_OPERATOR_CONFIRMATION_REQUIRED],
                summary_text=request.summary_text,
                health_overall_status=health_overall_status,
                freshness_status=freshness_status,
            )
        if not await self._ensure_repository_ready():
            return _build_workflow_response(
                decision_id=request.decision_id,
                success=False,
                blocked=True,
                decision="HOLD",
                reason_codes=[CONTINUAL_LEARNING_REPOSITORY_UNAVAILABLE],
                summary_text=request.summary_text,
                health_overall_status=health_overall_status,
                freshness_status=freshness_status,
            )

        active_profile = await self.repository.load_active_continual_learning_profile(
            execution_mode=request.execution_mode,
            symbol=request.symbol,
            regime_label=request.regime_label,
        )
        if active_profile is None:
            return _build_workflow_response(
                decision_id=request.decision_id,
                success=False,
                blocked=True,
                decision="HOLD",
                reason_codes=[CONTINUAL_LEARNING_NO_ACTIVE_PROFILE_FOR_ROLLBACK],
                summary_text=request.summary_text,
                health_overall_status=health_overall_status,
                freshness_status=freshness_status,
            )

        drift_cap = await self.repository.load_latest_continual_learning_drift_cap(
            execution_mode=request.execution_mode,
            symbol=request.symbol,
            regime_label=request.regime_label,
        )
        drift_status = None if drift_cap is None else drift_cap.status
        block_reasons: list[str] = []
        if health_overall_status is not None and health_overall_status != "HEALTHY":
            block_reasons.append(CONTINUAL_LEARNING_BLOCKED_BY_HEALTH_STATUS)
        if freshness_status is not None and freshness_status != "FRESH":
            block_reasons.append(CONTINUAL_LEARNING_BLOCKED_BY_FRESHNESS_STATUS)
        if active_profile.rollback_target_profile_id is None:
            block_reasons.append(CONTINUAL_LEARNING_NO_ROLLBACK_TARGET)

        if block_reasons:
            event_id = await self._persist_blocked_workflow(
                decision_id=request.decision_id,
                target_id=active_profile.profile_id,
                incumbent_id=active_profile.profile_id,
                candidate_type=active_profile.candidate_type,
                event_type="ROLLBACK_BLOCKED",
                reason_codes=block_reasons,
                summary_text=request.summary_text,
                payload_json={
                    "execution_mode": request.execution_mode,
                    "symbol": request.symbol,
                    "regime_label": request.regime_label,
                    "rollback_target_profile_id": active_profile.rollback_target_profile_id,
                    "health_overall_status": health_overall_status,
                    "freshness_status": freshness_status,
                    "drift_cap_status": drift_status,
                },
                live_eligible_after_decision=active_profile.live_eligible,
            )
            return _build_workflow_response(
                decision_id=request.decision_id,
                success=False,
                blocked=True,
                decision="HOLD",
                target_profile_id=active_profile.rollback_target_profile_id,
                incumbent_profile_id=active_profile.profile_id,
                promotion_stage_after=active_profile.promotion_stage,
                live_eligible_after=active_profile.live_eligible,
                drift_cap_status=drift_status,
                health_overall_status=health_overall_status,
                freshness_status=freshness_status,
                event_id=event_id,
                reason_codes=block_reasons,
                summary_text=request.summary_text,
            )

        decision = await self.rollback_active_profile(
            execution_mode=request.execution_mode,
            symbol=request.symbol,
            regime_label=request.regime_label,
            decision_id=request.decision_id,
            summary_text=request.summary_text,
        )
        restored_profile = await self.repository.load_continual_learning_profile(
            profile_id=decision.target_id,
        )
        return _build_workflow_response(
            decision_id=request.decision_id,
            success=True,
            blocked=False,
            decision="ROLLBACK",
            target_profile_id=decision.target_id,
            incumbent_profile_id=decision.incumbent_id,
            promotion_stage_after=(
                None if restored_profile is None else restored_profile.promotion_stage
            ),
            live_eligible_after=(
                None if restored_profile is None else restored_profile.live_eligible
            ),
            drift_cap_status=drift_status,
            health_overall_status=health_overall_status,
            freshness_status=freshness_status,
            event_id=f"event:{request.decision_id}",
            reason_codes=[CONTINUAL_LEARNING_ROLLBACK_APPLIED],
            summary_text=request.summary_text,
        )

    def write_profile_artifacts(
        self,
        *,
        profile: ContinualLearningProfileRecord,
        latest_promotion: ContinualLearningPromotionDecisionRecord | None = None,
    ) -> None:
        """Write deterministic profile and promotion artifacts."""
        ensure_continual_learning_artifact_root(self.config.artifacts.root_dir)
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
                f"# Continual Learning Profile {profile.profile_id}",
                "",
                f"- candidate_type: {profile.candidate_type}",
                f"- status: {profile.status}",
                f"- execution_mode_scope: {profile.execution_mode_scope}",
                f"- symbol_scope: {profile.symbol_scope}",
                f"- regime_scope: {profile.regime_scope}",
                f"- baseline_target_type: {profile.baseline_target_type}",
                f"- baseline_target_id: {profile.baseline_target_id}",
                f"- source_experiment_id: {profile.source_experiment_id}",
                f"- promotion_stage: {profile.promotion_stage}",
                f"- live_eligible: {profile.live_eligible}",
                f"- rollback_target_profile_id: {profile.rollback_target_profile_id}",
            ],
        )
        if latest_promotion is not None:
            append_jsonl_artifact(
                self.config.artifacts.promotions_history_path,
                latest_promotion.model_dump(mode="json"),
            )

    async def _ensure_repository_ready(self) -> bool:
        """Best-effort repository initialization for additive continual-learning reads."""
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

    def _write_summary_artifact(self, summary: ContinualLearningSummaryResponse) -> None:
        write_json_artifact(
            self.config.artifacts.summary_path,
            {
                "generated_at": to_rfc3339(utc_now()),
                **summary.model_dump(mode="json"),
            },
        )

    def _write_drift_caps_summary_artifact(
        self,
        *,
        execution_mode: str,
        symbol: str,
        regime_label: str,
        items: list,
    ) -> None:
        write_json_artifact(
            self.config.artifacts.drift_caps_summary_path,
            {
                "generated_at": to_rfc3339(utc_now()),
                "execution_mode": execution_mode,
                "symbol": symbol,
                "regime_label": regime_label,
                "item_count": len(items),
                "items": [item.model_dump(mode="json") for item in items],
            },
        )

    def _append_event_artifact(self, event: ContinualLearningEventRecord) -> None:
        append_jsonl_artifact(
            self.config.artifacts.events_history_path,
            event.model_dump(mode="json"),
        )

    async def _persist_blocked_workflow(
        self,
        *,
        decision_id: str,
        target_id: str,
        incumbent_id: str | None,
        candidate_type: str,
        event_type: str,
        reason_codes: list[str],
        summary_text: str,
        payload_json: dict[str, object],
        live_eligible_after_decision: bool,
    ) -> str:
        """Persist one blocked workflow decision and audit event when profile truth exists."""
        decided_at = utc_now()
        decision = ContinualLearningPromotionDecisionRecord(
            decision_id=decision_id,
            target_type="PROFILE",
            target_id=target_id,
            incumbent_id=incumbent_id,
            candidate_type=candidate_type,
            decision="HOLD",
            live_eligible_after_decision=live_eligible_after_decision,
            safety_checks_json=payload_json,
            reason_codes=reason_codes,
            summary_text=summary_text,
            decided_at=decided_at,
        )
        await self.repository.save_continual_learning_promotion_decision(decision)
        event = ContinualLearningEventRecord(
            event_id=f"event:{decision_id}",
            event_type=event_type,
            profile_id=target_id,
            decision_id=decision_id,
            reason_code=reason_codes[0],
            payload_json=payload_json,
            created_at=decided_at,
        )
        await self.repository.save_continual_learning_event(event)
        self._append_event_artifact(event)
        return event.event_id


def _runtime_context_from_profile(
    *,
    profile: ContinualLearningProfileRecord,
    drift_cap_status,
    latest_promotion_decision,
    frozen_by_health_gate: bool,
) -> ContinualLearningContextPayload:
    reason_codes = ["ACTIVE_PROFILE_PRESENT"]
    if frozen_by_health_gate:
        reason_codes.append("CONTINUAL_LEARNING_FROZEN_BY_HEALTH_GATE")
    return ContinualLearningContextPayload(
        enabled=True,
        active_profile_id=profile.profile_id,
        candidate_type=profile.candidate_type,
        promotion_stage=profile.promotion_stage,
        live_eligible=profile.live_eligible,
        baseline_target_type=profile.baseline_target_type,
        baseline_target_id=profile.baseline_target_id,
        source_experiment_id=profile.source_experiment_id,
        drift_cap_status=drift_cap_status,
        latest_promotion_decision=latest_promotion_decision,
        frozen_by_health_gate=frozen_by_health_gate,
        reason_codes=reason_codes,
    )


def _merge_reason_codes(
    stable_codes: list[str],
    operator_codes: list[str],
) -> list[str]:
    merged: list[str] = []
    for code in [*stable_codes, *operator_codes]:
        if code not in merged:
            merged.append(code)
    return merged


def _build_workflow_response(
    *,
    decision_id: str,
    success: bool,
    blocked: bool,
    summary_text: str,
    decision: str | None = None,
    target_profile_id: str | None = None,
    incumbent_profile_id: str | None = None,
    promotion_stage_after: str | None = None,
    live_eligible_after: bool | None = None,
    drift_cap_status: str | None = None,
    health_overall_status: str | None = None,
    freshness_status: str | None = None,
    event_id: str | None = None,
    reason_codes: list[str] | None = None,
) -> ContinualLearningWorkflowResponse:
    return ContinualLearningWorkflowResponse(
        success=success,
        blocked=blocked,
        decision_id=decision_id,
        decision=decision,
        target_profile_id=target_profile_id,
        incumbent_profile_id=incumbent_profile_id,
        promotion_stage_after=promotion_stage_after,
        live_eligible_after=live_eligible_after,
        drift_cap_status=drift_cap_status,
        health_overall_status=health_overall_status,
        freshness_status=freshness_status,
        event_id=event_id,
        reason_codes=[] if reason_codes is None else reason_codes,
        summary_text=summary_text,
    )


def _is_aggregate_scope(*, execution_mode: str, symbol: str, regime_label: str) -> bool:
    return execution_mode == "ALL" or symbol == "ALL" or regime_label == "ALL"


def _scope_matches(query_value: str, stored_value: str) -> bool:
    if query_value == "ALL":
        return True
    return stored_value in {query_value, "ALL"}


def _filter_profiles_for_scope(
    profiles: list[ContinualLearningProfileRecord],
    *,
    execution_mode: str,
    symbol: str,
    regime_label: str,
) -> list[ContinualLearningProfileRecord]:
    return [
        item
        for item in profiles
        if _scope_matches(execution_mode, item.execution_mode_scope)
        and _scope_matches(symbol, item.symbol_scope)
        and _scope_matches(regime_label, item.regime_scope)
    ]


def _filter_drift_caps_for_scope(
    drift_caps: list[ContinualLearningDriftCapRecord],
    *,
    execution_mode: str,
    symbol: str,
    regime_label: str,
) -> list[ContinualLearningDriftCapRecord]:
    return [
        item
        for item in drift_caps
        if _scope_matches(execution_mode, item.execution_mode_scope)
        and _scope_matches(symbol, item.symbol_scope)
        and _scope_matches(regime_label, item.regime_scope)
    ]


def _filter_experiments_for_scope(
    experiments: list[ContinualLearningExperimentRecord],
    *,
    execution_mode: str,
    symbol: str,
    regime_label: str,
) -> list[ContinualLearningExperimentRecord]:
    return [
        item
        for item in experiments
        if _scope_matches(execution_mode, item.execution_mode_scope)
        and _scope_matches(symbol, item.symbol_scope)
        and _scope_matches(regime_label, item.regime_scope)
    ]


def _worst_drift_cap_status(
    drift_caps: list[ContinualLearningDriftCapRecord],
) -> str | None:
    if not drift_caps:
        return None
    severity = {
        "HEALTHY": 0,
        "WATCH": 1,
        "BREACHED": 2,
    }
    return max(drift_caps, key=lambda item: severity.get(item.status, -1)).status
