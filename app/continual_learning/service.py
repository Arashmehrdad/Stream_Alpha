"""Read and manage the additive M21 continual learning data layer."""

# pylint: disable=too-many-arguments

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
    ContinualLearningDriftCapsResponse,
    ContinualLearningEventRecord,
    ContinualLearningEventsResponse,
    ContinualLearningExperimentsResponse,
    ContinualLearningProfileRecord,
    ContinualLearningProfilesResponse,
    ContinualLearningPromotionDecisionRecord,
    ContinualLearningPromotionsResponse,
    ContinualLearningSummaryResponse,
)
from app.trading.repository import TradingRepository


class ContinualLearningService:
    """Read-only M21 continual learning service with explicit rollback support."""

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
        profiles = await self.repository.load_continual_learning_profiles(limit=50)
        promotions = await self.repository.load_continual_learning_promotion_decisions(limit=1)
        events = await self.repository.load_continual_learning_events(limit=1)
        drift_cap = await self.repository.load_latest_continual_learning_drift_cap(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )
        active_profile = await self.repository.load_active_continual_learning_profile(
            execution_mode=execution_mode,
            symbol=symbol,
            regime_label=regime_label,
        )
        summary = ContinualLearningSummaryResponse(
            enabled=self.config.enabled,
            active_profile_count=sum(1 for item in profiles if item.status == "ACTIVE"),
            active_profile_id=(None if active_profile is None else active_profile.profile_id),
            continual_learning_status=("ACTIVE" if active_profile is not None else "IDLE"),
            active_candidate_type=(
                None if active_profile is None else active_profile.candidate_type
            ),
            latest_drift_cap_status=(None if drift_cap is None else drift_cap.status),
            latest_promotion_decision=(
                None if not promotions else promotions[0].decision
            ),
            latest_event_type=(None if not events else events[0].event_type),
            reason_codes=(
                ["ACTIVE_PROFILE_PRESENT"]
                if active_profile is not None
                else ["NO_ACTIVE_CONTINUAL_LEARNING_PROFILE"]
            ),
        )
        self._write_summary_artifact(summary)
        return summary

    async def experiments(self, *, limit: int = 50) -> ContinualLearningExperimentsResponse:
        """Return the read-only continual-learning experiments payload."""
        if not await self._ensure_repository_ready():
            return ContinualLearningExperimentsResponse()
        return ContinualLearningExperimentsResponse(
            items=await self.repository.load_continual_learning_experiments(limit=limit)
        )

    async def profiles(self, *, limit: int = 50) -> ContinualLearningProfilesResponse:
        """Return the read-only continual-learning profiles payload."""
        if not await self._ensure_repository_ready():
            return ContinualLearningProfilesResponse()
        return ContinualLearningProfilesResponse(
            items=await self.repository.load_continual_learning_profiles(limit=limit)
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
            reason_codes=["CONTINUAL_LEARNING_ROLLBACK_TARGET_ACTIVATED"],
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
            reason_code="CONTINUAL_LEARNING_ROLLBACK_TARGET_ACTIVATED",
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
