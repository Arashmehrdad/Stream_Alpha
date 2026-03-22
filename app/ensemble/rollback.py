"""Internal explicit rollback helpers for Packet 2 ensemble profile activation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.common.time import utc_now
from app.ensemble.schemas import EnsemblePromotionDecisionRecord


async def rollback_active_profile(  # pylint: disable=too-many-arguments
    repository: Any,
    *,
    active_profile_id: str,
    rollback_target_profile_id: str,
    decision_id: str,
    summary_text: str,
    decided_at: datetime | None = None,
) -> EnsemblePromotionDecisionRecord:
    """Roll one active Packet 2 profile back to an explicit rollback target."""
    timestamp = utc_now() if decided_at is None else decided_at
    active_profile = await repository.load_ensemble_profile(profile_id=active_profile_id)
    if active_profile is None:
        raise ValueError(f"Active ensemble profile was not found: {active_profile_id}")
    rollback_target = await repository.load_ensemble_profile(
        profile_id=rollback_target_profile_id,
    )
    if rollback_target is None:
        raise ValueError(
            f"Rollback target ensemble profile was not found: {rollback_target_profile_id}",
        )

    await repository.save_ensemble_profile(
        active_profile.model_copy(
            update={
                "status": "ROLLED_BACK",
                "approval_stage": "SUPERSEDED",
                "superseded_at": timestamp,
            }
        )
    )
    await repository.save_ensemble_profile(
        rollback_target.model_copy(
            update={
                "status": "ACTIVE",
                "approval_stage": "ACTIVATED",
                "activated_at": timestamp,
                "superseded_at": None,
            }
        )
    )

    decision = EnsemblePromotionDecisionRecord(
        decision_id=decision_id,
        target_type="PROFILE",
        target_id=rollback_target_profile_id,
        incumbent_id=active_profile_id,
        decision="ROLLBACK",
        metrics_delta_json={
            "rolled_back_profile_id": active_profile_id,
        },
        safety_checks_json={
            "runtime_rollback": True,
        },
        reason_codes=["ENSEMBLE_ROLLBACK_TARGET_ACTIVATED"],
        summary_text=summary_text,
        decided_at=timestamp,
    )
    await repository.save_ensemble_promotion_decision(decision)
    return decision
