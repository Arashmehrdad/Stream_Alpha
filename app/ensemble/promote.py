"""Internal Packet 2 profile drafting and activation helpers for M20 ensemble."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from app.common.time import utc_now
from app.ensemble.config import (
    AgreementPolicyConfig,
    EnsembleConfig,
    default_ensemble_config_path,
    load_ensemble_config,
)
from app.ensemble.schemas import (
    EnsembleChallengerRunRecord,
    EnsembleProfileRecord,
    EnsemblePromotionDecisionRecord,
    EnsembleRosterSelection,
)


def build_draft_profile(  # pylint: disable=too-many-arguments
    *,
    selection: EnsembleRosterSelection,
    profile_id: str,
    config: EnsembleConfig | None = None,
    execution_mode_scope: str = "paper",
    symbol_scope: str = "ALL",
    regime_scope: str = "ALL",
    created_at: datetime | None = None,
) -> EnsembleProfileRecord:
    """Build one draft Packet 2 ensemble profile from a selected 3-role roster."""
    resolved_config = config or load_ensemble_config(default_ensemble_config_path())
    return EnsembleProfileRecord(
        profile_id=profile_id,
        status="DRAFT",
        approval_stage="PENDING",
        execution_mode_scope=execution_mode_scope,
        symbol_scope=symbol_scope,
        regime_scope=regime_scope,
        candidate_roster_json=_candidate_roster_json(selection),
        regime_weight_matrix_json=_regime_weight_matrix_json(resolved_config),
        agreement_policy_json=_agreement_policy_json(resolved_config.agreement_policy),
        evidence_summary_json=dict(selection.evidence_summary_json),
        created_at=utc_now() if created_at is None else created_at,
    )


async def save_draft_profile(
    repository: Any,
    draft_profile: EnsembleProfileRecord,
) -> None:
    """Persist one explicit draft ensemble profile without activating it."""
    await repository.save_ensemble_profile(draft_profile)


async def save_challenger_evidence(
    repository: Any,
    *,
    challenger_run_id: str,
    selection: EnsembleRosterSelection,
    report_path: str,
    created_at: datetime | None = None,
) -> EnsembleChallengerRunRecord:
    """Persist one Packet 2 evaluated challenger evidence row."""
    timestamp = utc_now() if created_at is None else created_at
    record = EnsembleChallengerRunRecord(
        challenger_run_id=challenger_run_id,
        status="EVALUATED",
        config_json={
            "packet": "M20_PACKET_2",
            "top_level_model_name": "dynamic_ensemble",
            "report_path": report_path,
        },
        metrics_json={
            "generalist": selection.generalist.model_dump(mode="json"),
            "trend_specialist": selection.trend_specialist.model_dump(mode="json"),
            "range_specialist": selection.range_specialist.model_dump(mode="json"),
        },
        reason_codes=["ENSEMBLE_PACKET2_EVIDENCE_READY"],
        created_at=timestamp,
        updated_at=timestamp,
    )
    await repository.save_ensemble_challenger_run(record)
    return record


async def activate_draft_profile(  # pylint: disable=too-many-arguments
    repository: Any,
    *,
    draft_profile: EnsembleProfileRecord,
    decision_id: str,
    summary_text: str,
    metrics_delta_json: dict[str, Any],
    safety_checks_json: dict[str, Any],
    reason_codes: list[str],
    decided_at: datetime | None = None,
) -> tuple[EnsembleProfileRecord, EnsemblePromotionDecisionRecord]:
    """Explicitly activate one draft Packet 2 profile and persist promotion evidence."""
    timestamp = utc_now() if decided_at is None else decided_at
    incumbent = await repository.load_active_ensemble_profile(
        execution_mode=draft_profile.execution_mode_scope,
        symbol=draft_profile.symbol_scope,
        regime_label=draft_profile.regime_scope,
    )
    if incumbent is not None and incumbent.profile_id != draft_profile.profile_id:
        superseded = incumbent.model_copy(
            update={
                "status": "SUPERSEDED",
                "approval_stage": "SUPERSEDED",
                "superseded_at": timestamp,
            }
        )
        await repository.save_ensemble_profile(superseded)

    active_profile = draft_profile.model_copy(
        update={
            "status": "ACTIVE",
            "approval_stage": "ACTIVATED",
            "rollback_target_profile_id": (
                draft_profile.rollback_target_profile_id
                if draft_profile.rollback_target_profile_id is not None
                else None if incumbent is None else incumbent.profile_id
            ),
            "approved_at": timestamp,
            "activated_at": timestamp,
            "evidence_summary_json": {
                **dict(draft_profile.evidence_summary_json),
                "top_level_model_identity": {
                    "model_name": "dynamic_ensemble",
                    "model_version": f"ensemble_profile:{draft_profile.profile_id}",
                },
            },
        }
    )
    await repository.save_ensemble_profile(active_profile)
    decision = EnsemblePromotionDecisionRecord(
        decision_id=decision_id,
        target_type="PROFILE",
        target_id=active_profile.profile_id,
        incumbent_id=None if incumbent is None else incumbent.profile_id,
        decision="PROMOTE",
        metrics_delta_json=metrics_delta_json,
        safety_checks_json=safety_checks_json,
        reason_codes=list(reason_codes),
        summary_text=summary_text,
        decided_at=timestamp,
    )
    await repository.save_ensemble_promotion_decision(decision)
    return active_profile, decision


def _candidate_roster_json(selection: EnsembleRosterSelection) -> list[dict[str, Any]]:
    return [
        _roster_entry(
            selection.generalist,
            scope_regimes=("TREND_UP", "TREND_DOWN", "RANGE", "HIGH_VOL"),
        ),
        _roster_entry(
            selection.trend_specialist,
            scope_regimes=("TREND_UP", "TREND_DOWN"),
        ),
        _roster_entry(selection.range_specialist, scope_regimes=("RANGE",)),
    ]


def _roster_entry(
    result: Any,
    *,
    scope_regimes: tuple[str, ...],
) -> dict[str, Any]:
    role = str(result.candidate.candidate_role)
    return {
        "candidate_id": f"{role.lower()}:{result.candidate.model_version}",
        "candidate_role": role,
        "model_version": str(result.candidate.model_version),
        "scope_regimes": list(scope_regimes),
        "enabled": True,
        "expected_model_name": str(result.candidate.model_name),
        "notes": f"family={result.candidate.model_family}",
    }


def _regime_weight_matrix_json(config: EnsembleConfig) -> dict[str, dict[str, float]]:
    return {
        regime_label: {
            candidate_role: float(weight)
            for candidate_role, weight in weights.items()
        }
        for regime_label, weights in config.regime_weight_matrix.items()
    }

def _agreement_policy_json(policy: AgreementPolicyConfig) -> dict[str, float]:
    return {
        "high_ratio_min": float(policy.high_ratio_min),
        "high_spread_max": float(policy.high_spread_max),
        "medium_ratio_min": float(policy.medium_ratio_min),
        "medium_spread_max": float(policy.medium_spread_max),
        "high_multiplier": float(policy.high_multiplier),
        "medium_multiplier": float(policy.medium_multiplier),
        "low_multiplier": float(policy.low_multiplier),
    }
