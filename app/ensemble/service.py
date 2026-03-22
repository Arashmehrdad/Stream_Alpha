"""Dynamic ensemble scoring service for the Stream Alpha M20 foundation."""

# pylint: disable=too-many-arguments,too-many-locals

from __future__ import annotations

from typing import Any

from app.ensemble.config import (
    AgreementPolicyConfig,
    EnsembleConfig,
    default_ensemble_config_path,
    load_ensemble_config,
)
from app.ensemble.schemas import (
    EnsembleContextPayload,
    EnsembleProfileRecord,
    EnsembleResult,
    ParticipatingCandidate,
)


# ---------------------------------------------------------------------------
# Reason code families
# ---------------------------------------------------------------------------
ENSEMBLE_FALLBACK_SINGLE_MODEL = "ENSEMBLE_FALLBACK_SINGLE_MODEL"
ENSEMBLE_FALLBACK_DISABLED = "ENSEMBLE_FALLBACK_DISABLED"
ENSEMBLE_FALLBACK_NO_PROFILE = "ENSEMBLE_FALLBACK_NO_PROFILE"
ENSEMBLE_FALLBACK_INVALID_PROFILE = "ENSEMBLE_FALLBACK_INVALID_PROFILE"
ENSEMBLE_FALLBACK_NO_CANDIDATES = "ENSEMBLE_FALLBACK_NO_CANDIDATES"
ENSEMBLE_FALLBACK_ALL_SCORE_FAILED = "ENSEMBLE_FALLBACK_ALL_SCORE_FAILED"
ENSEMBLE_PROFILE_ACTIVE = "ENSEMBLE_PROFILE_ACTIVE"
ENSEMBLE_WEIGHT_RENORMALIZED = "ENSEMBLE_WEIGHT_RENORMALIZED"
ENSEMBLE_WEIGHT_FROM_MATRIX = "ENSEMBLE_WEIGHT_FROM_MATRIX"


def _fallback(reason: str) -> EnsembleResult:
    return EnsembleResult(
        active=False,
        fallback_reason=reason,
        weighting_reason_codes=(ENSEMBLE_FALLBACK_SINGLE_MODEL, reason),
    )


class EnsembleService:
    """Regime-aware dynamic ensemble scoring with agreement-adjusted confidence."""

    def __init__(
        self,
        *,
        config: EnsembleConfig | None = None,
        profile_loader=None,
    ) -> None:
        self.config = config or load_ensemble_config(default_ensemble_config_path())
        self._profile_loader = profile_loader

    async def startup(self) -> None:
        """Open additive resources."""

    async def shutdown(self) -> None:
        """Close additive resources."""

    async def resolve_ensemble(
        self,
        *,
        regime_label: str,
        candidate_scores: list[dict[str, Any]],
        active_profile: EnsembleProfileRecord | None = None,
    ) -> EnsembleResult:
        """Run the full ensemble scoring pipeline for one prediction cycle.

        Parameters
        ----------
        regime_label:
            The resolved M9 regime label for the current feature row.
        candidate_scores:
            Each entry must contain:
              candidate_id, candidate_role, model_name, model_version,
              scope_regimes (list), prob_up, prob_down, predicted_class
        active_profile:
            The loaded active ensemble profile from the repository, or None
            to trigger fallback.
        """
        if not self.config.enabled:
            return _fallback(ENSEMBLE_FALLBACK_DISABLED)

        if active_profile is None:
            return _fallback(ENSEMBLE_FALLBACK_NO_PROFILE)

        if not active_profile.candidate_roster_json:
            return _fallback(ENSEMBLE_FALLBACK_INVALID_PROFILE)

        if not candidate_scores:
            return _fallback(ENSEMBLE_FALLBACK_NO_CANDIDATES)

        # --- Filter eligible candidates by scope ---
        eligible: list[dict[str, Any]] = []
        all_candidates: list[ParticipatingCandidate] = []
        for cs in candidate_scores:
            scope = cs.get("scope_regimes", [])
            if scope and regime_label not in scope:
                all_candidates.append(
                    ParticipatingCandidate(
                        candidate_id=cs["candidate_id"],
                        candidate_role=cs["candidate_role"],
                        model_name=cs["model_name"],
                        model_version=cs["model_version"],
                        participation_status="EXCLUDED_SCOPE",
                        scope_regimes=list(scope),
                    )
                )
                continue
            eligible.append(cs)

        if not eligible:
            return _fallback(ENSEMBLE_FALLBACK_NO_CANDIDATES)

        # --- Resolve regime weights ---
        use_profile_matrix = bool(active_profile.regime_weight_matrix_json)
        weight_matrix = (
            active_profile.regime_weight_matrix_json
            if use_profile_matrix
            else self.config.regime_weight_matrix
        )
        regime_weights = weight_matrix.get(regime_label, {})

        weighting_codes = [ENSEMBLE_PROFILE_ACTIVE, ENSEMBLE_WEIGHT_FROM_MATRIX]

        # --- Score and weight ---
        scored_candidates: list[ParticipatingCandidate] = []
        weighted_prob_ups: list[float] = []
        weighted_prob_downs: list[float] = []
        total_weight = 0.0

        for cs in eligible:
            role = cs["candidate_role"]
            raw_weight = regime_weights.get(role, 0.0)
            if raw_weight <= 0.0:
                all_candidates.append(
                    ParticipatingCandidate(
                        candidate_id=cs["candidate_id"],
                        candidate_role=role,
                        model_name=cs["model_name"],
                        model_version=cs["model_version"],
                        participation_status="EXCLUDED_SCOPE",
                        scope_regimes=cs.get("scope_regimes", []),
                        applied_weight=0.0,
                        prob_up=cs["prob_up"],
                        prob_down=cs["prob_down"],
                        predicted_class=cs["predicted_class"],
                    )
                )
                continue
            total_weight += raw_weight
            scored_candidates.append(
                ParticipatingCandidate(
                    candidate_id=cs["candidate_id"],
                    candidate_role=role,
                    model_name=cs["model_name"],
                    model_version=cs["model_version"],
                    participation_status="ELIGIBLE",
                    scope_regimes=cs.get("scope_regimes", []),
                    applied_weight=raw_weight,
                    prob_up=cs["prob_up"],
                    prob_down=cs["prob_down"],
                    predicted_class=cs["predicted_class"],
                )
            )

        if not scored_candidates:
            return _fallback(ENSEMBLE_FALLBACK_NO_CANDIDATES)

        # --- Renormalize weights ---
        if abs(total_weight - 1.0) > 1e-9:
            weighting_codes.append(ENSEMBLE_WEIGHT_RENORMALIZED)
            renormalized = []
            for sc in scored_candidates:
                new_weight = sc.applied_weight / total_weight
                renormalized.append(
                    ParticipatingCandidate(
                        candidate_id=sc.candidate_id,
                        candidate_role=sc.candidate_role,
                        model_name=sc.model_name,
                        model_version=sc.model_version,
                        participation_status=sc.participation_status,
                        scope_regimes=sc.scope_regimes,
                        applied_weight=new_weight,
                        prob_up=sc.prob_up,
                        prob_down=sc.prob_down,
                        predicted_class=sc.predicted_class,
                    )
                )
            scored_candidates = renormalized

        # --- Compose ensemble probabilities ---
        ensemble_prob_up = sum(c.applied_weight * c.prob_up for c in scored_candidates)
        ensemble_prob_down = sum(c.applied_weight * c.prob_down for c in scored_candidates)

        ensemble_predicted_class = "UP" if ensemble_prob_up >= ensemble_prob_down else "DOWN"

        # --- Agreement calculation ---
        agreement_policy = _resolve_agreement_policy(
            active_profile.agreement_policy_json,
            self.config.agreement_policy,
        )
        vote_up = sum(1 for c in scored_candidates if c.predicted_class == "UP")
        vote_down = sum(1 for c in scored_candidates if c.predicted_class == "DOWN")
        eligible_count = len(scored_candidates)
        vote_agreement_ratio = max(vote_up, vote_down) / eligible_count
        prob_ups = [c.prob_up for c in scored_candidates]
        probability_spread = max(prob_ups) - min(prob_ups)

        agreement_band, agreement_multiplier = _resolve_agreement_band(
            vote_agreement_ratio=vote_agreement_ratio,
            probability_spread=probability_spread,
            policy=agreement_policy,
        )

        raw_ensemble_confidence = max(ensemble_prob_up, ensemble_prob_down)
        effective_confidence = min(
            max(raw_ensemble_confidence * agreement_multiplier, 0.0),
            1.0,
        )

        final_candidates = list(all_candidates) + list(scored_candidates)

        return EnsembleResult(
            active=True,
            ensemble_profile_id=active_profile.profile_id,
            approval_stage=active_profile.approval_stage,
            ensemble_prob_up=ensemble_prob_up,
            ensemble_prob_down=ensemble_prob_down,
            ensemble_predicted_class=ensemble_predicted_class,
            raw_ensemble_confidence=raw_ensemble_confidence,
            effective_confidence=effective_confidence,
            agreement_band=agreement_band,
            vote_agreement_ratio=vote_agreement_ratio,
            probability_spread=probability_spread,
            agreement_multiplier=agreement_multiplier,
            candidate_count=eligible_count,
            participating_candidates=tuple(final_candidates),
            weighting_reason_codes=tuple(weighting_codes),
        )


def _resolve_agreement_policy(
    profile_policy_json: dict[str, Any],
    default_policy: AgreementPolicyConfig,
) -> AgreementPolicyConfig:
    """Use profile-level agreement policy if present, otherwise config default."""
    if not profile_policy_json:
        return default_policy
    return AgreementPolicyConfig(
        high_ratio_min=float(profile_policy_json.get(
            "high_ratio_min", default_policy.high_ratio_min
        )),
        high_spread_max=float(profile_policy_json.get(
            "high_spread_max", default_policy.high_spread_max
        )),
        medium_ratio_min=float(profile_policy_json.get(
            "medium_ratio_min", default_policy.medium_ratio_min
        )),
        medium_spread_max=float(profile_policy_json.get(
            "medium_spread_max", default_policy.medium_spread_max
        )),
        high_multiplier=float(profile_policy_json.get(
            "high_multiplier", default_policy.high_multiplier
        )),
        medium_multiplier=float(profile_policy_json.get(
            "medium_multiplier", default_policy.medium_multiplier
        )),
        low_multiplier=float(profile_policy_json.get(
            "low_multiplier", default_policy.low_multiplier
        )),
    )


def _resolve_agreement_band(
    *,
    vote_agreement_ratio: float,
    probability_spread: float,
    policy: AgreementPolicyConfig,
) -> tuple[str, float]:
    """Return (agreement_band, agreement_multiplier) per the M20 spec."""
    if (
        vote_agreement_ratio >= policy.high_ratio_min
        and probability_spread <= policy.high_spread_max
    ):
        return "HIGH", policy.high_multiplier
    if (
        vote_agreement_ratio >= policy.medium_ratio_min
        and probability_spread <= policy.medium_spread_max
    ):
        return "MEDIUM", policy.medium_multiplier
    return "LOW", policy.low_multiplier
