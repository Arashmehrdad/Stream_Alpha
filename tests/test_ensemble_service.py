"""Tests for the Stream Alpha M20 dynamic ensemble service."""

# pylint: disable=duplicate-code

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import yaml

from app.ensemble.config import (
    AgreementPolicyConfig,
    EnsembleConfig,
    load_ensemble_config,
)
from app.ensemble.schemas import (
    EnsembleProfileRecord,
    EnsembleResult,
)
from app.ensemble.service import (
    ENSEMBLE_FALLBACK_DISABLED,
    ENSEMBLE_FALLBACK_INVALID_PROFILE,
    ENSEMBLE_FALLBACK_NO_CANDIDATES,
    ENSEMBLE_FALLBACK_NO_PROFILE,
    ENSEMBLE_FALLBACK_SINGLE_MODEL,
    ENSEMBLE_PROFILE_ACTIVE,
    ENSEMBLE_WEIGHT_FROM_MATRIX,
    ENSEMBLE_WEIGHT_RENORMALIZED,
    EnsembleService,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

AGREEMENT_POLICY = AgreementPolicyConfig(
    high_ratio_min=0.80,
    high_spread_max=0.12,
    medium_ratio_min=0.67,
    medium_spread_max=0.20,
    high_multiplier=1.00,
    medium_multiplier=0.93,
    low_multiplier=0.85,
)

CONFIG = EnsembleConfig(
    enabled=True,
    candidate_roles=("GENERALIST", "TREND_SPECIALIST", "RANGE_SPECIALIST"),
    regime_weight_matrix={
        "TREND_UP": {
            "GENERALIST": 0.25,
            "TREND_SPECIALIST": 0.75,
            "RANGE_SPECIALIST": 0.00,
        },
        "TREND_DOWN": {
            "GENERALIST": 0.25,
            "TREND_SPECIALIST": 0.75,
            "RANGE_SPECIALIST": 0.00,
        },
        "RANGE": {
            "GENERALIST": 0.30,
            "TREND_SPECIALIST": 0.10,
            "RANGE_SPECIALIST": 0.60,
        },
        "HIGH_VOL": {
            "GENERALIST": 1.00,
            "TREND_SPECIALIST": 0.00,
            "RANGE_SPECIALIST": 0.00,
        },
    },
    agreement_policy=AGREEMENT_POLICY,
    artifact_root="artifacts/ensemble",
)

DISABLED_CONFIG = EnsembleConfig(
    enabled=False,
    candidate_roles=("GENERALIST",),
    regime_weight_matrix={
        "TREND_UP": {"GENERALIST": 1.00},
    },
    agreement_policy=AGREEMENT_POLICY,
    artifact_root="artifacts/ensemble",
)


def _make_profile(
    *,
    roster: list | None = None,
    matrix: dict | None = None,
    policy: dict | None = None,
) -> EnsembleProfileRecord:
    return EnsembleProfileRecord(
        profile_id="ens-profile-001",
        status="ACTIVE",
        approval_stage="ACTIVATED",
        candidate_roster_json=[{"candidate_id": "c1"}] if roster is None else roster,
        regime_weight_matrix_json=matrix or {},
        agreement_policy_json=policy or {},
    )


def _candidate(
    *,
    candidate_id: str = "c1",
    role: str = "GENERALIST",
    prob_up: float = 0.70,
    prob_down: float = 0.30,
    predicted_class: str = "UP",
    scope_regimes: list | None = None,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "candidate_role": role,
        "model_name": f"model-{candidate_id}",
        "model_version": "v1",
        "prob_up": prob_up,
        "prob_down": prob_down,
        "predicted_class": predicted_class,
        "scope_regimes": scope_regimes or [],
    }


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def test_load_checked_in_config() -> None:
    """The checked-in configs/ensemble.yaml must parse without errors."""
    config = load_ensemble_config(Path("configs/ensemble.yaml"))
    assert config.enabled is True
    assert len(config.candidate_roles) == 3
    assert "TREND_UP" in config.regime_weight_matrix
    assert config.agreement_policy.high_multiplier == 1.00


def test_config_weight_matrix_sums_to_one() -> None:
    """Every regime row in the weight matrix must sum approximately to 1.0."""
    config = load_ensemble_config(Path("configs/ensemble.yaml"))
    for regime, weights in config.regime_weight_matrix.items():
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01, f"{regime} weights sum to {total}"


def test_config_rejects_invalid_weight_sum(tmp_path: Path) -> None:
    """Configs with weights not summing to ~1.0 must raise ValueError."""
    bad = {
        "enabled": True,
        "candidate_roles": ["GENERALIST"],
        "regime_weight_matrix": {
            "TREND_UP": {"GENERALIST": 0.50},
        },
        "agreement_policy": {
            "high_ratio_min": 0.80,
            "high_spread_max": 0.12,
            "medium_ratio_min": 0.67,
            "medium_spread_max": 0.20,
            "high_multiplier": 1.00,
            "medium_multiplier": 0.93,
            "low_multiplier": 0.85,
        },
        "artifact_root": "artifacts/ensemble",
    }
    path = tmp_path / "bad_ensemble.yaml"
    path.write_text(yaml.dump(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="sum to ~1.0"):
        load_ensemble_config(path)


# ---------------------------------------------------------------------------
# Fallback – single-model
# ---------------------------------------------------------------------------


def test_fallback_when_disabled() -> None:
    """Disabled ensemble must return explicit fallback with correct reason code."""
    async def _run():
        svc = EnsembleService(config=DISABLED_CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="TREND_UP",
            candidate_scores=[_candidate()],
            active_profile=_make_profile(),
        )
        assert result.active is False
        assert result.fallback_reason == ENSEMBLE_FALLBACK_DISABLED
        assert ENSEMBLE_FALLBACK_SINGLE_MODEL in result.weighting_reason_codes
    asyncio.run(_run())


def test_fallback_when_no_profile() -> None:
    """Missing active profile must fall back with NO_PROFILE reason."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="TREND_UP",
            candidate_scores=[_candidate()],
            active_profile=None,
        )
        assert result.active is False
        assert result.fallback_reason == ENSEMBLE_FALLBACK_NO_PROFILE
    asyncio.run(_run())


def test_fallback_when_invalid_profile() -> None:
    """Profile with empty roster must fall back with INVALID_PROFILE reason."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="TREND_UP",
            candidate_scores=[_candidate()],
            active_profile=_make_profile(roster=[]),
        )
        assert result.active is False
        assert result.fallback_reason == ENSEMBLE_FALLBACK_INVALID_PROFILE
    asyncio.run(_run())


def test_fallback_when_no_candidate_scores() -> None:
    """Empty candidate_scores must fall back with NO_CANDIDATES reason."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="TREND_UP",
            candidate_scores=[],
            active_profile=_make_profile(),
        )
        assert result.active is False
        assert result.fallback_reason == ENSEMBLE_FALLBACK_NO_CANDIDATES
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Regime-specific weights
# ---------------------------------------------------------------------------


def test_trend_up_weights_favor_trend_specialist() -> None:
    """TREND_UP regime must give 0.75 weight to TREND_SPECIALIST."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="TREND_UP",
            candidate_scores=[
                _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.60),
                _candidate(candidate_id="ts", role="TREND_SPECIALIST", prob_up=0.80),
            ],
            active_profile=_make_profile(),
        )
        assert result.active is True
        expected = 0.25 * 0.60 + 0.75 * 0.80
        assert abs(result.ensemble_prob_up - expected) < 1e-9
    asyncio.run(_run())


def test_high_vol_weights_all_to_generalist() -> None:
    """HIGH_VOL regime must give 1.00 weight to GENERALIST only."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="HIGH_VOL",
            candidate_scores=[
                _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.65),
                _candidate(candidate_id="ts", role="TREND_SPECIALIST", prob_up=0.55),
                _candidate(candidate_id="rs", role="RANGE_SPECIALIST", prob_up=0.50),
            ],
            active_profile=_make_profile(),
        )
        assert result.active is True
        assert abs(result.ensemble_prob_up - 0.65) < 1e-9
        assert result.candidate_count == 1  # only GENERALIST is scored
    asyncio.run(_run())


def test_range_weights() -> None:
    """RANGE regime must give G:0.30, T:0.10, R:0.60."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="RANGE",
            candidate_scores=[
                _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.50),
                _candidate(candidate_id="ts", role="TREND_SPECIALIST", prob_up=0.60),
                _candidate(candidate_id="rs", role="RANGE_SPECIALIST", prob_up=0.70),
            ],
            active_profile=_make_profile(),
        )
        expected = 0.30 * 0.50 + 0.10 * 0.60 + 0.60 * 0.70
        assert abs(result.ensemble_prob_up - expected) < 1e-9
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Candidate filtering by scope
# ---------------------------------------------------------------------------


def test_candidate_excluded_by_scope_regimes() -> None:
    """Candidates whose scope_regimes don't include the current regime are excluded."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="TREND_UP",
            candidate_scores=[
                _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.70),
                _candidate(
                    candidate_id="ts",
                    role="TREND_SPECIALIST",
                    prob_up=0.90,
                    scope_regimes=["RANGE"],  # excluded from TREND_UP
                ),
            ],
            active_profile=_make_profile(),
        )
        assert result.active is True
        # Only GENERALIST participates (weight renormalized to 1.0)
        assert abs(result.ensemble_prob_up - 0.70) < 1e-9
        assert ENSEMBLE_WEIGHT_RENORMALIZED in result.weighting_reason_codes
    asyncio.run(_run())


def test_all_candidates_excluded_falls_back() -> None:
    """If all candidates are excluded by scope, ensemble must fall back."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="TREND_UP",
            candidate_scores=[
                _candidate(
                    candidate_id="rs",
                    role="RANGE_SPECIALIST",
                    prob_up=0.70,
                    scope_regimes=["RANGE"],
                ),
            ],
            active_profile=_make_profile(),
        )
        assert result.active is False
        assert result.fallback_reason == ENSEMBLE_FALLBACK_NO_CANDIDATES
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Weight renormalization
# ---------------------------------------------------------------------------


def test_weight_renormalization_when_candidate_missing() -> None:
    """When only two of three roles present, weights must renormalize to sum to 1.0."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        # TREND_UP: G=0.25, TS=0.75, RS=0.00 — remove TREND_SPECIALIST
        result = await svc.resolve_ensemble(
            regime_label="TREND_UP",
            candidate_scores=[
                _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.60),
            ],
            active_profile=_make_profile(),
        )
        assert result.active is True
        # Renormalized: GENERALIST gets 0.25/0.25 = 1.0
        assert abs(result.ensemble_prob_up - 0.60) < 1e-9
        assert ENSEMBLE_WEIGHT_RENORMALIZED in result.weighting_reason_codes
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Agreement calculation determinism
# ---------------------------------------------------------------------------


def test_agreement_high_band() -> None:
    """When all candidates agree and spread is tight, agreement band must be HIGH."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="RANGE",
            candidate_scores=[
                _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.70),
                _candidate(candidate_id="ts", role="TREND_SPECIALIST", prob_up=0.72),
                _candidate(candidate_id="rs", role="RANGE_SPECIALIST", prob_up=0.71),
            ],
            active_profile=_make_profile(),
        )
        assert result.agreement_band == "HIGH"
        assert result.agreement_multiplier == 1.00
        # All vote UP, ratio = 1.0, spread = 0.02 < 0.12
        assert result.vote_agreement_ratio == 1.0
        assert abs(result.probability_spread - 0.02) < 1e-9
    asyncio.run(_run())


def test_agreement_medium_band() -> None:
    """Agreement is MEDIUM when ratio qualifies but spread fails HIGH criteria."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        # All vote UP (ratio=1.0 >= 0.67) but spread=0.15 > 0.12 → fails HIGH
        # ratio=1.0 >= 0.67, spread=0.15 <= 0.20 → MEDIUM
        result = await svc.resolve_ensemble(
            regime_label="RANGE",
            candidate_scores=[
                _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.70),
                _candidate(candidate_id="ts", role="TREND_SPECIALIST", prob_up=0.55),
                _candidate(candidate_id="rs", role="RANGE_SPECIALIST", prob_up=0.65),
            ],
            active_profile=_make_profile(),
        )
        # spread = 0.70-0.55 = 0.15
        assert result.agreement_band == "MEDIUM"
        assert result.agreement_multiplier == 0.93
    asyncio.run(_run())


def test_agreement_low_band() -> None:
    """Wide spread and low agreement lead to LOW band."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="RANGE",
            candidate_scores=[
                _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.80),
                _candidate(candidate_id="ts", role="TREND_SPECIALIST", prob_up=0.40),
                _candidate(
                    candidate_id="rs",
                    role="RANGE_SPECIALIST",
                    prob_up=0.30,
                    predicted_class="DOWN",
                ),
            ],
            active_profile=_make_profile(),
        )
        # vote_up=1, vote_down=2 → ratio=2/3≈0.667 >= 0.67
        # spread = 0.80-0.30 = 0.50 > 0.20 → LOW
        assert result.agreement_band == "LOW"
        assert result.agreement_multiplier == 0.85
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Effective confidence follows multiplier rules
# ---------------------------------------------------------------------------


def test_effective_confidence_formula() -> None:
    """effective_confidence = clamp(raw * multiplier, 0.0, 1.0)."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="TREND_UP",
            candidate_scores=[
                _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.60),
                _candidate(candidate_id="ts", role="TREND_SPECIALIST", prob_up=0.80),
            ],
            active_profile=_make_profile(),
        )
        raw = result.raw_ensemble_confidence
        mult = result.agreement_multiplier
        expected = min(max(raw * mult, 0.0), 1.0)
        assert abs(result.effective_confidence - expected) < 1e-9
    asyncio.run(_run())


def test_effective_confidence_clamped_to_one() -> None:
    """effective_confidence must never exceed 1.0 even with high raw confidence."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        # If raw confidence is very close to 1.0, clamp ensures <=1.0
        result = await svc.resolve_ensemble(
            regime_label="HIGH_VOL",
            candidate_scores=[
                _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.999),
            ],
            active_profile=_make_profile(),
        )
        assert result.effective_confidence <= 1.0
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Profile-level weight matrix override
# ---------------------------------------------------------------------------


def test_profile_weight_matrix_overrides_config() -> None:
    """If the profile carries its own regime weight matrix, it takes precedence."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        profile_matrix = {
            "TREND_UP": {
                "GENERALIST": 0.50,
                "TREND_SPECIALIST": 0.50,
            },
        }
        result = await svc.resolve_ensemble(
            regime_label="TREND_UP",
            candidate_scores=[
                _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.60),
                _candidate(candidate_id="ts", role="TREND_SPECIALIST", prob_up=0.80),
            ],
            active_profile=_make_profile(matrix=profile_matrix),
        )
        expected = 0.50 * 0.60 + 0.50 * 0.80
        assert abs(result.ensemble_prob_up - expected) < 1e-9
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Profile-level agreement policy override
# ---------------------------------------------------------------------------


def test_profile_agreement_policy_overrides_config() -> None:
    """If the profile carries its own agreement policy, it takes precedence."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="RANGE",
            candidate_scores=[
                _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.70),
                _candidate(candidate_id="ts", role="TREND_SPECIALIST", prob_up=0.72),
                _candidate(candidate_id="rs", role="RANGE_SPECIALIST", prob_up=0.71),
            ],
            active_profile=_make_profile(policy={
                # Extremely strict: ratio_min=0.99 means only perfect agreement is HIGH
                "high_ratio_min": 0.99,
                "high_spread_max": 0.01,
                "medium_ratio_min": 0.90,
                "medium_spread_max": 0.05,
                "high_multiplier": 1.00,
                "medium_multiplier": 0.93,
                "low_multiplier": 0.85,
            }),
        )
        # vote ratio=1.0 >= 0.99, but spread=0.02 > 0.01 → fails HIGH
        # ratio=1.0 >= 0.90, spread=0.02 < 0.05 → MEDIUM
        assert result.agreement_band == "MEDIUM"
        assert result.agreement_multiplier == 0.93
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Context payload conversion
# ---------------------------------------------------------------------------


def test_to_context_payload_on_active_result() -> None:
    """Active result must produce an EnsembleContextPayload with all fields populated."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="TREND_UP",
            candidate_scores=[
                _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.60),
                _candidate(candidate_id="ts", role="TREND_SPECIALIST", prob_up=0.80),
            ],
            active_profile=_make_profile(),
        )
        payload = result.to_context_payload(
            regime_label="TREND_UP",
            regime_run_id="run-001",
        )
        assert payload is not None
        assert payload.ensemble_profile_id == "ens-profile-001"
        assert payload.resolved_regime_label == "TREND_UP"
        assert payload.resolved_regime_run_id == "run-001"
        assert payload.effective_confidence == result.effective_confidence
        assert len(payload.participating_candidates) > 0
    asyncio.run(_run())


def test_to_context_payload_on_inactive_fallback() -> None:
    """Fallback result with a reason must still produce a payload for observability."""
    result = EnsembleResult(
        active=False,
        fallback_reason=ENSEMBLE_FALLBACK_NO_PROFILE,
        weighting_reason_codes=(ENSEMBLE_FALLBACK_SINGLE_MODEL, ENSEMBLE_FALLBACK_NO_PROFILE),
    )
    payload = result.to_context_payload(regime_label="TREND_UP")
    assert payload is not None
    assert payload.ensemble_profile_id is None
    assert ENSEMBLE_FALLBACK_SINGLE_MODEL in payload.weighting_reason_codes


def test_to_context_payload_returns_none_for_default() -> None:
    """Default EnsembleResult (no fallback, not active) produces None."""
    result = EnsembleResult()
    payload = result.to_context_payload()
    assert payload is None


# ---------------------------------------------------------------------------
# Weighting reason codes
# ---------------------------------------------------------------------------


def test_reason_codes_present_on_active_result() -> None:
    """Active results must contain PROFILE_ACTIVE and WEIGHT_FROM_MATRIX codes."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="TREND_UP",
            candidate_scores=[
                _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.60),
                _candidate(candidate_id="ts", role="TREND_SPECIALIST", prob_up=0.80),
            ],
            active_profile=_make_profile(),
        )
        assert ENSEMBLE_PROFILE_ACTIVE in result.weighting_reason_codes
        assert ENSEMBLE_WEIGHT_FROM_MATRIX in result.weighting_reason_codes
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Predicted class aggregation
# ---------------------------------------------------------------------------


def test_ensemble_predicted_class_follows_weighted_majority() -> None:
    """Ensemble predicted_class must follow weighted ensemble probabilities."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        result = await svc.resolve_ensemble(
            regime_label="TREND_UP",
            candidate_scores=[
                _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.30, predicted_class="DOWN"),
                _candidate(candidate_id="ts", role="TREND_SPECIALIST", prob_up=0.20, predicted_class="DOWN"),
            ],
            active_profile=_make_profile(),
        )
        # ensemble_prob_up = 0.25*0.30 + 0.75*0.20 = 0.075 + 0.150 = 0.225
        # ensemble_prob_down = 0.25*0.70 + 0.75*0.80 = 0.175 + 0.600 = 0.775
        assert result.ensemble_predicted_class == "DOWN"
        assert result.ensemble_prob_up < result.ensemble_prob_down
    asyncio.run(_run())


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_resolve_ensemble_is_deterministic() -> None:
    """Given the same inputs, resolve_ensemble must return identical results."""
    async def _run():
        svc = EnsembleService(config=CONFIG)
        candidates = [
            _candidate(candidate_id="gen", role="GENERALIST", prob_up=0.60),
            _candidate(candidate_id="ts", role="TREND_SPECIALIST", prob_up=0.80),
        ]
        profile = _make_profile()
        r1 = await svc.resolve_ensemble(
            regime_label="TREND_UP",
            candidate_scores=candidates,
            active_profile=profile,
        )
        r2 = await svc.resolve_ensemble(
            regime_label="TREND_UP",
            candidate_scores=candidates,
            active_profile=profile,
        )
        assert r1 == r2
    asyncio.run(_run())
