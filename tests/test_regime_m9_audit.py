"""Tests for the M9 regime integration audit."""

from __future__ import annotations

import json
from pathlib import Path

from app.regime.m9_audit import audit_m9_regime_integration


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    _write(path, json.dumps(payload))


def _complete_repo(root: Path) -> None:
    _write_json(
        root / "artifacts/regime/m8/20260320T165813Z/thresholds.json",
        {"schema_version": "m8_thresholds_v1"},
    )
    _write_json(
        root / "configs/regime_signal_policy.json",
        {"schema_version": "m9_regime_signal_policy_v1"},
    )
    _write(
        root / "app/regime/context.py",
        'REGIME_CONTEXT_SCHEMA_VERSION = "m9_regime_context_v1"',
    )
    _write(root / "app/regime/live.py", "resolve_feature_row_regime")
    _write(root / "app/inference/main.py", '@app.get("/regime"')
    _write(
        root / "app/inference/schemas.py",
        "regime_label: str | None\nregime_freshness_status: str",
    )
    _write(root / "app/trading/decision_trace.py", "regime_reason")
    _write(root / "app/trading/schemas.py", "regime_label: str | None")
    _write(root / "README.md", "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY")
    _write(root / "docs/training.md", "")
    _write(root / "PLANS.md", "")


def test_audit_classifies_complete_surface_set_as_consolidated(tmp_path: Path) -> None:
    """A repo with all critical surfaces should be consolidated."""
    _complete_repo(tmp_path)

    result = audit_m9_regime_integration(
        repo_root=tmp_path,
        output_dir=tmp_path / "audit",
    )

    assert result["m9_state"] == "M9_REGIME_INTEGRATION_CONSOLIDATED"
    assert result["gap_count"] == 0
    assert result["recommendation"] == "PROCEED_TO_M10_RISK_INTERFACE_AUDIT"
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]


def test_audit_marks_missing_critical_surface_as_partial(tmp_path: Path) -> None:
    """Missing critical surfaces should produce operator-readable gaps."""
    _complete_repo(tmp_path)
    (tmp_path / "configs/regime_signal_policy.json").unlink()

    result = audit_m9_regime_integration(
        repo_root=tmp_path,
        output_dir=tmp_path / "audit",
    )

    gap_names = {row["gap_name"] for row in result["gap_analysis"]}
    assert result["m9_state"] == "M9_REGIME_INTEGRATION_PARTIAL"
    assert "m9_signal_policy" in gap_names
    assert result["recommendation"] == "FILL_MISSING_M9_REGIME_GAPS"


def test_audit_writes_contract_and_recommendation_artifacts(tmp_path: Path) -> None:
    """The audit should write the complete artifact contract."""
    _complete_repo(tmp_path)

    result = audit_m9_regime_integration(
        repo_root=tmp_path,
        output_dir=tmp_path / "audit",
    )

    output_dir = Path(result["output_files"]["manifest_json"]).parent
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "m9_regime_integration_audit.json").exists()
    assert (output_dir / "regime_context_contract.json").exists()
    assert (output_dir / "surface_audit.csv").exists()
    assert (output_dir / "recommendation.json").exists()


def test_audit_preserves_m20_pause_and_no_promotion_claims(tmp_path: Path) -> None:
    """The audit must not reopen M20 or create promotion/profit claims."""
    _complete_repo(tmp_path)

    result = audit_m9_regime_integration(
        repo_root=tmp_path,
        output_dir=tmp_path / "audit",
    )

    assert "M20_PAUSED" in result["honesty_flags"]
    assert result["runtime_status"] == "NO_RUNTIME_EFFECT"
    assert result["promotion_status"] == "NOT_PROMOTABLE"
    assert result["profitability_status"] == "NO_PROFIT_CLAIM"
    assert result["recommendation_payload"]["promotable"] is False
