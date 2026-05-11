"""Focused tests for static M20 research dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from app.training.m20_research_dashboard import write_m20_research_dashboard

# pylint: disable=missing-function-docstring


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _write_sources(base: Path, prediction: Path) -> None:
    vol = base / "research_labels" / "vol_scaled"
    _write_json(
        vol / "strategy_candidate_factory" / "recommendation.json",
        {"recommendation": "WATCHLIST_OR_REFINE_STRATEGY_CANDIDATE_DEFINITIONS"},
    )
    _write_json(
        vol / "strategy_candidate_refinement" / "recommendation.json",
        {"recommendation": "REFINE_STRATEGY_CANDIDATE_DEFINITIONS"},
    )
    _write_json(
        vol / "strategy_slice_policy_evaluator" / "recommendation.json",
        {"recommendation": "REFINE_STRATEGY_CANDIDATE_DEFINITIONS"},
    )
    _write_json(
        vol / "strategy_model_factory_plan" / "recommendation.json",
        {
            "recommendation": (
                "DESIGN_REUSABLE_STRATEGY_CONDITIONED_MODEL_FACTORY_CONTRACT"
            )
        },
    )
    _write_json(
        vol / "research_candidate_comparator" / "recommendation.json",
        {
            "recommendation": "PAUSE_CURRENT_M20_CANDIDATE_PATHS",
            "evidence_blockers": ["NO_POSITIVE_PROXY_RESEARCH_CANDIDATE"],
        },
    )
    _write_json(
        prediction
        / "research_labels"
        / "vol_scaled"
        / "cost_aware_policy_adjudication"
        / "recommendation.json",
        {"recommendation": "WATCHLIST_NEURALFORECAST_SPECIALISTS_DO_NOT_PROMOTE"},
    )


def test_dashboard_rolls_up_evidence_and_blockers(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction"
    _write_sources(tmp_path, prediction)

    result = write_m20_research_dashboard(
        source_run_dir=tmp_path,
        prediction_run_dir=prediction,
    )

    assert result["overall_decision"] == "PAUSE_CURRENT_M20_RESEARCH_PATHS"
    assert result["evidence_count"] >= 6
    assert any(
        row["blocker"] == "NO_POSITIVE_PROXY_RESEARCH_CANDIDATE"
        for row in result["open_blockers"]
    )


def test_outputs_include_required_artifacts(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction"
    _write_sources(tmp_path, prediction)

    result = write_m20_research_dashboard(
        source_run_dir=tmp_path,
        prediction_run_dir=prediction,
    )
    files = result["output_files"]

    assert Path(files["manifest_json"]).exists()
    assert Path(files["m20_research_dashboard_json"]).exists()
    assert Path(files["m20_research_dashboard_md"]).exists()
    assert Path(files["evidence_index_csv"]).exists()
    assert Path(files["decision_timeline_csv"]).exists()
    assert Path(files["open_blockers_csv"]).exists()
    assert Path(files["next_actions_csv"]).exists()
    assert Path(files["recommendation_json"]).exists()


def test_dashboard_preserves_research_only_statuses(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction"
    _write_sources(tmp_path, prediction)

    result = write_m20_research_dashboard(
        source_run_dir=tmp_path,
        prediction_run_dir=prediction,
    )
    dashboard = json.loads(
        Path(result["output_files"]["m20_research_dashboard_json"]).read_text(
            encoding="utf-8"
        )
    )

    assert "NO_RUNTIME_EFFECT" in dashboard["honesty_flags"]
    assert "NOT_PROMOTABLE" in dashboard["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in dashboard["honesty_flags"]
    assert dashboard["promotion_status"] == "NOT_PROMOTABLE"


def test_no_runtime_imports_or_registry_writes() -> None:
    source = Path("app/training/m20_research_dashboard.py").read_text(
        encoding="utf-8"
    )

    assert "app.inference" not in source
    assert "app.trading" not in source
    assert "app.training.registry" not in source
