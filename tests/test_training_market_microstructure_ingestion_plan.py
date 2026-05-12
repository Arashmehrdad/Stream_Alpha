"""Tests for the market microstructure ingestion planning artifact."""

from __future__ import annotations

import json
from pathlib import Path

from app.training.market_microstructure_ingestion_plan import (
    write_market_microstructure_ingestion_plan,
)


def test_plan_detects_existing_kraken_ingestion_surfaces(tmp_path: Path) -> None:
    """Complete ingestion fixtures should allow the schema-contract recommendation."""
    _write_complete_fixture(tmp_path)

    result = write_market_microstructure_ingestion_plan(repo_root=tmp_path)

    assert result["present_ingestion_surface_count"] == result["existing_ingestion_surface_count"]
    assert result["proposed_source"] == "kraken_public_websocket_v2"
    assert result["recommendation"] == "IMPLEMENT_RESEARCH_ONLY_MICROSTRUCTURE_SCHEMA_CONTRACTS"
    assert (
        result["next_required_action"]
        == "BUILD_RESEARCH_ONLY_MICROSTRUCTURE_SCHEMA_AND_REPLAY_CONTRACTS"
    )


def test_plan_writes_expected_artifacts(tmp_path: Path) -> None:
    """The planner should persist every required output file."""
    _write_complete_fixture(tmp_path)

    result = write_market_microstructure_ingestion_plan(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent

    expected_files = {
        "manifest.json",
        "market_microstructure_ingestion_plan.json",
        "market_microstructure_ingestion_plan.md",
        "existing_ingestion_audit.csv",
        "proposed_source_contract.csv",
        "proposed_storage_contract.csv",
        "replay_contract.csv",
        "feature_derivation_plan.csv",
        "leakage_boundary_audit.csv",
        "implementation_batches.csv",
        "blocked_decisions.csv",
        "next_actions.csv",
        "recommendation.json",
    }
    assert expected_files == {path.name for path in output_dir.iterdir()}


def test_plan_emits_microstructure_contracts(tmp_path: Path) -> None:
    """The plan should cover book, spread, liquidity, and flow inputs."""
    _write_complete_fixture(tmp_path)

    result = write_market_microstructure_ingestion_plan(repo_root=tmp_path)

    source_families = {row["data_family"] for row in result["proposed_source_contract"]}
    feature_names = {row["feature_name"] for row in result["feature_derivation_plan"]}
    assert {"order_book_depth", "top_of_book_spread", "depth_liquidity"} <= source_families
    assert "trade_flow_imbalance" in source_families
    assert "relative_spread" in feature_names
    assert "order_book_imbalance" in feature_names


def test_plan_preserves_research_only_non_claims(tmp_path: Path) -> None:
    """The plan should not claim runtime, promotion, or profit readiness."""
    _write_complete_fixture(tmp_path)

    result = write_market_microstructure_ingestion_plan(repo_root=tmp_path)

    assert "RESEARCH_ONLY" in result["honesty_flags"]
    assert "NO_RUNTIME_EFFECT" in result["honesty_flags"]
    assert "NOT_BACKTEST" in result["honesty_flags"]
    assert "NOT_RUNTIME_READY" in result["honesty_flags"]
    assert "NOT_PROMOTABLE" in result["honesty_flags"]
    assert "NO_PROFIT_CLAIM" in result["honesty_flags"]
    assert {row["runtime_effect"] for row in result["implementation_batches"]} == {
        "NO_RUNTIME_EFFECT"
    }


def test_missing_ingestion_surface_blocks_contract_recommendation(tmp_path: Path) -> None:
    """Missing current ingestion boundaries should block the next implementation step."""
    _write_complete_fixture(tmp_path)
    (tmp_path / "app/ingestion/service.py").write_text("missing\n", encoding="utf-8")

    result = write_market_microstructure_ingestion_plan(repo_root=tmp_path)

    assert result["recommendation"] == "BLOCKED_SOURCE_CONTRACT_DECISION_REQUIRED"
    missing = {
        row["surface"]
        for row in result["existing_ingestion_audit"]
        if row["status"] == "MISSING"
    }
    assert "websocket_reconnect_loop" in missing


def test_recommendation_json_blocks_runtime_readiness(tmp_path: Path) -> None:
    """Recommendation payload should not claim runtime or promotion readiness."""
    _write_complete_fixture(tmp_path)

    result = write_market_microstructure_ingestion_plan(repo_root=tmp_path)
    output_dir = Path(result["output_files"]["manifest_json"]).parent
    recommendation = json.loads((output_dir / "recommendation.json").read_text())

    assert recommendation["runtime_ready"] is False
    assert recommendation["promotable"] is False
    assert recommendation["profitability_claim"] is False


def _write_complete_fixture(root: Path) -> None:
    _write(root / "app/ingestion/kraken.py", "trade\nohlc\n")
    _write(root / "app/ingestion/service.py", _service_fixture())
    _write(root / "app/ingestion/normalizers.py", "TradeEvent\nOhlcEvent\n")
    _write(root / "app/ingestion/db.py", "raw_trades_table\nraw_ohlc_table\n")
    _write(root / "app/ingestion/import_kraken_ohlcvt.py", "feature_replay\n")
    _write(root / "app/training/data_readiness.py", "raw_ohlc\n")


def _service_fixture() -> str:
    return "_backoff\nraw_trades\nraw_ohlc\n"


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
