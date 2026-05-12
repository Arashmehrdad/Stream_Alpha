"""Tests for research-only microstructure schema and replay contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.market_microstructure_schema_contracts import (
    write_market_microstructure_schema_contracts,
)


def test_schema_contracts_write_expected_artifacts(tmp_path: Path) -> None:
    """The DU2 contract writer should persist every required artifact."""
    output_dir = tmp_path / "contracts"

    result = write_market_microstructure_schema_contracts(
        repo_root=tmp_path,
        output_dir=output_dir,
    )

    assert result["contract_status"] == "RESEARCH_ONLY_SCHEMA_AND_REPLAY_CONTRACTS_DEFINED"
    for filename in (
        "manifest.json",
        "microstructure_schema_contracts.json",
        "microstructure_schema_contracts.md",
        "table_contracts.csv",
        "column_contracts.csv",
        "ddl_contracts.csv",
        "replay_ordering_contract.csv",
        "feature_contracts.csv",
        "leakage_boundary_audit.csv",
        "implementation_batches.csv",
        "blocked_decisions.csv",
        "next_actions.csv",
        "recommendation.json",
    ):
        assert (output_dir / filename).exists()


def test_contracts_define_research_tables_without_mutating_existing_tables(
    tmp_path: Path,
) -> None:
    """Research microstructure tables should be additive contract rows only."""
    output_dir = tmp_path / "contracts"

    write_market_microstructure_schema_contracts(repo_root=tmp_path, output_dir=output_dir)

    rows = _read_csv(output_dir / "table_contracts.csv")
    table_names = {row["table_name"] for row in rows}
    assert table_names == {
        "research_raw_order_book",
        "research_order_book_replay",
        "research_microstructure_features",
    }
    assert {row["mutates_existing_tables"] for row in rows} == {"False"}
    assert {row["runtime_effect"] for row in rows} == {"NO_RUNTIME_EFFECT"}


def test_contracts_include_deterministic_replay_ordering(tmp_path: Path) -> None:
    """Replay contract should specify stable ordering and explicit gap behavior."""
    output_dir = tmp_path / "contracts"

    write_market_microstructure_schema_contracts(repo_root=tmp_path, output_dir=output_dir)

    rows = _read_csv(output_dir / "replay_ordering_contract.csv")
    ordering = next(row for row in rows if row["contract"] == "ordering")
    assert "event_time" in ordering["rule"]
    assert "received_at" in ordering["rule"]
    assert "sequence_or_checksum" in ordering["required_fields"]
    assert "book_gap_flag" in ordering["gap_behavior"]


def test_contracts_preserve_research_only_non_claims(tmp_path: Path) -> None:
    """Outputs should keep M20 paused and avoid runtime/promotion/profit claims."""
    output_dir = tmp_path / "contracts"

    write_market_microstructure_schema_contracts(repo_root=tmp_path, output_dir=output_dir)

    report = json.loads((output_dir / "microstructure_schema_contracts.json").read_text())
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    for flag in (
        "RESEARCH_ONLY",
        "NO_RUNTIME_EFFECT",
        "NOT_BACKTEST",
        "NOT_RUNTIME_READY",
        "NOT_PROMOTABLE",
        "NO_PROFIT_CLAIM",
    ):
        assert flag in report["honesty_flags"]
        assert flag in recommendation["honesty_flags"]
    assert report["m20_research_decision"] == "M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY"
    assert recommendation["runtime_ready"] is False
    assert recommendation["promotable"] is False
    assert recommendation["profitability_claim"] is False


def test_contracts_block_capture_and_normalizers_until_future_batches(
    tmp_path: Path,
) -> None:
    """DU2 should leave capture and normalizer implementation blocked."""
    output_dir = tmp_path / "contracts"

    write_market_microstructure_schema_contracts(repo_root=tmp_path, output_dir=output_dir)

    blocked = _read_csv(output_dir / "blocked_decisions.csv")
    decisions = {row["decision"]: row for row in blocked}
    assert decisions["sample_payload_fixtures"]["required_action"] == (
        "ADD_STATIC_BOOK_PAYLOAD_FIXTURES"
    )
    assert decisions["capture_service"]["required_action"] == (
        "KEEP_CAPTURE_BLOCKED_UNTIL_DU6_APPROVAL"
    )


def test_contracts_recommend_sample_fixture_normalizers_next(tmp_path: Path) -> None:
    """After DU2, the next safe batch should be fixture-backed normalizers."""
    output_dir = tmp_path / "contracts"

    result = write_market_microstructure_schema_contracts(
        repo_root=tmp_path,
        output_dir=output_dir,
    )

    assert result["recommendation"] == "ADD_SAMPLE_KRAKEN_BOOK_FIXTURE_NORMALIZERS"
    assert (
        result["next_required_action"]
        == "BUILD_SAMPLE_BOOK_PAYLOAD_NORMALIZERS_CONTRACT_ONLY"
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
