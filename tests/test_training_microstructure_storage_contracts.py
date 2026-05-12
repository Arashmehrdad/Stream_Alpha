"""Tests for DU7 microstructure storage contracts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from app.training.microstructure_storage_contracts import (
    write_microstructure_storage_contracts,
)


def test_storage_contracts_write_artifacts(tmp_path: Path) -> None:
    """The DU7 storage writer should persist the full artifact contract."""
    output_dir = tmp_path / "storage"

    result = write_microstructure_storage_contracts(repo_root=tmp_path, output_dir=output_dir)

    assert result["storage_contract_status"] == "RESEARCH_STORAGE_CONTRACTS_DEFINED_DRY_RUN"
    assert result["ddl_apply_executed"] is False
    for filename in (
        "manifest.json",
        "microstructure_storage_contracts.json",
        "microstructure_storage_contracts.md",
        "table_contracts.csv",
        "ddl_contracts.csv",
        "boundary_audit.csv",
        "blocked_actions.csv",
        "next_actions.csv",
        "recommendation.json",
    ):
        assert (output_dir / filename).exists()


def test_storage_contracts_are_additive_and_dry_run(tmp_path: Path) -> None:
    """Storage contracts should only define additive research tables."""
    output_dir = tmp_path / "storage"

    write_microstructure_storage_contracts(repo_root=tmp_path, output_dir=output_dir)

    tables = _read_csv(output_dir / "table_contracts.csv")
    names = {row["table_name"] for row in tables}
    assert names == {
        "research_raw_order_book",
        "research_order_book_replay",
        "research_microstructure_features",
        "research_capture_health",
    }
    assert {row["mutates_existing_tables"] for row in tables} == {"False"}
    ddl_rows = _read_csv(output_dir / "ddl_contracts.csv")
    assert {row["execution_status"] for row in ddl_rows} == {"NOT_EXECUTED"}


def test_storage_apply_is_blocked(tmp_path: Path) -> None:
    """DDL execution must remain blocked in this dry-run batch."""
    with pytest.raises(ValueError, match="DDL apply is blocked"):
        write_microstructure_storage_contracts(
            repo_root=tmp_path,
            output_dir=tmp_path / "storage",
            apply=True,
        )


def test_storage_contracts_preserve_non_claims(tmp_path: Path) -> None:
    """The artifact should preserve research-only non-claim flags."""
    output_dir = tmp_path / "storage"

    write_microstructure_storage_contracts(repo_root=tmp_path, output_dir=output_dir)

    report = json.loads((output_dir / "microstructure_storage_contracts.json").read_text())
    recommendation = json.loads((output_dir / "recommendation.json").read_text())
    for flag in ("RESEARCH_ONLY", "NO_RUNTIME_EFFECT", "NOT_PROMOTABLE", "NO_PROFIT_CLAIM"):
        assert flag in report["honesty_flags"]
        assert flag in recommendation["honesty_flags"]
    assert recommendation["runtime_ready"] is False
    assert recommendation["promotable"] is False


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
