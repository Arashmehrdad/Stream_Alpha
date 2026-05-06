"""Focused tests for M20 volatility source audit and vol-scaled labels."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.volatility_audit import audit_completed_run_volatility_sources

# pylint: disable=missing-function-docstring


def _write_run(
    run_dir: Path,
    rows: list[dict[str, object]],
    *,
    feature_columns: list[str] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "economics_contract": {"fee_rate": 0.002},
                "winner": {"model_name": "winner"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "feature_columns.json").write_text(
        json.dumps({"expanded_feature_names": feature_columns or []}),
        encoding="utf-8",
    )
    fieldnames: list[str] = []
    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(column)
    with (run_dir / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    label_dir = run_dir / "research_labels"
    label_dir.mkdir(parents=True, exist_ok=True)
    with (label_dir / "triple_barrier_labels.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["row_id", "label", "symbol", "fold_index", "regime_label"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "row_id": row["row_id"],
                    "label": 1 if float(row.get("future_return_3", 0.0)) > 0.002 else 0,
                    "symbol": row.get("symbol", "BTC/USD"),
                    "fold_index": row.get("fold_index", 0),
                    "regime_label": row.get("regime_label", "RANGE"),
                }
            )


def _row(
    row_id: str,
    *,
    future_return: float,
    symbol: str = "BTC/USD",
    index: int = 0,
    volatility: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "model_name": "winner",
        "fold_index": 0,
        "row_id": row_id,
        "symbol": symbol,
        "interval_begin": f"2026-04-01T00:{index:02d}:00Z",
        "regime_label": "RANGE",
        "future_return_3": future_return,
        "long_only_gross_value_proxy": max(future_return, 0.0),
        "long_only_net_value_proxy": max(future_return - 0.002, 0.0),
    }
    if volatility is not None:
        row["realized_vol_12"] = volatility
    return row


def test_detects_existing_volatility_column_and_generates_labels(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = [
        _row("a", future_return=0.004, volatility=0.002, index=0),
        _row("b", future_return=-0.004, volatility=0.002, index=1),
        _row("c", future_return=0.0001, volatility=0.002, index=2),
    ]
    _write_run(run_dir, rows, feature_columns=["realized_vol_12"])

    audit = audit_completed_run_volatility_sources(run_dir=run_dir, lookback=2)

    assert audit["selected_volatility_column"] == "realized_vol_12"
    assert "EXISTING_VOLATILITY_COLUMN_FOUND" in audit["honesty_flags"]
    assert "VOLATILITY_SCALED_LABELS_GENERATED" in audit["honesty_flags"]
    assert Path(
        audit["vol_scaled_label_result"]["output_files"]["triple_barrier_labels_vol_scaled_csv"]
    ).exists()


def test_selects_preferred_manifest_column_deterministically(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = [_row(str(index), future_return=0.001 * index, index=index) for index in range(15)]
    _write_run(
        run_dir,
        rows,
        feature_columns=["volume_zscore_12", "return_std_12", "realized_vol_12"],
    )

    audit = audit_completed_run_volatility_sources(run_dir=run_dir, lookback=3)
    candidate_names = [row["column_name"] for row in audit["candidate_columns"]]

    assert "realized_vol_12" in candidate_names
    assert audit["selected_volatility_column"] == "research_volatility_proxy"
    assert "VOLATILITY_SOURCE_NOT_ALIGNED" in audit["honesty_flags"]
    assert "RESEARCH_COMPUTED_VOLATILITY_PROXY" in audit["honesty_flags"]


def test_computes_rolling_volatility_proxy_without_lookahead_and_by_symbol(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = [
        _row("a0", future_return=0.001, symbol="A", index=0),
        _row("a1", future_return=0.002, symbol="A", index=1),
        _row("a2", future_return=0.003, symbol="A", index=2),
        _row("b0", future_return=0.100, symbol="B", index=0),
        _row("b1", future_return=0.100, symbol="B", index=1),
        _row("b2", future_return=0.100, symbol="B", index=2),
    ]
    _write_run(run_dir, rows)

    audit = audit_completed_run_volatility_sources(run_dir=run_dir, lookback=2)

    summary = {
        (row["slice_column"], row["slice_value"]): row
        for row in _read_csv(Path(audit["output_files"]["volatility_summary_by_slice_csv"]))
    }
    assert float(summary[("symbol", "A")]["finite_volatility_count"]) == 1
    assert float(summary[("symbol", "B")]["finite_volatility_count"]) == 1
    assert "VOLATILITY_HAS_NON_POSITIVE_VALUES" in audit["honesty_flags"]


def test_missing_price_or_return_data_emits_missing_source_warning(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = [{"model_name": "winner", "row_id": "a", "fold_index": 0, "symbol": "A"}]
    _write_run(run_dir, rows)

    audit = audit_completed_run_volatility_sources(run_dir=run_dir)

    assert "VOLATILITY_SOURCE_MISSING" in audit["honesty_flags"]
    assert "VOLATILITY_SCALED_LABELS_NOT_READY" in audit["honesty_flags"]


def test_comparison_diagnostics_are_emitted(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = [
        _row("a", future_return=0.004, volatility=0.002, index=0),
        _row("b", future_return=-0.004, volatility=0.002, index=1),
        _row("c", future_return=0.0001, volatility=0.002, index=2),
    ]
    _write_run(run_dir, rows)

    audit = audit_completed_run_volatility_sources(run_dir=run_dir)
    comparison = audit["vol_scaled_label_result"]["diagnostics"]["fixed_bps_comparison"]

    assert comparison["paired_row_count"] == 3
    assert 0.0 <= comparison["label_agreement_rate"] <= 1.0
    assert comparison["event_rate_stability_by_slice"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]
