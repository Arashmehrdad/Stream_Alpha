"""Focused tests for research-only M20 trading-aware label artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.research_labels import (
    build_fee_exceedance_labels,
    build_incumbent_meta_labels,
    build_return_fee_exceedance_labels,
    build_return_proxy_triple_barrier_labels,
    build_triple_barrier_labels,
    generate_completed_run_research_labels,
    generate_training_frame_research_labels,
)

# pylint: disable=missing-function-docstring,too-many-arguments


def _write_run(run_dir: Path, rows: list[dict[str, object]]) -> None:
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
    fieldnames: list[str] = []
    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(column)
    with (run_dir / "oof_predictions.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(column)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_training_frame(run_dir: Path, rows: list[dict[str, object]]) -> None:
    frame_dir = run_dir / "training_frame"
    frame_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(frame_dir / "m20_training_frame_features.csv", rows)
    _write_csv(
        frame_dir / "m20_training_frame_keys.csv",
        [
            {
                "symbol": row["symbol"],
                "interval_begin": row["interval_begin"],
                "fold_index": row["fold_index"],
                "row_id": row["row_id"],
            }
            for row in rows
        ],
    )
    (run_dir / "run_config.json").write_text(
        json.dumps({"round_trip_fee_bps": 20}),
        encoding="utf-8",
    )


def _oof_row(
    row_id: str,
    *,
    future_return: float,
    prob_up: float = 0.60,
    symbol: str = "BTC/USD",
    fold_index: int = 0,
    regime_label: str = "RANGE",
) -> dict[str, object]:
    return {
        "model_name": "winner",
        "fold_index": fold_index,
        "row_id": row_id,
        "symbol": symbol,
        "interval_begin": f"2026-04-01T00:0{fold_index}:00Z",
        "as_of_time": f"2026-04-01T00:0{fold_index}:05Z",
        "y_true": int(future_return > 0.0),
        "prob_up": prob_up,
        "confidence": prob_up,
        "regime_label": regime_label,
        "future_return_3": future_return,
        "long_only_gross_value_proxy": max(future_return, 0.0),
        "long_only_net_value_proxy": max(future_return - 0.002, 0.0),
    }


def test_triple_barrier_label_generation_on_price_path() -> None:
    rows = [
        {"row_id": "a", "close_price": 100.0, "high_price": 100.0, "low_price": 100.0},
        {"row_id": "b", "close_price": 100.0, "high_price": 101.0, "low_price": 99.8},
        {"row_id": "c", "close_price": 100.0, "high_price": 100.1, "low_price": 99.0},
    ]

    labels = build_triple_barrier_labels(rows, horizon=1, fixed_barrier_bps=50.0)

    assert labels[0]["label"] == 1
    assert labels[0]["barrier_hit"] == "upper"
    assert labels[1]["label"] == -1
    assert labels[1]["barrier_hit"] == "lower"


def test_return_proxy_triple_barrier_vertical_behavior() -> None:
    labels = build_return_proxy_triple_barrier_labels(
        [
            {"row_id": "a", "future_return_3": 0.0001},
            {"row_id": "b", "future_return_3": 0.0030},
            {"row_id": "c", "future_return_3": -0.0030},
        ],
        horizon=3,
        fixed_barrier_bps=20.0,
    )

    assert [row["label"] for row in labels] == [0, 1, -1]
    assert labels[0]["barrier_hit"] == "vertical"


def test_fee_exceedance_label_generation() -> None:
    price_labels = build_fee_exceedance_labels(
        [
            {"row_id": "a", "close_price": 100.0, "high_price": 100.0},
            {"row_id": "b", "close_price": 100.0, "high_price": 100.5},
            {"row_id": "c", "close_price": 100.0, "high_price": 100.1},
        ],
        horizon=1,
        fee_rate=0.003,
    )
    return_labels = build_return_fee_exceedance_labels(
        [
            {"row_id": "a", "future_return_3": 0.004},
            {"row_id": "b", "future_return_3": 0.001},
        ],
        horizon=3,
        fee_rate=0.003,
    )

    assert price_labels[0]["label"] == 1
    assert return_labels[0]["label"] == 1
    assert return_labels[1]["label"] == 0


def test_meta_label_generation_when_signal_exists() -> None:
    labels = build_incumbent_meta_labels(
        [
            {"row_id": "a", "prob_up": 0.60, "long_only_net_value_proxy": 0.01},
            {"row_id": "b", "prob_up": 0.40, "long_only_net_value_proxy": 0.01},
        ],
    )

    assert [row["meta_label"] for row in labels] == [1, 0]


def test_research_label_manifest_and_diagnostics_are_deterministic(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [
            _oof_row("a", future_return=0.004, symbol="BTC/USD", fold_index=0),
            _oof_row("b", future_return=-0.004, symbol="ETH/USD", fold_index=0),
            _oof_row("c", future_return=0.0001, symbol="BTC/USD", fold_index=1),
        ],
    )

    result = generate_completed_run_research_labels(run_dir=run_dir)

    label_dir = Path(result["label_dir"])
    assert (label_dir / "research_labels_manifest.json").exists()
    assert (label_dir / "triple_barrier_labels.csv").exists()
    assert (label_dir / "fee_exceedance_labels.csv").exists()
    assert (label_dir / "incumbent_meta_labels.csv").exists()
    assert (label_dir / "label_diagnostics.json").exists()
    assert (label_dir / "label_diagnostics.md").exists()
    assert (label_dir / "label_distribution_by_slice.csv").exists()
    assert "MISSING_VOLATILITY_COLUMN_USING_FIXED_BPS" in result["honesty_flags"]
    assert "LABELS_NOT_TRAINING_READY" in result["honesty_flags"]
    assert result["diagnostics"]["triple_barrier"]["positive_event_rate"] > 0.0


def test_missing_incumbent_signal_warns_without_cryptic_failure(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [
            {
                "model_name": "winner",
                "fold_index": 0,
                "row_id": "a",
                "future_return_3": 0.004,
            }
        ],
    )

    result = generate_completed_run_research_labels(run_dir=run_dir)

    assert "MISSING_INCUMBENT_SIGNAL" in result["honesty_flags"]
    assert "incumbent_meta_labels_csv" not in result["output_files"]


def test_slice_diagnostics_include_symbol_fold_and_regime(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [
            _oof_row("a", future_return=0.004, symbol="BTC/USD", fold_index=0),
            _oof_row(
                "b",
                future_return=-0.004,
                symbol="ETH/USD",
                fold_index=1,
                regime_label="TREND_UP",
            ),
        ],
    )

    result = generate_completed_run_research_labels(run_dir=run_dir)
    slice_rows = result["diagnostics"]["label_distribution_by_slice"]
    slice_columns = {row["slice_column"] for row in slice_rows}

    assert {"symbol", "fold_index", "regime_label"}.issubset(slice_columns)


def test_missing_return_and_price_columns_flag_not_training_ready(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    _write_run(
        run_dir,
        [{"model_name": "winner", "fold_index": 0, "row_id": "a", "prob_up": 0.60}],
    )

    result = generate_completed_run_research_labels(run_dir=run_dir)

    assert "MISSING_PRICE_OR_RETURN_COLUMN" in result["honesty_flags"]
    assert "LOW_TOTAL_EVENT_COUNT" in result["honesty_flags"]


def test_training_frame_vol_scaled_labels_use_row_aligned_volatility(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    rows = []
    for symbol in ("BTC/USD", "ETH/USD"):
        prices = [100.0, 101.0, 99.0, 104.0, 104.5]
        for index, price in enumerate(prices):
            rows.append(
                {
                    "symbol": symbol,
                    "interval_begin": f"2026-04-01T00:0{index}:00Z",
                    "fold_index": 4,
                    "row_id": f"{symbol}|{index}",
                    "close_price": price,
                    "high_price": price,
                    "low_price": price,
                    "realized_vol_12": 0.02,
                    "volume": 10 + index,
                }
            )
    _write_training_frame(run_dir, rows)

    result = generate_training_frame_research_labels(run_dir=run_dir, horizon=2)

    label_dir = Path(result["label_dir"])
    assert result["source"] == "training_frame"
    assert result["volatility_column"] == "realized_vol_12"
    assert "ROW_ALIGNED_REALIZED_VOLATILITY_USED" in result["honesty_flags"]
    assert "META_LABEL_NOT_APPLICABLE_NO_OOF_SIGNALS" in result["honesty_flags"]
    assert (label_dir / "triple_barrier_labels_vol_scaled.csv").exists()
    assert (label_dir / "fee_exceedance_labels_vol_scaled.csv").exists()
    assert result["diagnostics"]["skipped_due_to_insufficient_future_horizon"] == 4
    assert {row["slice_column"] for row in result["diagnostics"]["label_distribution_by_slice"]}
    assert (run_dir / "research_labels" / "fixed_bps_comparison").exists()


def test_training_frame_fixed_bps_comparison_can_be_disabled(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    rows = [
        {
            "symbol": "BTC/USD",
            "interval_begin": f"2026-04-01T00:0{index}:00Z",
            "fold_index": 4,
            "row_id": f"BTC/USD|{index}",
            "close_price": 100.0 + index,
            "realized_vol_12": 0.01,
        }
        for index in range(5)
    ]
    _write_training_frame(run_dir, rows)

    generate_training_frame_research_labels(
        run_dir=run_dir,
        horizon=1,
        write_fixed_bps_comparison=False,
    )

    assert not (run_dir / "research_labels" / "fixed_bps_comparison").exists()
