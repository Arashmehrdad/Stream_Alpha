"""Focused tests for M20 research-label readiness diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.training.label_readiness import LabelReadinessThresholds, analyze_label_readiness

# pylint: disable=missing-function-docstring


def _write_label_dir(
    label_dir: Path,
    *,
    triple_rows: list[dict[str, object]],
    fee_rows: list[dict[str, object]],
    meta_rows: list[dict[str, object]] | None = None,
    source_flags: list[str] | None = None,
) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_dir": str(label_dir.parent),
        "label_dir": str(label_dir),
        "honesty_flags": source_flags or [],
        "output_files": {},
    }
    (label_dir / "research_labels_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    _write_csv(label_dir / "triple_barrier_labels.csv", triple_rows)
    _write_csv(label_dir / "fee_exceedance_labels.csv", fee_rows)
    if meta_rows is not None:
        _write_csv(label_dir / "incumbent_meta_labels.csv", meta_rows)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for column in row:
            if column not in fieldnames:
                fieldnames.append(column)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames or ["label"])
        writer.writeheader()
        writer.writerows(rows)


def _label_row(
    label: int,
    *,
    fold_index: int = 0,
    symbol: str = "BTC/USD",
    regime_label: str = "RANGE",
) -> dict[str, object]:
    return {
        "label": label,
        "fold_index": fold_index,
        "symbol": symbol,
        "regime_label": regime_label,
    }


def _meta_row(meta_label: int, *, base_signal: bool) -> dict[str, object]:
    return {
        "meta_label": meta_label,
        "base_signal": str(base_signal),
        "probability": 0.7 if base_signal else 0.3,
    }


def _thresholds() -> LabelReadinessThresholds:
    return LabelReadinessThresholds(
        min_total_rows=6,
        min_positive_class_rate=0.10,
        min_positives_per_fold=2,
        min_positives_per_symbol=2,
        max_neutral_rate=0.80,
        min_non_neutral_rate=0.10,
    )


def test_triple_barrier_readiness_passes_on_balanced_labels(tmp_path: Path) -> None:
    label_dir = tmp_path / "run" / "research_labels"
    triple_rows = [_label_row(label) for label in [1, 1, -1, -1, 0, 0]]
    fee_rows = [_label_row(label) for label in [1, 1, 0, 0, 0, 0]]
    meta_rows = [_meta_row(0, base_signal=True), _meta_row(1, base_signal=True)]
    _write_label_dir(label_dir, triple_rows=triple_rows, fee_rows=fee_rows, meta_rows=meta_rows)

    report = analyze_label_readiness(run_dir=label_dir.parent, thresholds=_thresholds())

    assert report["triple_barrier"]["ready"] is True
    assert "TRIPLE_BARRIER_READY" in report["honesty_flags"]


def test_triple_barrier_readiness_warns_on_neutral_collapse(tmp_path: Path) -> None:
    label_dir = tmp_path / "run" / "research_labels"
    triple_rows = [_label_row(0) for _ in range(8)] + [_label_row(1)]
    fee_rows = [_label_row(0) for _ in range(9)]
    _write_label_dir(label_dir, triple_rows=triple_rows, fee_rows=fee_rows, meta_rows=[])

    report = analyze_label_readiness(run_dir=label_dir.parent, thresholds=_thresholds())

    assert report["triple_barrier"]["ready"] is False
    assert "TRIPLE_BARRIER_NOT_READY" in report["honesty_flags"]
    assert "LABEL_SLICE_CLASS_COLLAPSE" in report["honesty_flags"]


def test_fee_exceedance_readiness_passes_with_enough_positives(tmp_path: Path) -> None:
    label_dir = tmp_path / "run" / "research_labels"
    triple_rows = [_label_row(label) for label in [1, -1, 0, 1, -1, 0]]
    fee_rows = [_label_row(label) for label in [1, 1, 1, 0, 0, 0]]
    _write_label_dir(label_dir, triple_rows=triple_rows, fee_rows=fee_rows, meta_rows=[])

    report = analyze_label_readiness(run_dir=label_dir.parent, thresholds=_thresholds())

    assert report["fee_exceedance"]["ready"] is True
    assert "FEE_EXCEEDANCE_READY" in report["honesty_flags"]


def test_fee_exceedance_sparse_warning(tmp_path: Path) -> None:
    label_dir = tmp_path / "run" / "research_labels"
    triple_rows = [_label_row(label) for label in [1, -1, 0, 1, -1, 0]]
    fee_rows = [_label_row(label) for label in [1, 0, 0, 0, 0, 0]]
    _write_label_dir(label_dir, triple_rows=triple_rows, fee_rows=fee_rows, meta_rows=[])

    report = analyze_label_readiness(run_dir=label_dir.parent, thresholds=_thresholds())

    assert report["fee_exceedance"]["ready"] is True
    assert "FEE_EXCEEDANCE_READY" in report["honesty_flags"]


def test_meta_label_all_zero_and_no_entry_detection(tmp_path: Path) -> None:
    label_dir = tmp_path / "run" / "research_labels"
    triple_rows = [_label_row(label) for label in [1, -1, 0, 1, -1, 0]]
    fee_rows = [_label_row(label) for label in [1, 1, 0, 0, 0, 0]]
    meta_rows = [_meta_row(0, base_signal=False) for _ in range(6)]
    _write_label_dir(label_dir, triple_rows=triple_rows, fee_rows=fee_rows, meta_rows=meta_rows)

    report = analyze_label_readiness(run_dir=label_dir.parent, thresholds=_thresholds())

    assert report["meta_label"]["all_zero"] is True
    assert report["meta_label"]["candidate_entry_count"] == 0
    assert "META_LABEL_NOT_READY_ALL_ZERO" in report["honesty_flags"]
    assert "META_LABEL_NOT_READY_NO_ENTRY_EVENTS" in report["honesty_flags"]


def test_per_fold_and_symbol_low_positive_warnings(tmp_path: Path) -> None:
    label_dir = tmp_path / "run" / "research_labels"
    triple_rows = [
        _label_row(1, fold_index=0, symbol="BTC/USD"),
        _label_row(0, fold_index=1, symbol="ETH/USD"),
        _label_row(-1, fold_index=1, symbol="ETH/USD"),
        _label_row(0, fold_index=1, symbol="ETH/USD"),
        _label_row(1, fold_index=0, symbol="BTC/USD"),
        _label_row(-1, fold_index=0, symbol="BTC/USD"),
    ]
    fee_rows = [
        _label_row(
            row["label"],
            fold_index=int(row["fold_index"]),
            symbol=str(row["symbol"]),
        )
        for row in triple_rows
    ]
    _write_label_dir(label_dir, triple_rows=triple_rows, fee_rows=fee_rows, meta_rows=[])

    report = analyze_label_readiness(run_dir=label_dir.parent, thresholds=_thresholds())

    assert "LOW_MIN_FOLD_POSITIVES" in report["honesty_flags"]
    assert "LOW_MIN_SYMBOL_POSITIVES" in report["honesty_flags"]


def test_fixed_bps_flag_is_preserved(tmp_path: Path) -> None:
    label_dir = tmp_path / "run" / "research_labels"
    triple_rows = [_label_row(label) for label in [1, -1, 0, 1, -1, 0]]
    fee_rows = [_label_row(label) for label in [1, 1, 0, 0, 0, 0]]
    _write_label_dir(
        label_dir,
        triple_rows=triple_rows,
        fee_rows=fee_rows,
        meta_rows=[],
        source_flags=["MISSING_VOLATILITY_COLUMN_USING_FIXED_BPS"],
    )

    report = analyze_label_readiness(run_dir=label_dir.parent, thresholds=_thresholds())

    assert "RESEARCH_ONLY_FIXED_BPS" in report["honesty_flags"]


def test_deterministic_report_manifest_and_majority_baseline(tmp_path: Path) -> None:
    label_dir = tmp_path / "run" / "research_labels"
    triple_rows = [_label_row(label) for label in [1, 1, -1, -1, 0, 0]]
    fee_rows = [_label_row(label) for label in [1, 1, 0, 0, 0, 0]]
    _write_label_dir(label_dir, triple_rows=triple_rows, fee_rows=fee_rows, meta_rows=[])

    report = analyze_label_readiness(run_dir=label_dir.parent, thresholds=_thresholds())

    readiness_dir = Path(report["readiness_dir"])
    assert (readiness_dir / "label_readiness_report.json").exists()
    assert (readiness_dir / "label_readiness_report.md").exists()
    assert (readiness_dir / "label_readiness_by_slice.csv").exists()
    assert (readiness_dir / "tiny_baseline_feasibility.csv").exists()
    assert (readiness_dir / "label_readiness_manifest.json").exists()
    majority = [
        row for row in report["output_files"].values()
        if row.endswith("tiny_baseline_feasibility.csv")
    ]
    assert majority
    baseline_text = (readiness_dir / "tiny_baseline_feasibility.csv").read_text(
        encoding="utf-8"
    )
    assert "majority_class" in baseline_text
    assert "not comparable to runtime incumbent" in baseline_text


def test_random_baseline_reproducibility_with_fixed_seed(tmp_path: Path) -> None:
    label_dir = tmp_path / "run" / "research_labels"
    triple_rows = [_label_row(label) for label in [1, 1, -1, -1, 0, 0]]
    fee_rows = [_label_row(label) for label in [1, 1, 0, 0, 0, 0]]
    _write_label_dir(label_dir, triple_rows=triple_rows, fee_rows=fee_rows, meta_rows=[])

    first = analyze_label_readiness(
        run_dir=label_dir.parent,
        thresholds=_thresholds(),
        random_seed=7,
    )
    second = analyze_label_readiness(
        run_dir=label_dir.parent,
        thresholds=_thresholds(),
        random_seed=7,
    )

    first_csv = Path(first["output_files"]["tiny_baseline_feasibility_csv"]).read_text(
        encoding="utf-8"
    )
    second_csv = Path(second["output_files"]["tiny_baseline_feasibility_csv"]).read_text(
        encoding="utf-8"
    )
    assert first_csv == second_csv


def test_readiness_accepts_training_frame_vol_scaled_layout(tmp_path: Path) -> None:
    label_dir = tmp_path / "run" / "research_labels" / "vol_scaled"
    triple_rows = [_label_row(label) for label in [1, 1, -1, -1, 0, 0]]
    fee_rows = [_label_row(label) for label in [1, 1, 0, 0, 0, 0]]
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / "research_labels_manifest_vol_scaled.json").write_text(
        json.dumps(
            {
                "run_dir": str(label_dir.parents[1]),
                "label_dir": str(label_dir),
                "source": "training_frame",
                "honesty_flags": [
                    "SOURCE_TRAINING_FRAME",
                    "META_LABEL_NOT_APPLICABLE_NO_OOF_SIGNALS",
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_csv(label_dir / "triple_barrier_labels_vol_scaled.csv", triple_rows)
    _write_csv(label_dir / "fee_exceedance_labels_vol_scaled.csv", fee_rows)

    report = analyze_label_readiness(run_dir=label_dir.parents[1], thresholds=_thresholds())

    assert str(report["label_dir"]).endswith("research_labels\\vol_scaled") or str(
        report["label_dir"]
    ).endswith("research_labels/vol_scaled")
    assert "META_LABEL_NOT_APPLICABLE_NO_OOF_SIGNALS" in report["honesty_flags"]
    assert "META_LABEL_NOT_READY_NO_ENTRY_EVENTS" not in report["honesty_flags"]
