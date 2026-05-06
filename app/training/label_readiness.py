"""Research-only readiness gates for M20 trading-aware label artifacts."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.common.serialization import make_json_safe
from app.training.threshold_analysis import write_csv_artifact, write_json_artifact


DEFAULT_READINESS_DIR_NAME = "readiness"
DEFAULT_MIN_TOTAL_ROWS = 1000
DEFAULT_MIN_POSITIVE_CLASS_RATE = 0.02
DEFAULT_MIN_POSITIVES_PER_FOLD = 25
DEFAULT_MIN_POSITIVES_PER_SYMBOL = 25
DEFAULT_MAX_NEUTRAL_RATE = 0.80
DEFAULT_MIN_NON_NEUTRAL_RATE = 0.10
DEFAULT_RANDOM_SEED = 1729


@dataclass(frozen=True, slots=True)
class LabelReadinessThresholds:
    """Conservative research-readiness thresholds."""

    min_total_rows: int = DEFAULT_MIN_TOTAL_ROWS
    min_positive_class_rate: float = DEFAULT_MIN_POSITIVE_CLASS_RATE
    min_positives_per_fold: int = DEFAULT_MIN_POSITIVES_PER_FOLD
    min_positives_per_symbol: int = DEFAULT_MIN_POSITIVES_PER_SYMBOL
    max_neutral_rate: float = DEFAULT_MAX_NEUTRAL_RATE
    min_non_neutral_rate: float = DEFAULT_MIN_NON_NEUTRAL_RATE


def analyze_label_readiness(
    *,
    run_dir: Path,
    thresholds: LabelReadinessThresholds | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> dict[str, Any]:
    """Analyze research-label readiness for one completed M20 run."""
    # pylint: disable=too-many-locals
    label_dir = _resolve_label_dir(run_dir)
    resolved_thresholds = thresholds or LabelReadinessThresholds()
    manifest_path = _first_existing_path(
        label_dir,
        ("research_labels_manifest.json", "research_labels_manifest_vol_scaled.json"),
    )
    triple_path = _first_existing_path(
        label_dir,
        ("triple_barrier_labels.csv", "triple_barrier_labels_vol_scaled.csv"),
    )
    fee_path = _first_existing_path(
        label_dir,
        ("fee_exceedance_labels.csv", "fee_exceedance_labels_vol_scaled.csv"),
    )
    manifest = _load_json(manifest_path)
    triple_rows = _read_csv_rows(triple_path)
    fee_rows = _read_csv_rows(fee_path)
    meta_path = label_dir / "incumbent_meta_labels.csv"
    meta_rows = _read_csv_rows(meta_path) if meta_path.exists() else []

    triple_report = _analyze_triple_barrier(triple_rows, resolved_thresholds)
    fee_report = _analyze_fee_exceedance(fee_rows, resolved_thresholds)
    meta_report = _analyze_meta_labels(
        meta_rows,
        not_applicable=manifest.get("source") == "training_frame",
    )
    slice_rows = (
        _slice_rows_for_target("triple_barrier", triple_rows, "label")
        + _slice_rows_for_target("fee_exceedance", fee_rows, "label")
        + _slice_rows_for_target("meta_label", meta_rows, "meta_label")
    )
    baseline_rows = _build_tiny_baselines(
        triple_rows=triple_rows,
        fee_rows=fee_rows,
        meta_rows=meta_rows,
        random_seed=random_seed,
    )
    honesty_flags = _build_honesty_flags(
        manifest=manifest,
        triple_report=triple_report,
        fee_report=fee_report,
        meta_report=meta_report,
        slice_rows=slice_rows,
    )
    recommendation = _recommend_next_branch(honesty_flags, triple_report, fee_report)
    readiness_dir = label_dir / DEFAULT_READINESS_DIR_NAME
    readiness_dir.mkdir(parents=True, exist_ok=True)
    output_files = {
        "label_readiness_report_json": str(readiness_dir / "label_readiness_report.json"),
        "label_readiness_report_md": str(readiness_dir / "label_readiness_report.md"),
        "label_readiness_by_slice_csv": str(readiness_dir / "label_readiness_by_slice.csv"),
        "tiny_baseline_feasibility_csv": str(readiness_dir / "tiny_baseline_feasibility.csv"),
        "label_readiness_manifest_json": str(readiness_dir / "label_readiness_manifest.json"),
    }
    report = {
        "run_dir": str(Path(run_dir).resolve()),
        "label_dir": str(label_dir),
        "readiness_dir": str(readiness_dir),
        "thresholds": _thresholds_to_dict(resolved_thresholds),
        "triple_barrier": triple_report,
        "fee_exceedance": fee_report,
        "meta_label": meta_report,
        "honesty_flags": honesty_flags,
        "recommendation": recommendation,
        "notes": [
            "Tiny baselines are feasibility diagnostics only.",
            "Results are not comparable to the runtime incumbent unless timestamps "
            "and target semantics match.",
            "No runtime inference, registry authority, promotion, execution, "
            "training, or roster behavior changed.",
        ],
        "output_files": output_files,
    }
    readiness_manifest = {
        "run_dir": report["run_dir"],
        "label_dir": str(label_dir),
        "readiness_dir": str(readiness_dir),
        "source_manifest": str(manifest_path),
        "source_honesty_flags": manifest.get("honesty_flags", []),
        "readiness_honesty_flags": honesty_flags,
        "recommendation": recommendation,
        "runtime_effect": "none_research_only",
        "output_files": output_files,
    }
    write_json_artifact(Path(output_files["label_readiness_report_json"]), report)
    write_json_artifact(Path(output_files["label_readiness_manifest_json"]), readiness_manifest)
    write_csv_artifact(Path(output_files["label_readiness_by_slice_csv"]), slice_rows)
    write_csv_artifact(Path(output_files["tiny_baseline_feasibility_csv"]), baseline_rows)
    Path(output_files["label_readiness_report_md"]).write_text(
        _build_markdown_report(report),
        encoding="utf-8",
    )
    return make_json_safe({**report, "manifest": readiness_manifest})


def _resolve_label_dir(run_dir: Path) -> Path:
    resolved_run_dir = Path(run_dir).resolve()
    if resolved_run_dir.name == DEFAULT_READINESS_DIR_NAME:
        return resolved_run_dir.parent
    if resolved_run_dir.name in {"research_labels", "vol_scaled"}:
        return resolved_run_dir
    vol_scaled_dir = resolved_run_dir / "research_labels" / "vol_scaled"
    if vol_scaled_dir.exists():
        return vol_scaled_dir
    return resolved_run_dir / "research_labels"


def _first_existing_path(label_dir: Path, names: Sequence[str]) -> Path:
    for name in names:
        path = label_dir / name
        if path.exists():
            return path
    return label_dir / names[0]


def _thresholds_to_dict(thresholds: LabelReadinessThresholds) -> dict[str, Any]:
    return {
        "min_total_rows": thresholds.min_total_rows,
        "min_positive_class_rate": thresholds.min_positive_class_rate,
        "min_positives_per_fold": thresholds.min_positives_per_fold,
        "min_positives_per_symbol": thresholds.min_positives_per_symbol,
        "max_neutral_rate": thresholds.max_neutral_rate,
        "min_non_neutral_rate": thresholds.min_non_neutral_rate,
    }


def _analyze_triple_barrier(
    rows: Sequence[Mapping[str, str]],
    thresholds: LabelReadinessThresholds,
) -> dict[str, Any]:
    distribution = _label_distribution(rows, "label")
    positive_count = int(distribution["label_counts"].get("1", 0))
    negative_count = int(distribution["label_counts"].get("-1", 0))
    neutral_count = int(distribution["label_counts"].get("0", 0))
    total_count = int(distribution["total_rows"])
    positive_rate = _safe_ratio(positive_count, total_count)
    negative_rate = _safe_ratio(negative_count, total_count)
    neutral_rate = _safe_ratio(neutral_count, total_count)
    min_class_count = min((positive_count, negative_count, neutral_count), default=0)
    min_class_rate = min((positive_rate, negative_rate, neutral_rate), default=0.0)
    non_neutral_rate = positive_rate + negative_rate
    ready = (
        total_count >= thresholds.min_total_rows
        and positive_rate >= thresholds.min_positive_class_rate
        and negative_rate >= thresholds.min_positive_class_rate
        and non_neutral_rate >= thresholds.min_non_neutral_rate
        and neutral_rate <= thresholds.max_neutral_rate
    )
    return {
        **distribution,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "positive_rate": positive_rate,
        "negative_rate": negative_rate,
        "neutral_rate": neutral_rate,
        "minimum_class_count": min_class_count,
        "minimum_class_rate": min_class_rate,
        "non_neutral_rate": non_neutral_rate,
        "ready": ready,
    }


def _analyze_fee_exceedance(
    rows: Sequence[Mapping[str, str]],
    thresholds: LabelReadinessThresholds,
) -> dict[str, Any]:
    distribution = _label_distribution(rows, "label")
    positive_count = int(distribution["label_counts"].get("1", 0))
    total_count = int(distribution["total_rows"])
    non_event_count = total_count - positive_count
    positive_rate = _safe_ratio(positive_count, total_count)
    ready = (
        total_count >= thresholds.min_total_rows
        and positive_rate >= thresholds.min_positive_class_rate
    )
    return {
        **distribution,
        "positive_count": positive_count,
        "positive_rate": positive_rate,
        "non_event_count": non_event_count,
        "non_event_rate": _safe_ratio(non_event_count, total_count),
        "ready": ready,
    }


def _analyze_meta_labels(
    rows: Sequence[Mapping[str, str]],
    *,
    not_applicable: bool = False,
) -> dict[str, Any]:
    distribution = _label_distribution(rows, "meta_label")
    total_count = int(distribution["total_rows"])
    positive_count = int(distribution["label_counts"].get("1", 0))
    zero_count = int(distribution["label_counts"].get("0", 0))
    candidate_entry_count = sum(1 for row in rows if _truthy(row.get("base_signal")))
    all_zero = total_count > 0 and zero_count == total_count
    all_one = total_count > 0 and positive_count == total_count
    ready = total_count > 0 and candidate_entry_count > 0 and not all_zero and not all_one
    return {
        **distribution,
        "candidate_entry_count": candidate_entry_count,
        "positive_count": positive_count,
        "zero_count": zero_count,
        "all_zero": all_zero,
        "all_one": all_one,
        "not_applicable": not_applicable,
        "ready": ready,
    }


def _slice_rows_for_target(
    target_name: str,
    rows: Sequence[Mapping[str, str]],
    label_column: str,
) -> list[dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    for slice_column in ("fold_index", "symbol", "regime_label"):
        grouped: dict[str, list[Mapping[str, str]]] = {}
        for row in rows:
            value = row.get(slice_column)
            if value is None or value == "":
                continue
            grouped.setdefault(str(value), []).append(row)
        for slice_value, slice_rows in sorted(grouped.items()):
            distribution = _label_distribution(slice_rows, label_column)
            positive_count = int(distribution["label_counts"].get("1", 0))
            negative_count = int(distribution["label_counts"].get("-1", 0))
            neutral_count = int(distribution["label_counts"].get("0", 0))
            total_count = int(distribution["total_rows"])
            output_rows.append(
                {
                    "target": target_name,
                    "slice_column": slice_column,
                    "slice_value": slice_value,
                    "total_rows": total_count,
                    "positive_count": positive_count,
                    "positive_rate": _safe_ratio(positive_count, total_count),
                    "negative_count": negative_count,
                    "negative_rate": _safe_ratio(negative_count, total_count),
                    "neutral_count": neutral_count,
                    "neutral_rate": _safe_ratio(neutral_count, total_count),
                    "class_collapse": _slice_class_collapsed(distribution),
                }
            )
    return output_rows


def _build_tiny_baselines(
    *,
    triple_rows: Sequence[Mapping[str, str]],
    fee_rows: Sequence[Mapping[str, str]],
    meta_rows: Sequence[Mapping[str, str]],
    random_seed: int,
) -> list[dict[str, Any]]:
    baseline_rows: list[dict[str, Any]] = []
    baseline_rows.extend(
        _baseline_rows_for_target("triple_barrier", triple_rows, "label", random_seed)
    )
    baseline_rows.extend(
        _baseline_rows_for_target("fee_exceedance", fee_rows, "label", random_seed)
    )
    baseline_rows.extend(
        _baseline_rows_for_target("meta_label", meta_rows, "meta_label", random_seed)
    )
    return baseline_rows


def _baseline_rows_for_target(
    target_name: str,
    rows: Sequence[Mapping[str, str]],
    label_column: str,
    random_seed: int,
) -> list[dict[str, Any]]:
    labels = [
        _as_int(row.get(label_column))
        for row in rows
        if _as_int(row.get(label_column)) is not None
    ]
    if not labels:
        return []
    majority_label = _majority_label(labels)
    majority_predictions = [majority_label for _ in labels]
    rng = random.Random(random_seed)
    unique_labels = sorted(set(labels))
    random_predictions = [rng.choice(unique_labels) for _ in labels]
    output_rows = [
        _classification_metrics_row(
            target_name,
            "majority_class",
            labels,
            majority_predictions,
            note="baseline only; not comparable to runtime incumbent",
        ),
        _classification_metrics_row(
            target_name,
            f"stratified_random_seed_{random_seed}",
            labels,
            random_predictions,
            note="fixed-seed random feasibility diagnostic only",
        ),
    ]
    if rows and "probability" in rows[0]:
        score_predictions = [int(float(row.get("probability", 0.0)) >= 0.5) for row in rows]
        output_rows.append(
            _classification_metrics_row(
                target_name,
                "score_only_probability_0_50",
                labels,
                score_predictions,
                note="score-only diagnostic; not a candidate model",
            )
        )
    return output_rows


def _classification_metrics_row(
    target_name: str,
    baseline_name: str,
    labels: Sequence[int],
    predictions: Sequence[int],
    *,
    note: str,
) -> dict[str, Any]:
    positive_label = 1
    correct = sum(1 for label, prediction in zip(labels, predictions) if label == prediction)
    true_positive = sum(
        1 for label, prediction in zip(labels, predictions)
        if label == positive_label and prediction == positive_label
    )
    false_positive = sum(
        1 for label, prediction in zip(labels, predictions)
        if label != positive_label and prediction == positive_label
    )
    false_negative = sum(
        1 for label, prediction in zip(labels, predictions)
        if label == positive_label and prediction != positive_label
    )
    return {
        "target": target_name,
        "baseline": baseline_name,
        "row_count": len(labels),
        "class_balance": json.dumps(_class_balance(labels), sort_keys=True),
        "accuracy": _safe_ratio(correct, len(labels)),
        "balanced_accuracy": _balanced_accuracy(labels, predictions),
        "positive_precision": _safe_ratio(true_positive, true_positive + false_positive),
        "positive_recall": _safe_ratio(true_positive, true_positive + false_negative),
        "coverage": 1.0,
        "note": note,
    }


def _build_honesty_flags(
    *,
    manifest: Mapping[str, Any],
    triple_report: Mapping[str, Any],
    fee_report: Mapping[str, Any],
    meta_report: Mapping[str, Any],
    slice_rows: Sequence[Mapping[str, Any]],
) -> list[str]:
    flags: list[str] = ["NOT_RUNTIME_COMPARABLE", "BASELINE_ONLY_NOT_PROMOTABLE"]
    if "MISSING_VOLATILITY_COLUMN_USING_FIXED_BPS" in manifest.get("honesty_flags", []):
        flags.append("RESEARCH_ONLY_FIXED_BPS")
    flags.append("TRIPLE_BARRIER_READY" if triple_report["ready"] else "TRIPLE_BARRIER_NOT_READY")
    flags.append("FEE_EXCEEDANCE_READY" if fee_report["ready"] else "FEE_EXCEEDANCE_SPARSE")
    if meta_report.get("not_applicable"):
        flags.append("META_LABEL_NOT_APPLICABLE_NO_OOF_SIGNALS")
    elif meta_report["all_zero"]:
        flags.append("META_LABEL_NOT_READY_ALL_ZERO")
    if int(meta_report["candidate_entry_count"]) == 0 and not meta_report.get("not_applicable"):
        flags.append("META_LABEL_NOT_READY_NO_ENTRY_EVENTS")
    if any(row["class_collapse"] for row in slice_rows):
        flags.append("LABEL_SLICE_CLASS_COLLAPSE")
    if _min_slice_positive_count(slice_rows, "fold_index") < DEFAULT_MIN_POSITIVES_PER_FOLD:
        flags.append("LOW_MIN_FOLD_POSITIVES")
    if _min_slice_positive_count(slice_rows, "symbol") < DEFAULT_MIN_POSITIVES_PER_SYMBOL:
        flags.append("LOW_MIN_SYMBOL_POSITIVES")
    return sorted(dict.fromkeys(flags))


def _recommend_next_branch(
    flags: Sequence[str],
    triple_report: Mapping[str, Any],
    fee_report: Mapping[str, Any],
) -> str:
    if "RESEARCH_ONLY_FIXED_BPS" in flags:
        return "C. collect/add volatility features before training"
    if (
        "FEE_EXCEEDANCE_READY" in flags
        and fee_report["positive_rate"] >= triple_report["positive_rate"]
    ):
        return "B. train a tiny fee-exceedance baseline in the next batch"
    if "TRIPLE_BARRIER_READY" in flags:
        return "A. train a tiny triple-barrier baseline in the next batch"
    if "META_LABEL_NOT_READY_ALL_ZERO" in flags:
        return "D. reject this target space for meta-labeling"
    return "E. keep M20 research-negative"


def _build_markdown_report(report: Mapping[str, Any]) -> str:
    triple = report["triple_barrier"]
    fee = report["fee_exceedance"]
    meta = report["meta_label"]
    flags = ", ".join(report["honesty_flags"])
    return "\n".join(
        [
            "# M20 Label Readiness Report",
            "",
            f"- Run directory: `{report['run_dir']}`",
            f"- Honesty flags: `{flags}`",
            f"- Recommendation: `{report['recommendation']}`",
            "",
            "## Triple-Barrier Readiness",
            "",
            f"- Total rows: `{triple['total_rows']}`",
            "- Positive/negative/neutral rates: "
            f"`{triple['positive_rate']:.6f}` / "
            f"`{triple['negative_rate']:.6f}` / "
            f"`{triple['neutral_rate']:.6f}`",
            "- Minimum class count/rate: "
            f"`{triple['minimum_class_count']}` / "
            f"`{triple['minimum_class_rate']:.6f}`",
            f"- Ready: `{triple['ready']}`",
            "",
            "## Fee-Exceedance Readiness",
            "",
            f"- Total rows: `{fee['total_rows']}`",
            f"- Positive rate: `{fee['positive_rate']:.6f}`",
            f"- Ready: `{fee['ready']}`",
            "",
            "## Meta-Label Readiness",
            "",
            f"- Candidate-entry count: `{meta['candidate_entry_count']}`",
            f"- All-zero labels: `{meta['all_zero']}`",
            f"- Ready: `{meta['ready']}`",
            "",
            "Tiny baselines are feasibility diagnostics only and are not comparable "
            "to the runtime incumbent.",
            "No runtime inference, registry authority, promotion, execution, "
            "training, thresholds, or roster behavior changed.",
            "",
        ]
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return [dict(row) for row in csv.DictReader(input_file)]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Missing research label manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Research label manifest must be a JSON object: {path}")
    return dict(payload)


def _label_distribution(rows: Sequence[Mapping[str, str]], label_column: str) -> dict[str, Any]:
    labels = [
        _as_int(row.get(label_column))
        for row in rows
        if _as_int(row.get(label_column)) is not None
    ]
    counts = {str(label): labels.count(label) for label in sorted(set(labels))}
    total_rows = len(labels)
    return {
        "label_column": label_column,
        "total_rows": total_rows,
        "label_counts": counts,
        "class_balance": {
            label: count / total_rows if total_rows else 0.0
            for label, count in counts.items()
        },
    }


def _slice_class_collapsed(distribution: Mapping[str, Any]) -> bool:
    counts = distribution["label_counts"]
    populated_class_count = len([count for count in counts.values() if count])
    return int(distribution["total_rows"]) > 0 and populated_class_count <= 1


def _min_slice_positive_count(rows: Sequence[Mapping[str, Any]], slice_column: str) -> int:
    counts = [
        int(row["positive_count"])
        for row in rows
        if row["slice_column"] == slice_column
    ]
    return min(counts) if counts else 0


def _majority_label(labels: Sequence[int]) -> int:
    return sorted(set(labels), key=lambda label: (-labels.count(label), label))[0]


def _class_balance(labels: Sequence[int]) -> dict[str, float]:
    return {
        str(label): labels.count(label) / len(labels)
        for label in sorted(set(labels))
    }


def _balanced_accuracy(labels: Sequence[int], predictions: Sequence[int]) -> float:
    recalls = []
    for label in sorted(set(labels)):
        total = sum(1 for actual in labels if actual == label)
        correct = sum(
            1 for actual, predicted in zip(labels, predictions)
            if actual == label and predicted == label
        )
        recalls.append(_safe_ratio(correct, total))
    return sum(recalls) / len(recalls) if recalls else 0.0


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0
