"""Bounded research-only AutoGluon M7 experiment discovery and summarization."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.common.serialization import make_json_safe
from app.training.dataset import load_training_config
from app.training.policy_candidate_analysis import (
    DEFAULT_ANALYSIS_DIR_NAME as DEFAULT_POLICY_ANALYSIS_DIR_NAME,
    evaluate_policy_candidates,
)


DEFAULT_CONFIG_GLOB = "training.m7.research.*.json"
DEFAULT_MAX_RESEARCH_CONFIGS = 4
DEFAULT_ANALYSIS_DIR = Path("artifacts") / "training" / "m7" / "_analysis" / "research_experiments"


@dataclass(frozen=True, slots=True)
class ResearchExperimentConfig:
    """One checked-in research config for bounded M7 experiment runs."""

    config_name: str
    config_path: Path

    def to_dict(self) -> dict[str, str]:
        return {
            "config_name": self.config_name,
            "config_path": str(self.config_path),
        }


@dataclass(frozen=True, slots=True)
class ResearchExperimentSpec:
    """One completed research experiment run to summarize."""

    config_name: str
    config_path: Path
    run_dir: Path


def discover_research_configs(
    config_dir: Path | None = None,
) -> tuple[ResearchExperimentConfig, ...]:
    """Discover and validate the bounded checked-in M7 research config set."""
    resolved_config_dir = (
        config_dir.resolve()
        if config_dir is not None
        else (_repo_root() / "configs").resolve()
    )
    if not resolved_config_dir.exists():
        raise ValueError(f"Research config directory does not exist: {resolved_config_dir}")
    if not resolved_config_dir.is_dir():
        raise ValueError(f"Research config path is not a directory: {resolved_config_dir}")

    config_paths = sorted(resolved_config_dir.glob(DEFAULT_CONFIG_GLOB))
    if not config_paths:
        raise ValueError(
            f"No bounded M7 research configs matching {DEFAULT_CONFIG_GLOB} were found under "
            f"{resolved_config_dir}"
        )
    if len(config_paths) > DEFAULT_MAX_RESEARCH_CONFIGS:
        raise ValueError(
            "Too many bounded M7 research configs were discovered. "
            f"Expected at most {DEFAULT_MAX_RESEARCH_CONFIGS}, found {len(config_paths)}."
        )

    discovered: list[ResearchExperimentConfig] = []
    for config_path in config_paths:
        load_training_config(config_path)
        discovered.append(
            ResearchExperimentConfig(
                config_name=_config_name_from_path(config_path),
                config_path=config_path.resolve(),
            )
        )
    return tuple(discovered)


def summarize_experiment_runs(
    experiment_specs: Iterable[ResearchExperimentSpec | Mapping[str, Any]],
    *,
    artifact_root: Path | None = None,
    analysis_dir: Path | None = None,
) -> dict[str, Any]:
    """Write a deterministic research-only summary for one bounded experiment batch."""
    normalized_specs = _normalize_experiment_specs(experiment_specs)
    if not normalized_specs:
        raise ValueError("At least one completed M7 research experiment run is required")

    resolved_analysis_dir = _resolve_analysis_dir(
        artifact_root=artifact_root,
        analysis_dir=analysis_dir,
    )
    resolved_analysis_dir.mkdir(parents=True, exist_ok=True)

    experiment_rows = [
        _build_experiment_row(spec)
        for spec in normalized_specs
    ]
    ranked_experiments = sorted(experiment_rows, key=_experiment_sort_key)
    best_experiment = ranked_experiments[0]

    summary_json_path = resolved_analysis_dir / "experiment_summary.json"
    summary_csv_path = resolved_analysis_dir / "experiment_summary.csv"
    summary_md_path = resolved_analysis_dir / "summary.md"

    summary_payload = {
        "analysis_dir": str(resolved_analysis_dir),
        "experiment_count": len(ranked_experiments),
        "experiments": ranked_experiments,
        "best_experiment": best_experiment,
        "output_files": {
            "experiment_summary_json": str(summary_json_path),
            "experiment_summary_csv": str(summary_csv_path),
            "summary_md": str(summary_md_path),
        },
    }

    _write_json(summary_json_path, summary_payload)
    _write_csv(
        summary_csv_path,
        [
            {
                **row,
                "is_best_experiment": row["run_id"] == best_experiment["run_id"]
                and row["config_name"] == best_experiment["config_name"],
            }
            for row in ranked_experiments
        ],
    )
    summary_md_path.write_text(_build_summary_markdown(summary_payload), encoding="utf-8")
    return make_json_safe(summary_payload)


def parse_experiment_argument(raw_value: str) -> ResearchExperimentSpec:
    """Parse one config_name::config_path::run_dir CLI argument."""
    parts = raw_value.split("::", 2)
    if len(parts) != 3:
        raise ValueError(
            "Research experiment arguments must use config_name::config_path::run_dir format"
        )
    config_name, raw_config_path, raw_run_dir = parts
    return ResearchExperimentSpec(
        config_name=str(config_name),
        config_path=Path(raw_config_path).resolve(),
        run_dir=Path(raw_run_dir).resolve(),
    )


def _normalize_experiment_specs(
    experiment_specs: Iterable[ResearchExperimentSpec | Mapping[str, Any]],
) -> list[ResearchExperimentSpec]:
    normalized: list[ResearchExperimentSpec] = []
    for spec in experiment_specs:
        if isinstance(spec, ResearchExperimentSpec):
            normalized.append(spec)
            continue
        normalized.append(
            ResearchExperimentSpec(
                config_name=str(spec["config_name"]),
                config_path=Path(str(spec["config_path"])).resolve(),
                run_dir=Path(str(spec["run_dir"])).resolve(),
            )
        )
    return normalized


def _build_experiment_row(spec: ResearchExperimentSpec) -> dict[str, Any]:
    if not spec.config_path.exists():
        raise ValueError(f"Research config path does not exist: {spec.config_path}")
    load_training_config(spec.config_path)
    if not spec.run_dir.exists():
        raise ValueError(f"Experiment run directory does not exist: {spec.run_dir}")
    if not spec.run_dir.is_dir():
        raise ValueError(f"Experiment run path is not a directory: {spec.run_dir}")

    summary_payload = _load_json(spec.run_dir / "summary.json")
    policy_summary = _load_policy_candidate_summary(spec.run_dir)
    best_candidate = policy_summary["best_candidate"]
    acceptance_payload = summary_payload.get("acceptance", {})
    winner_payload = summary_payload.get("winner", {})
    return {
        "config_name": spec.config_name,
        "config_path": str(spec.config_path),
        "run_id": spec.run_dir.name,
        "run_dir": str(spec.run_dir),
        "winner_model_name": str(winner_payload.get("model_name", "")),
        "winner_after_cost_positive": bool(
            acceptance_payload.get("winner_after_cost_positive", False)
        ),
        "meets_acceptance_target": bool(
            acceptance_payload.get("meets_acceptance_target", False)
        ),
        "best_policy_name": str(best_candidate["policy_name"]),
        "best_policy_mean_long_only_net_value_proxy": float(
            best_candidate["mean_long_only_net_value_proxy"]
        ),
        "best_policy_trade_count": int(best_candidate["trade_count"]),
        "best_policy_after_cost_positive": bool(best_candidate["after_cost_positive"]),
        "best_policy_caution_text": str(best_candidate.get("caution_text") or ""),
    }


def _load_policy_candidate_summary(run_dir: Path) -> Mapping[str, Any]:
    summary_path = run_dir / DEFAULT_POLICY_ANALYSIS_DIR_NAME / "policy_candidate_summary.json"
    if summary_path.exists():
        return _load_json(summary_path)
    return evaluate_policy_candidates(run_dir=run_dir)


def _experiment_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(row["best_policy_mean_long_only_net_value_proxy"]),
        -int(row["best_policy_trade_count"]),
        -int(bool(row["winner_after_cost_positive"])),
        str(row["config_name"]),
        str(row["run_id"]),
    )


def _build_summary_markdown(summary: Mapping[str, Any]) -> str:
    best_experiment = summary["best_experiment"]
    lines = [
        "# M7 Research Experiment Summary",
        "",
        f"- Experiment count: `{int(summary['experiment_count'])}`",
        "",
        "## Best Experiment",
        "",
        (
            f"- `{best_experiment['config_name']}` -> `{best_experiment['run_id']}` "
            f"(best_policy={best_experiment['best_policy_name']}, "
            f"best_policy_mean_net={float(best_experiment['best_policy_mean_long_only_net_value_proxy']):.6f}, "
            f"best_policy_trade_count={int(best_experiment['best_policy_trade_count'])}, "
            f"winner_after_cost_positive={bool(best_experiment['winner_after_cost_positive'])})"
        ),
        "",
        "## Experiments",
        "",
    ]
    for row in summary["experiments"]:
        lines.append(
            "- "
            f"`{row['config_name']}` -> `{row['run_id']}` "
            f"(winner={row['winner_model_name']}, "
            f"winner_after_cost_positive={bool(row['winner_after_cost_positive'])}, "
            f"meets_acceptance_target={bool(row['meets_acceptance_target'])}, "
            f"best_policy={row['best_policy_name']}, "
            f"best_policy_mean_net={float(row['best_policy_mean_long_only_net_value_proxy']):.6f}, "
            f"best_policy_trade_count={int(row['best_policy_trade_count'])}, "
            f"best_policy_after_cost_positive={bool(row['best_policy_after_cost_positive'])})"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
        ]
    )
    for label, path in summary["output_files"].items():
        lines.append(f"- {label}: `{path}`")
    lines.extend(
        [
            "",
            "This research summary is evaluation support only. It does not change production "
            "behavior, promotion semantics, or runtime policy.",
            "",
        ]
    )
    return "\n".join(lines)


def _config_name_from_path(config_path: Path) -> str:
    prefix = "training.m7.research."
    suffix = ".json"
    file_name = config_path.name
    if file_name.startswith(prefix) and file_name.endswith(suffix):
        return file_name[len(prefix):-len(suffix)]
    return config_path.stem


def _resolve_analysis_dir(
    *,
    artifact_root: Path | None,
    analysis_dir: Path | None,
) -> Path:
    if analysis_dir is not None:
        return analysis_dir.resolve()
    if artifact_root is None:
        return (_repo_root() / DEFAULT_ANALYSIS_DIR).resolve()
    return (artifact_root.resolve() / "_analysis" / "research_experiments").resolve()


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ValueError(f"Required research artifact file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(make_json_safe(dict(payload)), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    field_names = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(rows)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    """Discover bounded research configs or summarize completed experiment runs."""
    parser = argparse.ArgumentParser(
        description="Stream Alpha bounded M7 AutoGluon research experiments",
    )
    parser.add_argument(
        "--config-dir",
        help="Optional config directory for bounded research config discovery.",
    )
    parser.add_argument(
        "--artifact-root",
        help="Optional artifact root used to place the experiment summary.",
    )
    parser.add_argument(
        "--analysis-dir",
        help="Optional explicit analysis directory for summary outputs.",
    )
    parser.add_argument(
        "--list-configs",
        action="store_true",
        help="List the bounded checked-in research configs and exit.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="One config_name::config_path::run_dir triple. Can be repeated.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of plain text.",
    )
    arguments = parser.parse_args()

    try:
        if arguments.list_configs:
            configs = discover_research_configs(
                Path(arguments.config_dir) if arguments.config_dir else None
            )
            payload = {"configurations": [config.to_dict() for config in configs]}
            if arguments.json:
                print(json.dumps(make_json_safe(payload), sort_keys=True))
            else:
                for config in payload["configurations"]:
                    print(f"{config['config_name']}={config['config_path']}")
            return

        if not arguments.experiment:
            raise ValueError("At least one --experiment must be supplied when not listing configs")

        summary = summarize_experiment_runs(
            [parse_experiment_argument(raw_value) for raw_value in arguments.experiment],
            artifact_root=Path(arguments.artifact_root) if arguments.artifact_root else None,
            analysis_dir=Path(arguments.analysis_dir) if arguments.analysis_dir else None,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    if arguments.json:
        print(json.dumps(make_json_safe(summary), sort_keys=True))
        return

    best_experiment = summary["best_experiment"]
    print(f"analysis_dir={summary['analysis_dir']}")
    print(
        "best_experiment="
        f"{best_experiment['config_name']}:{best_experiment['run_id']}"
        f"(best_policy={best_experiment['best_policy_name']},"
        f" best_policy_mean_net={float(best_experiment['best_policy_mean_long_only_net_value_proxy']):.6f},"
        f" best_policy_trade_count={int(best_experiment['best_policy_trade_count'])})"
    )


if __name__ == "__main__":
    main()
