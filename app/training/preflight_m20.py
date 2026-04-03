"""Lightweight preflight checks for manual M20 specialist training runs."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from app.training.dataset import load_training_config
from app.training.neuralforecast import (
    build_neuralforecast_nhits_classifier,
    build_neuralforecast_patchtst_classifier,
)
from app.training.registry import list_registry_entries, repo_root


_SPECIALIST_MODEL_BUILDERS = {
    "neuralforecast_nhits": build_neuralforecast_nhits_classifier,
    "neuralforecast_patchtst": build_neuralforecast_patchtst_classifier,
}
_SPECIALIST_MODEL_FAMILIES = {
    "NEURALFORECAST_NHITS",
    "NEURALFORECAST_PATCHTST",
}


def build_m20_preflight_report(
    *,
    config_path: Path,
    require_gpu: bool = False,
) -> dict[str, Any]:
    """Return one JSON-safe preflight report for a manual M20 specialist run."""
    resolved_config_path = Path(config_path).resolve()
    blockers: list[str] = []
    warnings: list[str] = []

    report: dict[str, Any] = {
        "config_path": str(resolved_config_path),
        "require_gpu": bool(require_gpu),
        "config_ok": False,
        "artifact_root": None,
        "source_table": None,
        "symbols": [],
        "time_column": None,
        "configured_models": [],
        "model_preflight": {},
        "runtime": {},
        "registry": {},
        "blockers": blockers,
        "warnings": warnings,
        "ready_for_manual_training": False,
        "preferred_execution_device": "cpu",
    }

    try:
        config = load_training_config(resolved_config_path)
    except Exception as error:  # pragma: no cover - exercised through CLI/script path
        blockers.append(f"Could not load training config: {error}")
        return report

    report["config_ok"] = True
    report["artifact_root"] = config.artifact_root
    report["source_table"] = config.source_table
    report["symbols"] = list(config.symbols)
    report["time_column"] = config.time_column
    report["configured_models"] = list(config.models)

    missing_specialists = [
        model_name
        for model_name in _SPECIALIST_MODEL_BUILDERS
        if model_name not in config.models
    ]
    if missing_specialists:
        blockers.append(
            "training.m20 specialist config is missing required model(s): "
            f"{missing_specialists}"
        )

    for model_name in _SPECIALIST_MODEL_BUILDERS:
        model_config = config.models.get(model_name)
        if model_config is None:
            continue
        builder = _SPECIALIST_MODEL_BUILDERS[model_name]
        wrapper = builder(dict(model_config))
        training_config = wrapper.get_training_config()
        report["model_preflight"][model_name] = {
            "model_family": training_config["model_family"],
            "candidate_role": training_config["candidate_role"],
            "scope_regimes": list(training_config["scope_regimes"]),
            "lookback_candles": int(wrapper.get_sequence_lookback_candles()),
            "dataset_mode": training_config.get("dataset_mode"),
            "batch_size": int(training_config["batch_size"]),
            "valid_batch_size": training_config["model_kwargs"].get("valid_batch_size"),
            "windows_batch_size": training_config["model_kwargs"].get("windows_batch_size"),
            "inference_windows_batch_size": training_config["model_kwargs"].get(
                "inference_windows_batch_size"
            ),
            "step_size": training_config["model_kwargs"].get("step_size"),
            "precision": training_config["model_kwargs"].get("precision"),
            "accelerator": training_config["model_kwargs"].get("accelerator", "unspecified"),
            "devices": training_config["model_kwargs"].get("devices", "unspecified"),
        }

    lightning_probe = _probe_optional_module("lightning")
    neuralforecast_probe = _probe_optional_module("neuralforecast")
    torch_probe = _probe_torch_runtime()
    report["runtime"] = {
        "lightning": lightning_probe,
        "neuralforecast": neuralforecast_probe,
        "torch": torch_probe,
    }

    if not bool(lightning_probe["installed"]):
        blockers.append(
            "Missing optional dependency: lightning. Install requirements before M20 training."
        )
    if not bool(neuralforecast_probe["installed"]):
        blockers.append(
            "Missing optional dependency: neuralforecast. Install requirements before M20 training."
        )
    if not bool(torch_probe["installed"]):
        blockers.append(
            "Missing dependency: torch. Install requirements before M20 training."
        )

    cuda_available = bool(torch_probe.get("cuda_available"))
    if cuda_available:
        report["preferred_execution_device"] = "gpu"
    else:
        warnings.append(
            "CUDA is not currently available in this environment; the M20 run will fall back to CPU."
        )
    if require_gpu and not cuda_available:
        blockers.append(
            "GPU was required for this preflight, but torch does not currently report CUDA availability."
        )

    registry_entries = list_registry_entries(registry_root=repo_root() / "artifacts" / "registry")
    specialist_entries = [
        {
            "model_version": str(entry["model_version"]),
            "model_name": str(entry["model_name"]),
            "model_family": str((entry.get("metadata") or {}).get("model_family", "")),
            "candidate_role": str((entry.get("metadata") or {}).get("candidate_role", "")),
        }
        for entry in registry_entries
        if str((entry.get("metadata") or {}).get("model_family", "")) in _SPECIALIST_MODEL_FAMILIES
    ]
    report["registry"] = {
        "registry_entry_count": len(registry_entries),
        "specialist_entry_count": len(specialist_entries),
        "specialist_entries": specialist_entries,
    }
    if not specialist_entries:
        warnings.append(
            "No real registry-backed NHITS or PatchTST specialist candidates exist yet; this batch only prepares the first real run."
        )

    report["ready_for_manual_training"] = not blockers
    return report


def _probe_optional_module(module_name: str) -> dict[str, Any]:
    """Return a small import-availability probe for one optional dependency."""
    try:
        module = importlib.import_module(module_name)
    except Exception as error:  # pragma: no cover - small environment probe
        return {
            "installed": False,
            "version": None,
            "detail": f"{type(error).__name__}: {error}",
        }
    return {
        "installed": True,
        "version": str(getattr(module, "__version__", "unknown")),
        "detail": None,
    }


def _probe_torch_runtime() -> dict[str, Any]:
    """Return the local torch and CUDA visibility status without starting training."""
    try:
        torch = importlib.import_module("torch")
    except Exception as error:  # pragma: no cover - small environment probe
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "device_count": 0,
            "device_names": [],
            "cuda_version": None,
            "detail": f"{type(error).__name__}: {error}",
        }

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_names = [
        str(torch.cuda.get_device_name(index))
        for index in range(device_count)
    ]
    return {
        "installed": True,
        "version": str(getattr(torch, "__version__", "unknown")),
        "cuda_available": cuda_available,
        "device_count": device_count,
        "device_names": device_names,
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "detail": None,
    }


def main() -> None:
    """Run the bounded M20 preflight checks and print a JSON summary."""
    parser = argparse.ArgumentParser(
        description="Run lightweight preflight checks for Stream Alpha M20 specialist training"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/training.m20.json",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail the preflight if CUDA is not currently available.",
    )
    arguments = parser.parse_args()

    report = build_m20_preflight_report(
        config_path=Path(arguments.config),
        require_gpu=bool(arguments.require_gpu),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["blockers"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
