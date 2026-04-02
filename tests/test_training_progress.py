"""Focused tests for local AutoGluon fit progress inspection."""

from __future__ import annotations

import pickle
from time import sleep
from pathlib import Path

from app.training.progress import build_autogluon_fit_progress_snapshot


class _FakeGraph:
    """Minimal graph object exposing the nodes contract used by the progress helper."""

    def __init__(self, nodes: list[str]) -> None:
        self.nodes = nodes


class _FakeTrainer:
    """Minimal trainer payload that can be pickled into trainer.pkl for tests."""

    def __init__(self, *, model_best: str, nodes: list[str]) -> None:
        self.model_best = model_best
        self.model_graph = _FakeGraph(nodes)


def test_build_autogluon_fit_progress_snapshot_reports_current_best_and_latest_model(
    tmp_path: Path,
) -> None:
    """The progress helper should report visible local fit state without loading the predictor."""
    fit_dir = tmp_path / "streamalpha-autogluon-fit-test"
    models_dir = fit_dir / "predictor" / "models"
    models_dir.mkdir(parents=True)

    for name in ("LightGBM_BAG_L1", "CatBoost_BAG_L1", "WeightedEnsemble_L2"):
        model_dir = models_dir / name
        model_dir.mkdir()
        (model_dir / "marker.txt").write_text(name, encoding="utf-8")
        sleep(0.01)

    trainer = _FakeTrainer(
        model_best="WeightedEnsemble_L2",
        nodes=["LightGBM_BAG_L1", "CatBoost_BAG_L1", "WeightedEnsemble_L2", "XGBoost_BAG_L2"],
    )
    with (models_dir / "trainer.pkl").open("wb") as output_file:
        pickle.dump(trainer, output_file)

    snapshot = build_autogluon_fit_progress_snapshot(fit_dir)

    assert snapshot.current_best_model == "WeightedEnsemble_L2"
    assert snapshot.total_model_count == 4
    assert snapshot.discovered_model_count == 3
    assert snapshot.latest_model_name == "WeightedEnsemble_L2"
    assert snapshot.models_dir == str(models_dir.resolve())


def test_build_autogluon_fit_progress_snapshot_handles_uninitialized_fit_dir(
    tmp_path: Path,
) -> None:
    """The helper should stay honest before predictor/models exist."""
    fit_dir = tmp_path / "streamalpha-autogluon-fit-empty"
    fit_dir.mkdir()

    snapshot = build_autogluon_fit_progress_snapshot(fit_dir)

    assert snapshot.predictor_dir is None
    assert snapshot.models_dir is None
    assert snapshot.trainer_path is None
    assert snapshot.current_best_model is None
    assert snapshot.discovered_model_count == 0
