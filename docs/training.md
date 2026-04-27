# Training

Training and research workflows are local/operator workflows, not Docker runtime behavior.

## Dependency Boundary

- Runtime services use `requirements-runtime.txt`.
- Training and research use `requirements-training.txt`.
- `requirements.txt` points to the training superset for local development compatibility.

## Visible Training Configs

Confirmed configs include:

- `configs/training.m3.json`
- `configs/training.m7.json`
- `configs/training.m20.json`
- `configs/training.m20.colab.json`
- `configs/training.m7.research.best_quality.json`
- `configs/training.m7.research.best_quality_v150.json`
- `configs/training.m7.research.high_quality.json`

## Visible Training Scripts

- `scripts/prepare_m7_training.ps1`
- `scripts/start_m7_training.ps1`
- `scripts/start_m20_training.ps1`
- `scripts/preflight_m20_training.ps1`
- `scripts/rescore_m20_training.ps1`
- `scripts/status_m20_training.ps1`
- `scripts/run_m7_research_experiments.ps1`
- `scripts/analyze_m7_data_regime.ps1`
- `scripts/analyze_m7_thresholds.ps1`
- `scripts/evaluate_m7_policy_candidates.ps1`
- `scripts/evaluate_m7_policy_candidates_multi_run.ps1`
- `scripts/evaluate_m7_policy_replay.ps1`
- `scripts/evaluate_m7_policy_replay_multi_run.ps1`

## Registry and Champion Process

The inference runtime resolves model metadata through `app/training/registry.py`.

Do not infer promotion from a completed training run alone. Promotion/runtime truth requires registry and runtime discoverability plus a loadable artifact.

## Training Tests

Focused training tests include:

```powershell
python -m pytest `
  tests/test_training_service.py `
  tests/test_training_specialist_verdicts.py `
  tests/test_training_neuralforecast.py `
  tests/test_training_registry.py `
  tests/test_training_compare.py `
  -q
```

Inference model loading test:

```powershell
python -m pytest `
  tests/test_inference_model_loader.py `
  -q
```

## Latest M20 Score-Only Proof

`PLANS.md` records that the local M20 score-only proof now completes at `artifacts/training/m20/20260427T112021Z` and produces `summary.json` with `acceptance.verdict_basis = "incumbent_comparison"`.

The result is not promotable: both `neuralforecast_nhits` and `neuralforecast_patchtst` were rejected by incumbent comparison against `m7-20260401T043003Z`, so M20 remains `ACTIVE_WEAK`.
