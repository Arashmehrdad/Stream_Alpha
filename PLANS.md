# Stream Alpha strongest-honest alignment plan

## Frozen Findings
1. M20 is active but limited and candidate-pool constrained.
2. M19 and M21 have code and read surfaces but lack runtime-generated persisted truth.
3. /health currently misreports the active top-level model identity.
4. README currently overclaims completion through M21.
5. Real trained/runtime-usable model support is narrower than design intent.
6. Research hooks do not count as trained model presence.
7. No credit for AutoGluon unless there is real training/import support plus registry/runtime-usable artifacts.

## Corrected model-stack judgment
### Present and trained now
- logistic_regression
- hist_gradient_boosting
- simple baselines only if actually trained and used

### Supported indirectly via real AutoGluon
- none until proven by real workflow plus registry/runtime artifacts

### Referenced in design but not actually trained / registry-backed / runtime-usable
- AutoGluon
- NeuralForecast NHITS
- NeuralForecast NBEATSx
- NeuralForecast TFT
- NeuralForecast PatchTST
- River
- any other named family without real artifact and runtime proof

### Not needed separately if AutoGluon becomes real later
- XGBoost
- LightGBM
- CatBoost
- tabular MLP-like learners managed under AutoGluon Tabular

### Impact
- M19: cannot be judged strong if upstream model/runtime truth is still limited
- M20: current ensemble must be judged only from the actual trained candidate pool
- M21: cannot be judged strong if underlying promotion candidate ecosystem is still narrow

## Batch matrix
| Batch | Goal | Files | Status | Targeted checks | Notes |
|---|---|---|---|---|---|
| 1 | Fix /health ensemble identity truth | app/inference/service.py, tests/... | DONE | health endpoint, targeted tests | /health now reports dynamic_ensemble when ensemble is active |
| 2 | Generate M19 drift/performance persisted truth | runner/repository/adaptation files | DONE | targeted runtime + DB checks | runtime writer now persists M19 drift/performance rows from paper feature, trace, position, and ledger truth |
| 3 | Generate M21 drift-cap persisted truth | runner/repository/continual_learning files | DONE | targeted runtime + DB checks | runtime writer now persists M21 drift caps from existing M19 drift truth only |
| 4 | Rewrite README honesty | README.md, PLANS.md | DONE | targeted readme text checks | README now no longer claims complete through M21 and includes corrected model-stack judgment |

## Current batch log
- Batch 1 complete:
  - Updated `app/inference/service.py` so `/health` reports the truthful top-level model identity when an active ensemble profile is in use.
  - Updated `tests/test_inference_api.py` to assert `model_name == "dynamic_ensemble"` and `model_artifact_path is None` under active ensemble health.
  - Targeted checks passed:
    - `python -m pytest tests\test_inference_api.py -q` -> `29 passed`
    - `python -m pylint app\inference\service.py tests\test_inference_api.py` -> `10.00/10`
  - Targeted runtime truth check passed:
    - `GET /health` now returns `model_name = "dynamic_ensemble"` and `model_artifact_path = null`
    - active ensemble fields remain aligned: `ensemble_profile_id = "m20-paper-all-minimal-20260331T205941Z"`, `ensemble_status = "ACTIVE"`, `ensemble_candidate_count = 1`
- Batch 2 complete:
  - Updated `app/trading/repository.py` with explicit adaptation-source loaders for recent feature rows and persisted trade ledger entries.
  - Updated `app/adaptation/service.py` with additive runtime writers for `adaptive_drift_state` and `adaptive_performance_windows` using existing feature, decision-trace, position, and ledger truth only.
  - Updated `app/trading/runner.py` so each completed paper cycle writes M19 persisted truth after summaries and before alert evaluation.
  - Updated focused tests in `tests/test_adaptation_service.py` and `tests/trading/test_runner_idempotency.py` to cover service-level and runner-path M19 persistence.
  - Targeted checks passed:
    - `python -m pytest tests\test_adaptation_service.py tests\trading\test_runner_idempotency.py -q` -> `20 passed`
    - `python -m pylint app\adaptation\service.py app\trading\repository.py app\trading\runner.py tests\test_adaptation_service.py tests\trading\test_runner_idempotency.py` -> `10.00/10`
  - Targeted runtime truth check passed:
    - rebuilt the paper trader image and ran a paper cycle with `docker compose --profile paper --env-file .env up -d --build trader` and `docker exec streamalpha-trader python scripts/run_paper_trader.py --once`
    - `adaptive_drift_state` now contains current rows for `BTC/USD`, `ETH/USD`, `SOL/USD`, and aggregate `ALL/ALL`
    - `adaptive_performance_windows` now contains current paper-mode rows such as `SOL/USD RANGE last_20_trades/last_50_trades/last_100_trades/last_7d/last_30d` and `ETH/USD TREND_UP ...`
- Batch 3 complete:
  - Updated `app/continual_learning/service.py` with an additive runtime writer for `continual_learning_drift_caps` that reuses persisted M19 `adaptive_drift_state` truth only.
  - Updated `app/trading/runner.py` so each paper cycle evaluates the M21 drift-cap writer immediately after the M19 adaptation writer.
  - Updated focused tests in `tests/test_continual_learning_service.py` and `tests/trading/test_runner_idempotency.py` to cover service-level and runner-path M21 persistence.
  - Targeted checks passed:
    - `python -m pytest tests\test_continual_learning_service.py tests\trading\test_runner_idempotency.py -q` -> `27 passed`
    - `python -m pylint app\continual_learning\service.py app\trading\runner.py tests\test_continual_learning_service.py tests\trading\test_runner_idempotency.py` -> `10.00/10`
  - Targeted runtime truth check passed:
    - rebuilt the paper trader image and confirmed the M21 runtime writer can persist `continual_learning_drift_caps` inside the live paper container from existing `adaptive_drift_state` rows only
    - `continual_learning_drift_caps` now contains current rows for `ALL/ALL`, `BTC/USD`, `ETH/USD`, and `SOL/USD` across `CALIBRATION_OVERLAY` and `INCREMENTAL_SHADOW_CHALLENGER`
    - honest note: the strongest runtime proof this batch used was direct in-container invocation of the M21 runtime writer after confirming the source M19 drift rows were present; the runner-path integration itself is covered by focused regression tests
- Batch 4 complete:
  - Updated `README.md` so it no longer claims the repo is complete through M21.
  - Rewrote the top-level status, M18-M21 milestone summary, M21 workflow wording, operator-console wording, and closeout so they reflect the established current state:
    - M20 is active but limited and candidate-pool constrained
    - M19 now has runtime-generated persisted drift/performance truth
    - M21 now has runtime-generated persisted drift-cap truth
    - M19 and M21 do not claim invented active profiles, experiments, or promotions beyond real persisted truth
  - Added a `Corrected model-stack judgment` section covering:
    - present and trained now
    - supported indirectly via real AutoGluon only if actually real
    - referenced in design but not actually trained / registry-backed / runtime-usable
    - not needed separately if AutoGluon becomes real later
  - Targeted checks passed:
    - `Select-String -Path README.md -Pattern 'complete through \\*\\*M21\\*\\*|Second Foundation v2 is complete\\.|Second Foundation v2 is complete through accepted M21'` -> no matches
    - `Select-String -Path README.md -Pattern 'Corrected model-stack judgment|candidate-pool constrained|runtime-generated persisted drift/performance truth|runtime-generated persisted drift-cap truth|AutoGluon'` -> expected matches present
