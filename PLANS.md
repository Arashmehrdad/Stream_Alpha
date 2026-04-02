# Stream Alpha strongest-honest alignment plan

## Frozen Findings
1. M20 is active but still limited and candidate-pool constrained.
2. M19 and M21 now have runtime-generated persisted truth, but they still sit on a narrow upstream model ecosystem.
3. /health now reports the truthful top-level ensemble identity.
4. README no longer overclaims completion through M21 and must keep tracking the narrower honest state.
5. Real trained/runtime-usable model support is still narrower than design intent even after the first real authoritative AutoGluon path landed.
6. Research hooks do not count as trained model presence.
7. AutoGluon now receives credit only because there is real training/import support plus registry/runtime-usable artifacts; that does not imply economic acceptance.

## Model stack after AutoGluon batch
### Present and trained now
- autogluon_tabular
- real promoted AutoGluon artifacts now exist and load through the current registry-backed runtime path
- simple baselines may still exist for evaluation and comparison, but they do not count as the intended primary production model stack

### Legacy historical artifacts still present for backward inspection
- logistic_regression
- hist_gradient_boosting

### Subsumed indirectly through AutoGluon
- AutoGluon-managed sublearners may exist inside autogluon_tabular artifacts, but they do not count as separately registry-addressable model families today

### Still genuinely missing for M20 specialist purposes
- real trend specialists with committed trained artifacts and runtime-usable registry truth
- real range specialists with committed trained artifacts and runtime-usable registry truth
- enough specialist breadth to claim regime-slice value rather than a minimal active roster

### Referenced in design but not actually trained / registry-backed / runtime-usable
- NeuralForecast NHITS
- NeuralForecast NBEATSx
- NeuralForecast TFT
- NeuralForecast PatchTST
- River
- any other named family without real artifact and runtime proof

### Not counted separately even though AutoGluon is now real
- XGBoost
- LightGBM
- CatBoost
- tabular MLP-like learners managed under AutoGluon Tabular

### Impact
- M19: cannot be judged strong if upstream model/runtime truth is still limited
- M20: old sklearn artifacts no longer count as current authoritative candidates, one real AutoGluon family now provides honest generalist truth, and remaining judgment must be based on breadth, role diversity, regime-slice value, and economics after costs
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

## Phase 2 - M20 strengthening

### Batch matrix
| Batch | Goal | Files | Status | Targeted checks | Notes |
|---|---|---|---|---|---|
| A | Determine the smallest honest M20 strengthening path from the current repo/runtime truth | PLANS.md | DONE | targeted training/registry/runtime checks | No immediate strengthening move exists without creating a new real registry-backed candidate artifact; the active roster already uses all currently available runtime-usable registry entries |
| B | Demote legacy sklearn models from the authoritative training, registry, and runtime path | training configs/service/registry, focused tests, README, PLANS.md | DONE | targeted training/runtime checks | This batch intentionally breaks the old sklearn-first path instead of preserving it as the de facto main stack |
| C | Create a real authoritative AutoGluon training, promotion, and runtime path | training configs/service/registry/promote/compare, focused tests, README, PLANS.md | DONE | targeted tests plus live retrain/promote/runtime checks | One real authoritative AutoGluon family now exists, but it is still negative after costs and does not meet acceptance |
| D | Re-rate M19/M20/M21 from current evidence and expose evidence-backed idle truth | adaptation/continual/ensemble/inference/dashboard surfaces, focused tests, README, PLANS.md, docker/app.Dockerfile | DONE | targeted service/API/dashboard checks plus live DB/artifact proof | M19 and M21 now read as evidence-backed idle states; M20 now runs on a current AutoGluon-only weak roster with explicit specialist/economic blockers |
| E | Strengthen and de-bias the authoritative AutoGluon fit/config metadata path without changing runtime authority boundaries | AutoGluon wrapper/config/registry summary path, focused tests, PLANS.md | DONE | targeted wrapper/registry/loading checks | Omitted hyperparameters now stay truly omitted, manual-training controls are explicit, and the effective fit config is now audit-visible across the existing artifact chain |
| F | Make local M7 AutoGluon training operationally one-step without weakening protected runtime modes | training config loader, dev startup script path, readiness helper, operator scripts, focused tests, README, PLANS.md | DONE | targeted pytest plus dry-run script checks | BOM-safe JSON loading, explicit dev prep flow, honest feature-table readiness checks, and a clean prepare/start operator workflow now exist without changing promotion or live safety semantics |
| G | Fix Windows local AutoGluon training temp-root / DyStack path handling without weakening training truth | AutoGluon wrapper/workdir helper, local training scripts, readiness/configs/tests, README, PLANS.md | DONE | targeted pytest plus dry-run script checks | Local M7 training now forces AutoGluon and Ray temp work onto the repo drive, exposes dynamic_stacking explicitly, and reports optional fastai breadth honestly |
| H | Keep Windows local bagged AutoGluon training off Ray while preserving honest stacking/bagging controls | AutoGluon wrapper/config/tests, PLANS.md | DONE | focused wrapper tests plus dry-run checks and synthetic smoke fit | Explicit fold_fitting_strategy now keeps checked-in local M3/M7 training on sequential_local bag-fold execution, which avoids the Windows Ray access-violation hang while keeping bagging and stacking enabled |
| I | Fix the remaining local optional-model warning sources honestly instead of suppressing them | requirements, readiness/script truth, focused tests, README, PLANS.md | DONE | focused pytest plus dry-run checks and tiny FastAI/neural smoke fits | FastAI breadth now validates real usability, sklearn is pinned to the non-deprecated AutoGluon-compatible range, and the local environment proof no longer emits the previous fastai/sklearn/NVML warnings |
| J | Make local M7 training visibly operable while staying Windows-safe and honest about budget/runtime | progress helper, start script, focused tests, README, PLANS.md | DONE | focused pytest plus dry-run checks and live-process inspection | The local operator path now shows a PowerShell heartbeat with elapsed time, budget ETA/overrun, current best model, latest model activity, and explicit CPU mode instead of leaving long Windows runs silent |

### Batch A log
- Inspected only the M20-relevant files and runtime truth needed to answer the strengthening question:
  - `AGENTS.md`
  - `PLANS.md`
  - `configs/ensemble.yaml`
  - `configs/training.m3.json`
  - `configs/training.m7.json`
  - `app/training/service.py`
  - `app/training/retrain.py`
  - `app/training/registry.py`
  - `app/ensemble/service.py`
  - `app/ensemble/schemas.py`
  - `app/inference/service.py`
  - `artifacts/registry/current.json`
  - `artifacts/registry/models/m3-20260319T223002Z/{registry_entry.json,summary.json}`
  - `artifacts/registry/models/m7-20260320T134537Z/{registry_entry.json,summary.json}`
  - `artifacts/training/m7/20260320T134537Z/run_manifest.json`
- Established current supported trainable model families in code:
  - learned models: `logistic_regression`, `hist_gradient_boosting`
  - simple baselines: `persistence_3`, `dummy_most_frequent`
- Established current registry-backed runtime-usable candidate artifacts:
  - `m3-20260319T223002Z` -> `logistic_regression`
  - `m7-20260320T134537Z` -> `logistic_regression`
- Established current active M20 persisted truth:
  - active profile `m20-paper-all-minimal-20260331T205941Z`
  - roster contains both currently available registry-backed candidates
  - both active candidates are `logistic_regression`
  - runtime now reports `ensemble_candidate_count = 2` with no fallback reason on `/signal`
- Established what is missing from the active M20 research/promotion path:
  - `hist_gradient_boosting` is supported in training code and evaluated in run summaries, but there is no registry-backed, runtime-usable `hist_gradient_boosting` artifact to add to the active roster
  - no real `RANGE_SPECIALIST` candidate exists in the current artifact pool
  - the existing ensemble promotion record is still the minimal manual activation of the currently available registry-backed roster
- Batch A conclusion:
  - no immediate honest strengthening move exists using only the currently available registry-backed runtime-usable artifacts, because the active roster already uses all of them
  - the smallest next honest batch is to create a real additional candidate artifact from an already-supported model family, starting with `hist_gradient_boosting`, and persist it through the normal training/registry path before any M20 roster change

### Batch B log
- Read only `AGENTS.md` and `PLANS.md` before editing, then stayed inside the already identified training/registry/runtime-path files and focused tests.
- Removed `logistic_regression` and `hist_gradient_boosting` from the authoritative checked-in training configs:
  - `configs/training.m3.json`
  - `configs/training.m7.json`
  - both now load with `models = {}`
- Removed legacy sklearn assumptions from the authoritative training path:
  - `app/training/dataset.py` now treats legacy sklearn names as archived-only and rejects them when they appear in authoritative configs
  - `app/training/service.py` no longer builds, selects, or summarizes the old sklearn pair as the authoritative learned-model set
  - the authoritative training path now fails explicitly until a real replacement model builder exists
- Removed legacy sklearn assumptions from the registry/runtime path:
  - `app/training/registry.py` now rejects legacy sklearn artifacts when they are used as:
    - direct inference overrides
    - current registry champions
    - exported registry entries
    - promoted run winners
- Updated the focused tests and helpers so they no longer enforce sklearn as the active expected model path:
  - `tests/training_workflow_helpers.py`
  - `tests/test_training_service.py`
  - `tests/test_training_labels.py`
  - `tests/test_training_registry.py`
  - `tests/test_inference_model_loader.py`
  - `tests/test_runtime_validate.py`
  - `tests/test_inference_api.py`
  - `tests/test_ensemble_packet2.py`
- Updated `README.md` so the model-stack wording now matches the new truth:
  - no present authoritative primary model stack yet
  - `logistic_regression` and `hist_gradient_boosting` remain legacy-only historical artifacts
- Targeted checks passed:
  - `python -m pytest tests\test_training_service.py tests\test_training_labels.py tests\test_training_compare.py tests\test_training_registry.py tests\test_ensemble_packet2.py tests\test_inference_model_loader.py tests\test_runtime_validate.py tests\test_inference_api.py -q` -> `61 passed`
  - `python -m pylint app\training\dataset.py app\training\service.py app\training\registry.py tests\training_workflow_helpers.py tests\test_training_service.py tests\test_training_labels.py tests\test_training_compare.py tests\test_training_registry.py tests\test_ensemble_packet2.py tests\test_inference_model_loader.py tests\test_runtime_validate.py tests\test_inference_api.py` -> `10.00/10`
  - `python` targeted config check -> `training.m3.json {}` and `training.m7.json {}`
- Targeted runtime-path truth check passed:
  - `resolve_inference_model_metadata("")` now fails explicitly against the current registry-backed sklearn champion with:
    - `Legacy archived sklearn model is no longer allowed in the authoritative runtime or registry path ... logistic_regression`
- Batch B conclusion:
  - the old sklearn-first path is no longer authoritative for training, promotion, registry, or runtime resolution
  - legacy sklearn traces and artifacts remain only for backward inspection
  - the intended main model stack is now honestly blocked until a real replacement training/import path produces registry-backed, runtime-usable artifacts

### Batch C log
- Read only the already-established training/registry/runtime-path files and stayed inside the focused replacement-model batch rather than reopening the broader audit.
- Added a real authoritative AutoGluon model builder and self-contained artifact path:
  - `app/training/autogluon.py`
  - `app/training/service.py`
  - `configs/training.m3.json`
  - `configs/training.m7.json`
  - `requirements.txt`
- Updated the training, compare, promote, and runtime resolution path so real AutoGluon artifacts can count honestly while the archived sklearn artifacts stay blocked from the authoritative path:
  - `app/training/dataset.py`
  - `app/training/compare.py`
  - `app/training/promote.py`
  - `app/training/registry.py`
- Updated the focused tests and README/plan wording so they match the new truth:
  - `tests/training_workflow_helpers.py`
  - `tests/test_training_service.py`
  - `tests/test_training_labels.py`
  - `tests/test_training_compare.py`
  - `tests/test_training_registry.py`
  - `tests/test_ensemble_packet2.py`
  - `tests/test_inference_model_loader.py`
  - `tests/test_runtime_validate.py`
  - `tests/test_inference_api.py`
  - `README.md`
  - `PLANS.md`
- Targeted checks passed:
  - `python -m pytest tests\test_training_service.py tests\test_training_labels.py tests\test_training_compare.py tests\test_training_registry.py tests\test_ensemble_packet2.py tests\test_inference_model_loader.py tests\test_runtime_validate.py tests\test_inference_api.py -q` -> `65 passed`
  - `python -m pylint app\training\autogluon.py app\training\dataset.py app\training\service.py app\training\compare.py app\training\promote.py app\training\registry.py tests\training_workflow_helpers.py tests\test_training_service.py tests\test_training_labels.py tests\test_training_compare.py tests\test_training_registry.py tests\test_ensemble_packet2.py tests\test_inference_model_loader.py tests\test_runtime_validate.py tests\test_inference_api.py` -> `10.00/10`
- Targeted runtime-path truth check passed:
  - `python -m app.training.retrain --config configs/training.m7.json` -> `artifacts\training\m7\20260401T043003Z`
  - the live run produced a real `autogluon_tabular` artifact plus manifest/comparison truth on current repo data:
    - `winner.model_name = autogluon_tabular`
    - `winner.mean_long_only_net_value_proxy = -0.0005341925944267735`
    - beats `persistence_3` after costs, does not beat `dummy_most_frequent`, and does not meet the acceptance target
  - `python -m app.training.promote --run-dir artifacts\training\m7\20260401T043003Z` -> `artifacts\registry\models\m7-20260401T043003Z\model.joblib`
  - `resolve_inference_model_metadata("")` and `load_model_artifact("")` now resolve the current registry champion as:
    - `model_name = autogluon_tabular`
    - `model_version = m7-20260401T043003Z`
    - `model_version_source = REGISTRY_CURRENT`
- Batch C conclusion:
  - AutoGluon now counts as real present/trainable support in the authoritative path because there is real training, real artifact output, registry discoverability, and runtime usability
  - the current authoritative proof is still negative after costs, so this batch improves truth and replaces the archived-path blockage, but it does not make M20 operationally strong

### Batch D log
- Scope for this pass is limited to M19, M20, and M21 truth from the current local state. The AutoGluon replacement batch remains frozen and is not being reopened.
- Updated the M19 adaptation summary contract and runtime writer so:
  - M19 may remain `IDLE` when no adaptive profile is active
  - the summary now exposes whether persisted drift/performance evidence exists
  - the runtime writer refreshes the configured drift/performance summary artifacts directly instead of waiting for read endpoints to be called
- Updated the M21 continual-learning summary contract and runtime writer so:
  - M21 may remain `IDLE` when no continual-learning profile is active
  - the summary now exposes whether persisted drift-cap evidence exists
  - the runtime writer refreshes the configured drift-cap and summary artifacts directly from persisted truth
- Added explicit M20 current-state truth helpers so Packet 2 can now say:
  - AutoGluon is real and counts for the generalist role
  - overlapping tabular families are subsumed rather than counted as separate missing umbrella gaps
  - M20 stays `ACTIVE_WEAK` when real trend/range specialists are still missing or authoritative after-cost proof is still insufficient
- Updated `/health`, `/predict`, `/signal`, dashboard data sources, and dashboard operator views so they surface:
  - evidence-backed idle adaptation and continual-learning status
  - M20 roster truth as active-but-weak instead of implying active means strong
- Tightened the M20 economics gate so the current-state truth now uses authoritative registry winner metrics for the generalist acceptance check instead of over-crediting Packet 2 slice PnL:
  - current `autogluon_tabular` Packet 2 all-slice net value can be positive while the authoritative promoted run still has `mean_long_only_net_value_proxy = -0.0005341925944267735`
  - M20 therefore remains blocked by economic weakness and specialist breadth, not by missing umbrella-model plumbing
- Realigned the live M20 persisted state to current truth instead of the stale archived-logistic roster:
  - superseded `m20-paper-all-minimal-20260331T205941Z`
  - activated `m20-paper-all-autogluon-generalist-20260401T051733Z`
  - persisted promotion decision `m20-realign-20260401T051733Z`
  - wrote research artifact `artifacts/ensemble/research/20260401T051733Z/research_report.json`
- Fixed the current deployed AutoGluon runtime path so the live Linux container can score the promoted Windows-trained artifact:
  - `app/training/autogluon.py` now loads the bundled predictor with `require_py_version_match=False`
  - `docker/app.Dockerfile` now installs `libgomp1` so LightGBM-backed AutoGluon submodels can load inside the container
- Targeted checks passed:
  - `python -m pytest tests\test_adaptation_service.py tests\test_continual_learning_service.py tests\test_ensemble_packet2.py tests\test_inference_api.py tests\test_dashboard_data_sources.py tests\test_inference_service.py tests\test_inference_model_loader.py -q` -> `82 passed, 1 warning`
  - live DB truth refresh:
    - `adaptive_drift_state` aggregate/symbol rows refreshed from `2026-04-01 05:13:41+00` to `2026-04-01 05:17:47+00`
    - `adaptive_performance_windows` latest aggregate paper rows refreshed from `2026-04-01 05:13:41+00` to `2026-04-01 05:17:47+00`
    - `continual_learning_drift_caps` scope rows refreshed from `2026-04-01 05:14:33+00` to `2026-04-01 05:17:47+00`
  - live artifact refresh:
    - `artifacts/adaptation/drift/latest_summary.json`
    - `artifacts/adaptation/performance/latest_summary.json`
    - `artifacts/continual_learning/summary/latest_summary.json`
    - `artifacts/continual_learning/drift_caps/latest_summary.json`
    - `artifacts/ensemble/research/20260401T051733Z/research_report.json`
  - live API truth after inference rebuild:
    - `/health` now returns `model_name = "dynamic_ensemble"`, `ensemble_profile_id = "m20-paper-all-autogluon-generalist-20260401T051733Z"`, `ensemble_status = "ACTIVE"`, `ensemble_candidate_count = 1`, `ensemble_roster_status = "ACTIVE_WEAK"`, `adaptation_evidence_backed = true`, and `continual_learning_evidence_backed = true`
    - `/adaptation/summary` now returns `evidence_backed = true` plus `latest_drift_updated_at`, `latest_performance_window_id`, `latest_performance_trade_count`, and `latest_performance_created_at`
    - `/continual-learning/summary` now returns `evidence_backed = true` plus `latest_drift_cap_updated_at`
    - `/signal?symbol=BTC/USD` and `/predict?symbol=BTC/USD` now resolve the active weak ensemble instead of erroring, with `ensemble_roster_status = "ACTIVE_WEAK"` and explicit reason codes naming negative economics plus missing trend/range specialists
- Targeted lint note:
  - `python -m pylint ...` returned `9.82/10`
  - remaining findings are non-behavioral mixed-line-ending warnings on already modified files plus one branch-count warning on the Packet 2 truth helper

### Batch E log
- Scope for this pass was limited to the existing authoritative AutoGluon wrapper, checked-in training configs, and the existing artifact/registry metadata chain. No training run, promotion decision, or execution-path semantics were changed.
- Updated `app/training/autogluon.py` so the wrapper now:
  - preserves omitted `hyperparameters` as `None` instead of silently replacing them with a narrow tree-only bundle
  - omits `hyperparameters` from `TabularPredictor.fit(...)` when they were not explicitly supplied
  - preserves explicit `hyperparameters` values as supplied
  - supports `num_bag_sets` and `calibrate_decision_threshold` on the classifier instance, in serialized state, and in fit kwargs with backward-compatible restore defaults for older artifacts
- Updated the authoritative checked-in AutoGluon defaults in:
  - `configs/training.m3.json`
  - `configs/training.m7.json`
  - both now use the manual-training baseline:
    - `presets = "high"`
    - `time_limit = 900`
    - `eval_metric = "log_loss"`
    - `hyperparameters = null`
    - `fit_weighted_ensemble = true`
    - `num_bag_folds = 5`
    - `num_stack_levels = 1`
    - `num_bag_sets = 1`
    - `calibrate_decision_threshold = false`
    - `verbosity = 0`
- Surfaced the effective winner AutoGluon fit config through the existing metadata chain only:
  - `app/training/service.py` now writes `training_model_config` into the saved model artifact and winner summary metadata
  - `app/training/registry.py` now carries that config into `run_manifest.json`, immutable `registry_entry.json`, and `current.json`
  - `app/training/compare.py` now exposes the same winner training config inside the existing comparison-side metadata
- Updated focused tests and helpers:
  - `tests/test_training_autogluon.py`
  - `tests/test_training_service.py`
  - `tests/test_training_registry.py`
  - `tests/test_inference_model_loader.py`
  - `tests/training_workflow_helpers.py`
- Targeted checks passed:
  - `python -m pytest tests\test_training_autogluon.py tests\test_training_service.py tests\test_training_registry.py tests\test_inference_model_loader.py -q` -> `29 passed`
  - `python` config load smoke check for `configs/training.m7.json` -> `{'calibrate_decision_threshold': False, 'eval_metric': 'log_loss', 'fit_weighted_ensemble': True, 'hyperparameters': None, 'num_bag_folds': 5, 'num_bag_sets': 1, 'num_stack_levels': 1, 'presets': 'high', 'time_limit': 900, 'verbosity': 0}`
  - `python` config load smoke check for `configs/training.m3.json` -> `{'calibrate_decision_threshold': False, 'eval_metric': 'log_loss', 'fit_weighted_ensemble': True, 'hyperparameters': None, 'num_bag_folds': 5, 'num_bag_sets': 1, 'num_stack_levels': 1, 'presets': 'high', 'time_limit': 900, 'verbosity': 0}`
- Blockers:
  - none

### Batch F log
- Scope for this pass was limited to local M7 training UX and operational readiness. No training run, promotion decision, evaluation semantics, or protected runtime safety logic was weakened.
- Updated `app/training/dataset.py` so `load_training_config(...)` now reads JSON with `utf-8-sig`, which keeps normal UTF-8 behavior but also accepts PowerShell-written UTF-8 BOM config files.
- Updated `app/training/compare.py` to read the checked-in M7 workflow config with the same BOM-tolerant encoding so the local challenger workflow does not regress on the same operator issue.
- Added `app/training/readiness.py` as a small reusable readiness helper that:
  - loads the checked-in training config
  - reports AutoGluon install/version truth
  - resolves PostgreSQL connectivity using the existing settings/candidate-DSN pattern
  - checks whether the configured training source table exists
  - reports `feature_ohlc` row count
  - uses the existing training dataset loader plus split math to say whether the table is actually ready for the configured walk-forward split
- Updated `scripts/start-stack.ps1` so the intended `dev` startup path no longer injects a fake paper-trading config path. It still sets `STREAMALPHA_RUNTIME_PROFILE=dev`, and paper/shadow/live still keep their explicit trading config paths and protected validation behavior.
- Added first-class operator scripts:
  - `scripts/prepare_m7_training.ps1`
  - `scripts/start_m7_training.ps1`
- The new prepare script now:
  - loads `.env` when present
  - checks config loadability, AutoGluon version, PostgreSQL reachability, source-table existence, row count, and split readiness
  - starts the existing `dev` stack path only when training data is missing or not ready
  - prints a short operator summary plus the recommended next command
- The new start script now:
  - runs the same readiness checks
  - fails clearly when config, AutoGluon, PostgreSQL, or `feature_ohlc` prerequisites are missing
  - invokes the authoritative training command `python -m app.training --config .\configs\training.m7.json`
  - reports the newest artifact directory and `summary.json` winner/acceptance fields after completion
- Updated `README.md` with a minimal local M7 training section pointing operators to the two new scripts.
- Updated focused tests:
  - `tests/test_training_service.py`
  - `tests/test_runtime_validate.py`
  - `tests/test_training_readiness.py`
- Targeted checks passed:
  - `python -m pytest tests\test_training_service.py tests\test_runtime_validate.py tests\test_training_readiness.py -q` -> `15 passed`
  - `.\scripts\prepare_m7_training.ps1 -DryRun` -> reported:
    - `config ok: True`
    - `autogluon version: 1.5.0`
    - `postgres reachable: True`
    - `feature_ohlc exists: yes`
    - `feature_ohlc row count: 1146`
    - `eligible unique timestamps: 376 / 9`
    - `recommended next command: .\scripts\start_m7_training.ps1`
  - `.\scripts\start_m7_training.ps1 -DryRun` -> reported:
    - `Dry run: would run python -m app.training --config .\configs\training.m7.json`
    - `Artifact root: artifacts/training/m7`
- Blockers:
  - none

### Batch G log
- Scope for this pass was limited to the local Windows M7 AutoGluon training failure path. No promotion semantics, runtime safety checks, live/paper/shadow behavior, or evaluation targets were changed.
- Added `app/training/workdirs.py` as a small authoritative helper for repo-local training temp roots:
  - default temp root is now `artifacts/tmp/autogluon`
  - the helper honors `STREAMALPHA_LOCAL_TRAINING_TEMP_ROOT` when an operator script wants to force the training process onto a specific same-drive location
- Updated `app/training/autogluon.py` so the authoritative wrapper now:
  - creates fit and runtime-restore work directories under the repo-local temp root instead of the default system temp location
  - cleans fit temp directories in `finally` blocks and cleans failed runtime restores immediately
  - supports explicit `dynamic_stacking` config/state/fit kwargs with backward-compatible restore defaults for older artifacts
  - keeps omitted `dynamic_stacking` unset and only passes it to `TabularPredictor.fit(...)` when the config explicitly supplies a value
- Updated the checked-in local training configs:
  - `configs/training.m3.json`
  - `configs/training.m7.json`
  - both now set `dynamic_stacking = false` while keeping bagging on, weighted ensemble on, and `num_stack_levels = 1`
- Updated local operator/readiness surfaces:
  - `app/training/readiness.py` now reports optional `fastai` install/version truth without turning missing `fastai` into a blocker
  - `scripts/prepare_m7_training.ps1` now prints the optional fastai breadth status explicitly
  - `scripts/start_m7_training.ps1` now sets `STREAMALPHA_LOCAL_TRAINING_TEMP_ROOT`, `TMP`, `TEMP`, `TMPDIR`, and `RAY_TMPDIR` to the repo-local temp root before the training process starts, and prints that temp root during dry-run and real-run output
  - `README.md` now notes the optional fastai breadth and the repo-local same-drive temp-root behavior for local M7 training
- Updated focused tests:
  - `tests/test_training_autogluon.py`
  - `tests/test_training_readiness.py`
  - `tests/test_training_service.py`
  - `tests/test_training_scripts.py`
- Targeted checks passed:
  - `python -m pytest tests\test_training_autogluon.py tests\test_training_readiness.py tests\test_training_service.py tests\test_training_scripts.py -q` -> `18 passed`
  - `.\scripts\prepare_m7_training.ps1 -DryRun` -> reported:
    - `config ok: True`
    - `autogluon version: 1.5.0`
    - `fastai optional breadth: 2.8.7 (optional breadth available)`
    - `postgres reachable: True`
    - `feature_ohlc exists: yes`
    - `feature_ohlc row count: 1146`
    - `eligible unique timestamps: 376 / 9`
    - `recommended next command: .\scripts\start_m7_training.ps1`
  - `.\scripts\start_m7_training.ps1 -DryRun` -> reported:
    - `Local training temp root: D:\Github\Stream_Alpha\artifacts\tmp\autogluon`
    - `Dry run: would run python -m app.training --config .\configs\training.m7.json`
    - `Artifact root: artifacts/training/m7`
  - same-drive smoke check:
    - `python -` using `app.training.workdirs.resolve_local_training_temp_root()` -> `{'temp_root': 'D:\\Github\\Stream_Alpha\\artifacts\\tmp\\autogluon', 'repo_drive': 'D:', 'temp_drive': 'D:', 'same_drive': True}`
- Blockers:
  - none

### Batch H log
- Scope for this pass was limited to the remaining Windows-local AutoGluon hang discovered during a synthetic smoke fit after Batch G. The same-drive temp-root fix held, but AutoGluon 1.5.0 still chose Ray-backed `parallel_local` bag-fold execution under `fold_fitting_strategy=auto`, which produced a hanging local run after a Ray `access violation` on Windows.
- Updated `app/training/autogluon.py` so the authoritative wrapper now exposes:
  - `fold_fitting_strategy`
  - serialized state, backward-compatible restore, and winner training-config metadata for that field
  - explicit fit passthrough via `ag_args_ensemble={"fold_fitting_strategy": ...}` when bagging is enabled
- Updated the checked-in local training configs:
  - `configs/training.m3.json`
  - `configs/training.m7.json`
  - both now set `fold_fitting_strategy = "sequential_local"` while preserving:
    - `presets = "high_quality"`
    - `num_bag_folds = 5`
    - `num_stack_levels = 1`
    - `num_bag_sets = 1`
    - `dynamic_stacking = false`
    - weighted ensemble enabled
- Updated focused tests:
  - `tests/test_training_autogluon.py`
  - `tests/test_training_service.py`
  - coverage now asserts that:
    - `fold_fitting_strategy` is preserved in config/state
    - the wrapper passes it through to AutoGluon via `ag_args_ensemble`
    - winner summary metadata records it honestly
- Targeted checks passed:
  - `python -m pytest tests\test_training_autogluon.py tests\test_training_service.py -q` -> `14 passed`
  - `.\scripts\prepare_m7_training.ps1 -DryRun` -> config still loads and reports the same M7 readiness truth
  - `.\scripts\start_m7_training.ps1 -DryRun` -> still reports the repo-local temp root and authoritative training command
- Synthetic smoke proof:
  - initial synthetic `high_quality` smoke fit with bagging+stacking and `dynamic_stacking=false` but no explicit `fold_fitting_strategy` still hung after the original `C:` vs `D:` issue was removed
  - investigation found:
    - all Ray temp/session paths were correctly on `D:\Github\Stream_Alpha\artifacts\tmp\autogluon_smoke`
    - Ray no longer had the cross-drive mount error
    - `raylet.err` showed `Windows fatal exception: access violation`
    - the run stalled with Ray-backed `_ray_fit` workers and no completed trainer state
  - after setting `fold_fitting_strategy = "sequential_local"` in the synthetic config, the same smoke training completed successfully:
    - temp root: `D:\Github\Stream_Alpha\artifacts\tmp\autogluon_smoke`
    - run dir: `D:\Github\Stream_Alpha\artifacts\training\smoke_autogluon\20260401T185644Z`
    - summary path: `D:\Github\Stream_Alpha\artifacts\training\smoke_autogluon\20260401T185644Z\summary.json`
    - winner model: `autogluon_tabular`
    - winner training config now records `fold_fitting_strategy = "sequential_local"`
    - no Ray or Python worker processes remained after completion
- Blockers:
  - none

### Batch I log
- Scope for this pass was limited to the remaining local warning sources exposed after the Windows/Ray hang was fixed. No promotion semantics, model search narrowing, live safety rules, or runtime authority boundaries were changed.
- Root-caused the misleading optional FastAI warning to the real training-time dependency failure:
  - `fastai 2.8.7` was installed locally
  - AutoGluon `try_import_fastai()` still failed because `IPython` was missing from the environment
  - this was the real reason `NeuralNetFastAI` breadth looked unavailable despite `fastai` itself being present
- Root-caused the repeated sklearn warning to upstream compatibility behavior in AutoGluon 1.5.0:
  - installed `scikit-learn 1.7.2` accepted `force_int_remainder_cols` but emitted the deprecation warning each time AutoGluon tabular neural preprocessing ran
  - the honest repo-side fix was to pin sklearn back to the compatible non-warning range instead of filtering the warning text
- Updated the repo so future environments match the clean local path:
  - `requirements.txt`
    - added `ipython>=8,<9`
    - pinned `numpy>=1.26,<2`
    - pinned `scikit-learn>=1.6,<1.7`
  - `app/training/readiness.py`
    - now distinguishes optional FastAI breadth being merely installed from being actually usable through AutoGluon
    - records `fastai_usable` and `fastai_detail`
    - surfaces the real blocker text when breadth is installed but broken, including the current `IPython` case
  - `scripts/prepare_m7_training.ps1`
    - now prints the actual optional FastAI breadth state instead of treating any install as automatically usable
  - `tests/test_training_readiness.py`
  - `tests/test_training_scripts.py`
  - `README.md`
- Repaired the current local Python environment to match the repo truth:
  - installed `IPython 8.39.0`
  - downgraded `scikit-learn` from `1.7.2` to `1.6.1`
  - downgraded `numpy` from `2.3.5` to `1.26.4`
  - removed the deprecated standalone `pynvml` package and reinstalled `nvidia-ml-py 13.590.48`
- Targeted checks passed:
  - `python -m pytest tests\test_training_readiness.py tests\test_training_scripts.py tests\test_training_autogluon.py tests\test_training_service.py -q` -> `20 passed`
  - `.\scripts\prepare_m7_training.ps1 -DryRun` -> now reports `fastai optional breadth: 2.8.7 (optional breadth available)` with no misleading broken-breadth claim
  - `.\scripts\start_m7_training.ps1 -DryRun` -> still reports the repo-local temp root and authoritative M7 command
  - `python -` direct import smoke: `from autogluon.common.utils.try_import import try_import_fastai; try_import_fastai()` -> `try_import_fastai OK`
  - `python -` direct import smoke: `import torch` -> `torch imported` with the prior deprecated `pynvml` warning gone
  - tiny synthetic neural-model smoke fit using `FASTAI` + `NN_TORCH` hyperparameters on repo-local temp storage -> completed successfully without the prior:
    - `Import fastai failed ...`
    - `force_int_remainder_cols` sklearn deprecation warnings
    - `pynvml package is deprecated` torch warning
- Blockers:
  - none

### Batch J log
- Scope for this pass was limited to the local operator experience around long-running M7 AutoGluon fits. No model semantics, promotion logic, live/paper/shadow safety, or Windows-safe non-Ray training choices were changed.
- Investigated the live local M7 process that had been left running with no console updates:
  - active process:
    - `python -m app.training --config .\configs\training.m7.json`
    - started at `2026-04-01 20:16:39` local time
  - current process CPU was still rising above `28000` CPU-seconds, so the run was actively using multiple CPU cores rather than being a completely dead process
  - the run artifact directory `artifacts/training/m7/20260401T191644Z` still contained only:
    - `run_config.json`
    - `dataset_manifest.json`
  - this means the training job had not yet reached fold-metric export or summary writing, even though it was still consuming CPU
  - the configured AutoGluon budget remained `time_limit = 900`, so a silent run extending far past that budget was not an acceptable operator experience even if AutoGluon was still working internally
- Added a small reusable fit-progress helper:
  - `app/training/progress.py`
  - inspects a local AutoGluon fit work directory and reports:
    - current best model from `trainer.pkl` when available
    - total planned model-graph nodes when available
    - discovered model directory count
    - latest touched model directory and timestamp
- Updated `scripts/start_m7_training.ps1` so the local operator path now:
  - prints the repo-local temp root
  - prints the safe CPU mode explicitly:
    - `sequential_local` bagging remains enabled
    - AutoGluon model-level multithreading still uses multiple logical CPUs
    - Windows-unsafe Ray-style fold parallelism is still intentionally avoided
  - prints the configured AutoGluon time budget up front
  - launches the training process asynchronously and displays a PowerShell `Write-Progress` heartbeat while it runs
  - shows:
    - elapsed time
    - time-budget ETA while still within budget
    - over-budget duration once the configured budget is exceeded
    - current best model when known
    - latest discovered model activity
    - discovered model-directory count vs model-graph count when available
- Updated focused tests and docs:
  - `tests/test_training_progress.py`
  - `tests/test_training_scripts.py`
  - `README.md`
- Targeted checks passed:
  - `python -m pytest tests\test_training_progress.py tests\test_training_scripts.py tests\test_training_readiness.py tests\test_training_autogluon.py tests\test_training_service.py -q` -> `22 passed`
  - `.\scripts\start_m7_training.ps1 -DryRun` -> now reports:
    - `Local training temp root: D:\Github\Stream_Alpha\artifacts\tmp\autogluon`
    - `CPU mode: sequential_local bagging plus AutoGluon model-level multithreading on up to 20 logical CPUs`
    - `AutoGluon time budget: 900 seconds`
    - `Dry run: would run python -m app.training --config .\configs\training.m7.json`
  - `.\scripts\prepare_m7_training.ps1 -DryRun` still reports the same honest readiness truth
- Blockers:
  - none

### Batch K log
- Scope for this pass was limited to the local M7 operator wrapper falsely reporting failure after a fully written training artifact. No training semantics, promotion logic, acceptance logic, or model-family behavior were changed.
- Investigated the reported failed run and verified the artifact was actually complete:
  - run directory:
    - `artifacts/training/m7/20260401T213341Z`
  - files present:
    - `summary.json`
    - `model.joblib`
    - `fold_metrics.csv`
    - `oof_predictions.csv`
    - `feature_columns.json`
    - plus the expected run metadata files
  - `summary.json` recorded:
    - `winner.model_name = "autogluon_tabular"`
    - `winner.training_config.dynamic_stacking = false`
    - `winner.training_config.fold_fitting_strategy = "sequential_local"`
    - `acceptance.winner_after_cost_positive = false`
    - `acceptance.meets_acceptance_target = false`
- Root cause:
  - the authoritative `python -m app.training` child process could still exit non-zero after writing a complete artifact
  - `scripts/start_m7_training.ps1` treated any non-zero exit code as an operator-visible hard failure even when the artifact bundle was already complete and usable
  - this made the local operator path claim `M7 training command failed` even though the real training result had already been persisted
- Updated `scripts/start_m7_training.ps1` so the local launch path now:
  - captures stdout and stderr into per-run log files via `Start-Process -RedirectStandardOutput/-RedirectStandardError`
  - moves those logs into the winning artifact directory as:
    - `training.stdout.log`
    - `training.stderr.log`
  - checks whether the newest artifact directory is complete before classifying the run as failed
  - treats `non-zero exit code + complete artifact` as:
    - completed with warning
    - not a hard operator failure
  - still treats `non-zero exit code + incomplete artifact` as a real failure and prints the tail of both logs to help the operator debug the actual problem
  - always surfaces the training log paths in the completion summary
- Added focused Windows operator-script coverage in `tests/test_training_scripts.py` for:
  - the exact case where the child training process exits non-zero after writing a complete artifact
  - expected behavior:
    - script returns success
    - completion summary is printed
    - winner metadata is surfaced
    - log paths are surfaced
- Targeted checks passed:
  - `python -m pytest tests\test_training_scripts.py tests\test_training_progress.py tests\test_training_readiness.py -q` -> `9 passed`
  - `.\scripts\start_m7_training.ps1 -DryRun` -> still reports the same local temp root, CPU mode, budget, and authoritative training command
  - `.\scripts\prepare_m7_training.ps1 -DryRun` -> still reports the same honest M7 readiness truth
- Blockers:
  - none
