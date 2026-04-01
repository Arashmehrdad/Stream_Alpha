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
