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
- `scripts/analyze_m20_policy_candidates.py`
- `scripts/train_m20_research_baseline.py`
- `scripts/build_m20_research_feature_matrix.py`
- `scripts/export_m20_training_frame_features.py`
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

## Research-Only M20 Policy Evaluation

To post-process a completed M20 run into simple offline BUY/HOLD threshold
artifacts without changing runtime or promotion behavior, run:

```powershell
python .\scripts\analyze_m20_policy_candidates.py --run-dir .\artifacts\training\m20\<run_id>
```

The helper defaults to the newest completed M20 run when `--run-dir` is
omitted. It reads `summary.json` plus the winner-model rows from
`oof_predictions.csv`, sweeps global probability thresholds, compares each
candidate against the simple `prob_up >= 0.50` baseline threshold policy, and
writes deterministic research outputs under `policy_eval/`:

- `policy_candidates.csv`
- `policy_report.json`
- `policy_report.md`

If regime labels are already present in the saved OOF rows, the report includes
regime-conditional summaries as a read-only research aid. It also reports fold
stability, low-trade/low-coverage honesty flags, and simple higher-cost or
slippage scenarios so abstention-heavy results are interpreted conservatively.
This does not change runtime rosters, promotion semantics, inference behavior,
or execution policy.

The current completed score-only artifact
`artifacts/training/m20/20260427T112021Z` now has a generated `policy_eval/`
report. The honest result is not promotable: every tested threshold in the
default `0.50` to `0.90` grid takes zero trades on the winning PatchTST OOF
rows.

## Research-Only M20 OOF Signal Diagnostics

To explain zero-trade policy results without changing thresholds, labels,
runtime behavior, or promotion state, run:

```powershell
python .\scripts\diagnose_m20_oof_signal.py --run-dir .\artifacts\training\m20\<run_id>
```

The helper writes deterministic artifacts under `policy_eval/diagnostics/`:

- `oof_signal_diagnostics.json`
- `oof_score_quantiles.csv`
- `oof_threshold_crossing_counts.csv`
- `oof_filter_funnel.csv`
- `oof_signal_diagnostics.md`

For `20260427T112021Z`, the diagnostic shows that PatchTST `prob_up` is tightly
clustered around `0.398436`. The policy thresholds `0.50` through `0.90` have
zero crossings before cost, while lower diagnostic thresholds cross before cost
but still produce zero positive after-cost rows under the tested fee/slippage
scenarios. This points the next recovery batch toward calibration/score mapping
analysis or candidate rejection before new label generation.

## Research-Only M20 Specialist Confirmation Adjudication

To adjudicate specialist confirmation from existing saved artifacts only:

```powershell
python .\scripts\write_m20_specialist_confirmation_adjudication.py `
  --confirmation-run-dir .\artifacts\training\m20\20260507T135017Z `
  --original-run-dir .\artifacts\training\m20\20260505T212518Z
```

This helper reads existing `research_labels/vol_scaled/specialist_conditional_usefulness/`
artifacts and writes deterministic research-only outputs under
`research_labels/vol_scaled/specialist_confirmation_adjudication/`:

- `manifest.json`
- `specialist_confirmation_adjudication.json`
- `specialist_confirmation_adjudication.md`
- `candidate_decisions.csv`
- `evidence_metrics.csv`
- `next_actions.csv`

The adjudication is explicitly conservative:

- PatchTST can be retained as a selective rank/top-k/slice research candidate.
- PatchTST is not treated as globally strong, runtime-ready, or promotable.
- NHITS remains secondary/watchlist unless stronger confirmation artifacts appear.
- No runtime inference, registry, promotion, paper/live execution, backtest logic, or profitability claim is changed.

## Research-Only M20 Generic Specialist Edge Evaluator

To evaluate specialist edge across discovered or requested models from existing
saved artifacts only:

```powershell
python .\scripts\analyze_m20_specialist_edge.py `
  --prediction-run-dir .\artifacts\training\m20\20260507T135017Z `
  --label-source-run-dir .\artifacts\training\m20\20260506T054337Z `
  --prediction-source score_only_confirmation
```

The evaluator reads `research_labels/vol_scaled/specialist_predictions/`,
discovers `predictions_{model}_{source}.csv` files unless `--models` is
provided, joins predictions to labels by `symbol` and `interval_begin` with
model-aware label support, and writes `specialist_edge_evaluator/`. Outputs
include `manifest.json`, `specialist_edge_report.json`,
`specialist_edge_report.md`, `model_edge_metrics.csv`,
`topk_policy_metrics.csv`, optional `threshold_policy_metrics.csv`,
`by_symbol.csv`, `by_time.csv`, `candidate_decisions.csv`,
`next_actions.csv`, and `recommendation.json`.

This is one reusable model-selection research tool, not a PatchTST-only policy
module. It does not change training, scoring, runtime inference, registry,
promotion, paper/live execution, trading logic, backtests, or profitability
claims.

## Research-Only M20 Cost-Aware Specialist Policy Evaluator

To evaluate generic specialist policy candidates from existing predictions,
labels, and optional edge-evaluator artifacts only:

```powershell
python .\scripts\analyze_m20_cost_aware_specialist_policy.py `
  --prediction-run-dir .\artifacts\training\m20\20260507T135017Z `
  --label-source-run-dir .\artifacts\training\m20\20260506T054337Z `
  --prediction-source score_only_confirmation
```

The evaluator discovers `predictions_{model}_{source}.csv` files unless
`--models` is provided, joins by `symbol` and `interval_begin` with model-aware
label support, and writes `cost_aware_specialist_policy_evaluator/`. Outputs
include `manifest.json`, `cost_aware_policy_report.json`,
`cost_aware_policy_report.md`, `policy_candidates.csv`,
`model_policy_metrics.csv`, `topk_policy_metrics.csv`, optional
`threshold_policy_metrics.csv`, `by_symbol.csv`, `by_time.csv`,
`economics_availability.json`, `candidate_decisions.csv`, `next_actions.csv`,
and `recommendation.json`.

Economic proxy metrics are computed only from safe net/proxy outcome columns in
label or evaluation artifacts. Binary fee-exceedance labels alone do not become
net PnL. When safe economics are absent, the recommendation is
`ADD_SAFE_NET_PROXY_OR_ECONOMIC_OUTCOME_ARTIFACTS`, and the next required action
is `DESIGN_SAFE_ECONOMIC_OUTCOME_ARTIFACTS_FOR_SPECIALIST_POLICIES`.

## Research-Only M20 Economic Outcome Artifacts

To build safe evaluation-only economic outcome rows for future policy
evaluation:

```powershell
python .\scripts\build_m20_economic_outcomes.py `
  --source-run-dir .\artifacts\training\m20\20260506T054337Z `
  --prediction-run-dir .\artifacts\training\m20\20260507T135017Z
```

The builder writes `research_labels/vol_scaled/economic_outcome_artifacts/`
under the source run. It uses safe label/evaluation artifacts with
`future_return` when available, otherwise it can derive forward returns from a
training-frame price column. Binary labels alone do not create a fake net proxy;
they emit `ECONOMIC_MAGNITUDE_NOT_AVAILABLE`.

Outputs include `manifest.json`, `economic_outcome_report.json`,
`economic_outcome_report.md`, optional `economic_outcomes.csv`,
`schema_audit.csv`, `blockers.csv`, `next_actions.csv`, and
`recommendation.json`. These artifacts are research/evaluation-only and must
stay out of prediction exports, runtime inference, registry, promotion,
trading, and backtests.

After building economic outcome artifacts, re-run the generic policy evaluator
with the safe economic source:

```powershell
python .\scripts\analyze_m20_cost_aware_specialist_policy.py `
  --prediction-run-dir .\artifacts\training\m20\20260507T135017Z `
  --label-source-run-dir .\artifacts\training\m20\20260506T054337Z `
  --prediction-source score_only_confirmation `
  --economic-outcome-dir .\artifacts\training\m20\20260506T054337Z\research_labels\vol_scaled\economic_outcome_artifacts
```

The evaluator joins predictions, labels, and economic outcomes by `symbol`,
`interval_begin`, and `fold_index` when available. It removes
`NET_PROXY_NOT_AVAILABLE` only when safe economic outcome rows are actually
joined. Positive research metrics are not runtime readiness, promotion,
backtest, or profit evidence.

## Research-Only M20 Cost-Aware Policy Adjudication

To freeze the current specialist policy interpretation from existing outputs
only:

```powershell
python .\scripts\write_m20_cost_aware_policy_adjudication.py `
  --prediction-run-dir .\artifacts\training\m20\20260507T135017Z
```

The adjudication reads `cost_aware_specialist_policy_evaluator/` plus optional
specialist edge and confirmation context. It does not recompute predictions,
labels, or economic outcomes. Current evidence keeps NeuralForecast specialist
policies research-only: statistical lift exists, but safe net-proxy economics
are negative. The required next action is
`MOVE_TO_GENERIC_STRATEGY_CONDITIONED_CANDIDATE_FACTORY`, not a PatchTST-only or
NHITS-only rerun.

## Research-Only M20 Trading-Aware Labels

`app.training.research_labels` contains offline helpers for the next recovery
phase:

- triple-barrier event labels from ordered OHLC rows
- fee/slippage exceedance labels
- incumbent meta-labels for whether a BUY signal should have been taken
- deterministic label diagnostics JSON

These helpers are intentionally not connected to runtime inference, model
promotion, registry discovery, live execution, or paper execution.

To generate the artifacts for a completed M20 run:

```powershell
python .\scripts\generate_m20_research_labels.py --run-dir .\artifacts\training\m20\<run_id>
```

The helper writes `research_labels/` artifacts:

- `research_labels_manifest.json`
- `triple_barrier_labels.csv`
- `fee_exceedance_labels.csv`
- `incumbent_meta_labels.csv` when incumbent/default signal columns are present
- `label_diagnostics.json`
- `label_diagnostics.md`
- `label_distribution_by_slice.csv`

For `20260427T112021Z`, PatchTST remains research-negative under the current
next-3-candle direction policy sweep, but the generated target-space diagnostic
shows non-zero research label event space: return-proxy triple-barrier positives
are about `0.192638`, and fee-exceedance positives after costs are about
`0.128820`. The incumbent meta-labels are all zero because the default signal
threshold never fires. These labels are not runtime training labels yet and do
not change runtime inference, registry authority, promotion, paper/live
execution, thresholds, or roster behavior.

## Research-Only M20 Label Readiness

To gate the generated label artifacts before any future training batch:

```powershell
python .\scripts\analyze_m20_label_readiness.py --run-dir .\artifacts\training\m20\<run_id>
```

The helper writes `research_labels/readiness/` artifacts:

- `label_readiness_report.json`
- `label_readiness_report.md`
- `label_readiness_by_slice.csv`
- `tiny_baseline_feasibility.csv`
- `label_readiness_manifest.json`

For `20260427T112021Z`, the report marks triple-barrier and fee-exceedance
targets as coarse research-ready by class balance, but not production/training
ready. Meta-labeling is explicitly blocked because all meta-labels are zero and
candidate-entry count is zero. Fixed-bps fallback is preserved as an honesty
flag, so the recommended next research branch is collecting or adding volatility
features before training. Tiny baselines are feasibility diagnostics only and
are not comparable to the runtime incumbent.

## Research-Only M20 Volatility Audit

To audit volatility sources and optionally generate volatility-scaled research
labels:

```powershell
python .\scripts\audit_m20_volatility_sources.py --run-dir .\artifacts\training\m20\<run_id>
python .\scripts\generate_m20_research_labels.py --run-dir .\artifacts\training\m20\<run_id> --use-volatility
```

The helper writes audit artifacts under `research_labels/volatility_audit/` and
volatility-scaled labels under `research_labels/vol_scaled/` when possible.
Fixed-bps labels are retained for comparison.

For `20260427T112021Z`, volatility-like columns are present in the saved
manifests but not aligned in `oof_predictions.csv`, so the audit computes a
strictly past-looking research volatility proxy from ordered per-symbol returns.
The volatility-scaled labels were generated, but remain not training-ready
because the source is a research proxy rather than an exported feature column.
No model training, runtime inference, registry authority, promotion, paper/live
execution, thresholds, NeuralForecast behavior, or roster behavior changed.

## Research-Only M20 Tiny Baselines

To train tiny feasibility baselines on the volatility-scaled triple-barrier
research labels:

```powershell
python .\scripts\train_m20_research_baseline.py --run-dir .\artifacts\training\m20\<run_id>
```

The helper writes deterministic artifacts under
`research_labels/vol_scaled/baselines/`:

- `baseline_manifest.json`
- `feature_audit.json`
- `feature_audit.md`
- `baseline_metrics.json`
- `baseline_metrics.csv`
- `baseline_confusion_matrix.csv`
- `baseline_by_slice.csv`
- `baseline_report.md`

For `20260427T112021Z`, the feature audit keeps only limited row-aligned
numeric inputs such as PatchTST score outputs and the research volatility proxy.
The best tiny diagnostic is `logistic_regression_tiny` with balanced accuracy
`0.372175`, versus `0.333333` for majority and `0.338663` for fixed-seed
stratified random. Positive and negative recall remain low, so this is only a
feasibility signal for a future cost-aware policy-evaluation batch. It is not a
runtime challenger, not promotable, and not written to the registry.

## Research-Only M20 Feature Matrix

To audit row-aligned safe feature availability for the volatility-scaled
triple-barrier labels:

```powershell
python .\scripts\build_m20_research_feature_matrix.py --run-dir .\artifacts\training\m20\<run_id>
```

The helper writes deterministic artifacts under
`research_labels/vol_scaled/feature_matrix/`:

- `feature_source_audit.json`
- `feature_source_audit.md`
- `feature_candidate_columns.csv`
- `feature_exclusion_reasons.csv`
- `feature_alignment_report.json`
- `feature_alignment_report.md`
- `research_feature_matrix.csv` when safe alignment succeeds
- `research_feature_matrix_manifest.json`
- `feature_matrix_preview.csv`

For `20260427T112021Z`, the audit selects `oof_predictions.csv`, filters it to
the winning PatchTST rows, confirms `symbol + interval_begin` timestamp
alignment, and matches all `236117` volatility-scaled label rows. The resulting
research matrix is still limited: only `y_pred`, `prob_up`, `confidence`, and
`long_trade_taken` survive the leakage screen. Future training-frame feature
exports would be needed before claiming richer row-aligned feature support. This
batch performs no model training and has no runtime, registry, promotion,
paper/live, NeuralForecast, threshold, or roster effect.

## Research-Only M20 Training Frame Export

To attempt to export a row-aligned market-feature frame for the
volatility-scaled research labels:

```powershell
python .\scripts\export_m20_training_frame_features.py --run-dir .\artifacts\training\m20\<run_id>
```

If a real row-level feature frame exists, the helper writes
`research_labels/vol_scaled/training_frame_export/` artifacts:

- `m20_training_frame_features.csv`
- `m20_training_frame_keys.csv`
- `m20_training_frame_feature_columns.json`
- `m20_training_frame_export_manifest.json`
- `m20_training_frame_export_report.json`
- `m20_training_frame_export_report.md`
- `m20_training_frame_preview.csv`

If the frame is missing, it writes:

- `m20_training_frame_export_blocker.json`
- `m20_training_frame_export_blocker.md`
- `m20_required_future_export_schema.json`

For `20260427T112021Z`, the export is blocked. The run preserves configured
OHLC-derived feature names in `feature_columns.json`, `run_config.json`, and
`dataset_manifest.json`, but it does not preserve the row-level training/scoring
feature frame. OOF score columns are explicitly excluded from market features.
The required future schema keeps `symbol + interval_begin` as required keys and
lists the configured prediction-time market features such as OHLC, volume,
lagged returns, rolling return stats, realized volatility, RSI, MACD, and
z-score features. This patch performs no model training and has no runtime,
registry, promotion, paper/live, NeuralForecast, threshold, or roster effect.

Future M20 runs can opt in to the research-only export hook:

```powershell
python -m app.training --config .\configs\training.m20.json --export-training-frame
.\scripts\start_m20_training.ps1 -DryRun -ExportTrainingFrame
```

When enabled, the active run writes `training_frame/` artifacts:

- `m20_training_frame_features.csv`
- `m20_training_frame_keys.csv`
- `m20_training_frame_feature_columns.json`
- `m20_training_frame_export_manifest.json`
- `m20_training_frame_export_report.json`
- `m20_training_frame_export_report.md`
- `m20_training_frame_preview.csv`

The hook preserves deterministic feature order and records a feature-column hash,
schema version, key coverage, symbol/fold/timestamp coverage, duplicate-key
count, missing-value summary, and excluded-column reasons. The flag is off by
default, does not write to the registry, and is not a promotion signal.

The first export-enabled score-only evidence run wrote
`artifacts/training/m20/20260505T200658Z`, but no `training_frame/` artifact was
persisted before the run timed out/interrupted around final score-only
processing. Keep the old artifact blocked and move the hook earlier, or add an
export-only path, before training any market-feature baseline.

The follow-up export-only run wrote
`artifacts/training/m20/20260505T212518Z/training_frame/` before model scoring.
It exported `236153` rows for fold `4` with `22` OHLC-derived market features,
recorded folds `0` through `3` as skipped by the recent-window filter, and
marked model scoring as skipped. This is research evidence only; it is not a
promotion or runtime signal.

The export-only frame can now be used as a label source when OOF/final scoring
artifacts are absent:

```powershell
python .\scripts\generate_m20_research_labels.py --run-dir .\artifacts\training\m20\20260505T212518Z --source training-frame --use-volatility
python .\scripts\analyze_m20_label_readiness.py --run-dir .\artifacts\training\m20\20260505T212518Z
```

This writes `research_labels/vol_scaled/` from row-aligned `realized_vol_12`
and `close_price`, keeps fixed-bps labels only as comparison, and marks
meta-labels not applicable because export-only runs have no incumbent/default
OOF signals. No model training, runtime inference, registry authority,
promotion, paper/live execution, thresholds, NeuralForecast behavior, or roster
behavior changed.

For a tiny research-only fee-exceedance market-feature baseline:

```powershell
python .\scripts\train_m20_fee_exceedance_baseline.py --run-dir .\artifacts\training\m20\20260505T212518Z
```

The helper uses only exported market features, defaults to the `current_fee`
label scenario, applies a chronological 60/20/20 split within the single recent
fold, and writes `research_labels/vol_scaled/fee_exceedance_baselines/`.
Outputs are feasibility diagnostics only and are not runtime-comparable,
promotable, registry-written, or profitability evidence.

For conditional usefulness diagnostics over the emitted baseline predictions:

```powershell
python .\scripts\analyze_m20_conditional_usefulness.py --run-dir .\artifacts\training\m20\20260505T212518Z
```

This writes `research_labels/vol_scaled/conditional_usefulness/` and reports all
eligible slices, including weak slices. The classifications are research-only
enable/disable/watchlist labels for future strategy-ensemble investigation, not
runtime routing, promotion, registry state, or profitability evidence.

To confirm conditional usefulness on the full test split instead of the
deterministic preview sample:

```powershell
python .\scripts\train_m20_fee_exceedance_baseline.py --run-dir .\artifacts\training\m20\20260505T212518Z --export-full-predictions
python .\scripts\analyze_m20_conditional_usefulness.py --run-dir .\artifacts\training\m20\20260505T212518Z --prediction-source full-test
```

This writes full train/validation/test prediction rows for
`logistic_regression_tiny` and writes
`research_labels/vol_scaled/conditional_usefulness_full_test/`, including a
sample-vs-full comparison. The output remains research-only and requires
confirmation on another fold/window before runtime use.

To audit model/member candidates and build a research strategy-ensemble ledger:

```powershell
python .\scripts\audit_m20_model_members.py --run-dir .\artifacts\training\m20\20260505T212518Z --previous-run-dir .\artifacts\training\m20\20260427T112021Z --fitted-models-dir .\artifacts\training\m20\20260405T023104Z\fitted_models
```

The audit treats AutoGluon as a model factory and ensemble manager. If
AutoGluon leaderboard/member metadata is unavailable in the inspected paths, it
records that blocker and writes manual confirmation commands rather than
loading, refitting, promoting, or running a long job.

To prepare the next confirmation window without launching it:

```powershell
python .\scripts\plan_m20_confirmation_window.py --run-dir .\artifacts\training\m20\20260505T212518Z
```

This writes `research_labels/vol_scaled/confirmation_plan/` with target slices,
success rules, expected artifacts, manual commands, and a comparison schema for
future confirmation runs. The current fee-exceedance logistic evidence remains
single recent-fold only; alternate-window confirmation is blocked until a safe
window/fold override or reviewed confirmation config is supplied. No runtime
inference, registry authority, promotion, paper/live execution, thresholds,
NeuralForecast behavior, or roster behavior changes.

The M20 training CLI now supports a manual-only confirmation window override for
score-only/export-only research runs:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_m20_training.ps1 -DryRun -ExportTrainingFrameOnly -ConfirmationWindowStart 2024-04-02T11:30:00Z -ConfirmationWindowEnd 2025-04-02T11:30:00Z -ConfirmationTag confirm_prev_year
```

The dry run prints the exact `python -m app.training ... --export-training-frame-only`
command with `--confirmation-window-start`, `--confirmation-window-end`, and
`--confirmation-tag`, and does not launch the long run. The override is
research-only, records metadata in future run/export manifests when used, and
does not affect default training behavior when the flags are absent.

The manually launched confirmation export
`artifacts/training/m20/20260506T054337Z` was processed through the short
post-export research pipeline. Its confirmation window was
`2024-04-02T11:30:00Z` to `2025-04-02T11:30:00Z` with tag
`confirm_prev_year`, and it exported `312494` market-feature rows with `22`
features across BTC/USD, ETH/USD, and SOL/USD. Fee-exceedance labels were ready
by the coarse gates, while triple-barrier labels remained not ready and
meta-labels remained not applicable. The confirmation `logistic_regression_tiny`
fee-exceedance baseline reached test balanced accuracy `0.611509`,
average precision `0.343911`, ROC-AUC `0.653201`, top-5% precision `0.422215`,
and top-5% lift `1.799698`. Full-test conditional usefulness found `21` enable
and `11` watchlist slices with no disable candidates in this prior-year window.
The original strong slices `momentum=flat`, `range=low`, `symbol=BTC/USD`,
`macd=positive`, and `volume=low` strongly confirmed. The original disable
slices from April 2026 were missing from the prior-year confirmation window, so
they remain unconfirmed coverage gaps rather than failures. This is still
research-only, not runtime-comparable, not promotable, and not profitability
evidence.

The confirmed fee-exceedance evidence can be converted into a research-only
strategy selector design:

```powershell
python .\scripts\design_m20_strategy_selector.py --original-run-dir .\artifacts\training\m20\20260505T212518Z --confirmation-run-dir .\artifacts\training\m20\20260506T054337Z
```

This writes `research_labels/vol_scaled/strategy_selector_design/` for
`fee_exceedance_gate_v0_research`. The selector is an `OPPORTUNITY_GATE`
design only: it proposes future `ALLOW_STRATEGY_SEARCH` versus `HOLD_OR_SKIP`
semantics for research simulation, and it does not decide final LONG/SHORT.
The default design mode is `WEIGHTED_CONFIRMED_SLICES`; disable gaps such as
`month=2026-04` and `quarter=2026Q2` remain explicitly untested. The artifact is
not runtime-enabled, not promotable, not registry-backed, and not profitability
evidence.

The first selector simulation is available at
`research_labels/vol_scaled/strategy_selector_simulation/`:

```powershell
python .\scripts\simulate_m20_strategy_selector.py --original-run-dir .\artifacts\training\m20\20260505T212518Z --confirmation-run-dir .\artifacts\training\m20\20260506T054337Z
```

The result is intentionally conservative. Global logistic top 5% selection
remained the sharper opportunity filter: original precision/lift
`0.348158` / `1.845677`, confirmation precision/lift `0.422215` / `1.799698`.
The weighted selector admitted nearly all rows (`0.973194` original coverage
and `1.000000` confirmation coverage) and collapsed to base-rate precision.
This means the first selector design is too broad and should not advance to
runtime or strategy-family modules as-is. It needs narrower, held-out selector
tuning or another confirmation-window simulation. Disable gaps remain tracked
but unconfirmed.

Rank-gated selector evaluation can be run with:

```powershell
python .\scripts\tune_m20_rank_gated_selector.py --original-run-dir .\artifacts\training\m20\20260505T212518Z --confirmation-run-dir .\artifacts\training\m20\20260506T054337Z
```

This writes `research_labels/vol_scaled/rank_gated_selector/` and compares
global top-k ranking, condition-then-top-k ranking, minimum-condition top-k,
per-condition top-k, and disable-gap-filtered top-k policies. The first real
run keeps the result conservative: rank gating preserves the useful logistic
ranking signal, with `CONDITION_THEN_TOP_1` reaching min lift `1.943` across
the original and confirmation runs, but these thresholds are research choices
only and are not runtime selector logic, not a backtest, not promotable, and not
profitability evidence.

Nested held-out rank-gate tuning can be run with:

```powershell
python .\scripts\tune_m20_rank_gate_nested.py --original-run-dir .\artifacts\training\m20\20260505T212518Z --confirmation-run-dir .\artifacts\training\m20\20260506T054337Z
```

This writes `research_labels/vol_scaled/rank_gate_nested_tuning/`. It tunes only
on the original validation split and locks the selected params for original test
and confirmation test. The first real selection is `CONDITION_THEN_TOP_0.25`
with original test precision/lift `0.347458` / `1.841966` and confirmation
precision/lift `0.512821` / `2.185905`. It remains research-only and does not
change runtime, registry, promotion, paper/live, trading, or threshold behavior.

Manual execution break for the next comparable confirmation window:

Codex must not run this confirmation job. Codex should print these commands,
stop, and wait for Arash to run them manually in a separate PowerShell terminal
and paste back the new run directory and results.

```powershell
python -m app.training --config .\configs\training.m20.json --score-only .\artifacts\training\m20\20260405T023104Z\fitted_models --parquet-dir .\exports\feature_ohlc_for_colab --export-training-frame-only --confirmation-window-start 2023-04-02T11:30:00Z --confirmation-window-end 2024-04-02T11:30:00Z --confirmation-tag confirm_prev_prev_year
```

After Arash provides the new `<CONFIRMATION_RUN_DIR>`, only the short
post-export research pipeline should be run:

```powershell
python .\scripts\generate_m20_research_labels.py --run-dir <CONFIRMATION_RUN_DIR> --source training-frame --use-volatility
python .\scripts\analyze_m20_label_readiness.py --run-dir <CONFIRMATION_RUN_DIR>
python .\scripts\train_m20_fee_exceedance_baseline.py --run-dir <CONFIRMATION_RUN_DIR> --export-full-predictions
python .\scripts\analyze_m20_conditional_usefulness.py --run-dir <CONFIRMATION_RUN_DIR> --prediction-source full-test
python .\scripts\compare_m20_confirmation_slices.py --original-run-dir .\artifacts\training\m20\20260505T212518Z --confirmation-run-dir <CONFIRMATION_RUN_DIR>
python .\scripts\tune_m20_rank_gate_nested.py --original-run-dir .\artifacts\training\m20\20260505T212518Z --confirmation-run-dir <CONFIRMATION_RUN_DIR>
```

Do not promote `CONDITION_THEN_TOP_0.25` to runtime, registry, policy
simulation, trading/backtest, model-retrain, or profit-claim workflows before
that manual confirmation is complete and reviewed.

Manual export result reviewed:

- Run directory: `artifacts/training/m20/20260506T063818Z`
- Window: `2023-04-02T11:30:00Z` to `2024-04-02T11:30:00Z`
- Tag: `confirm_prev_prev_year`
- Exported rows: `306319`
- Feature count: `22`
- Symbol coverage: BTC/USD, ETH/USD, SOL/USD
- Eligible folds: `2`, `3`
- Export complete: `true`

Short post-export research pipeline result for `20260506T063818Z`:

- Label readiness: fee-exceedance READY, triple-barrier NOT_READY.
- Fee baseline best model: `logistic_regression_tiny`
- Test positive rate: `0.231057`
- Balanced accuracy: `0.607586`
- PR-AUC / average precision: `0.338437`
- ROC-AUC: `0.641974`
- Top 5% precision/lift: `0.419197` / `1.814259`
- Conditional usefulness: `61262` rows, `21` enable, `11` watchlist, `0`
  disable candidates.
- Slice comparison recommendation: `CONFIRM_FEE_GATE_FOR_RESEARCH_POLICY`
- Slice comparison counts: `15` strongly confirmed, `10` confirmed, `1`
  inconclusive, `5` missing, `0` not confirmed.
- Locked nested rank gate: `CONDITION_THEN_TOP_0.25`
- Locked confirmation coverage/precision/lift: `0.002497` / `0.522876` /
  `2.262976`
- Locked confirmation recall: `0.005652`
- Locked confirmation selected rows: `153`
- Locked confirmation false positives: `73`
- Locked confirmation average probability: `0.998087`
- Locked confirmation disable-gap exposure: `0`

This remains research-only. It does not add runtime, registry, promotion, policy
simulation, trading/backtest, model-retrain, or profit-claim behavior.

Missing-slice adjudication from existing artifacts only:

- Evidence files:
  - `confirmation_plan/confirmation_comparison/confirmation_slice_comparison.csv`
  - `rank_gate_nested_tuning/disable_gap_exposure.csv`
  - `rank_gate_nested_tuning/confirmation_metrics.csv`
- Missing slices:
  - `month=2025-11`
  - `month=2025-12`
  - `month=2026-04`
  - `quarter=2025Q4`
  - `quarter=2026Q2`
- Adjudication: `CALENDAR_SLICE_NON_OVERLAP`
- Locked gate status: `DISABLE_GAP_NO_SELECTED_EXPOSURE_IN_LOCKED_GATE`
- General blocker preserved:
  `DISABLE_GAP_STILL_UNCONFIRMED_FOR_GENERAL_CONDITIONAL_ANALYSIS`

This means the missing slices should not be counted as model failures for the
locked `CONDITION_THEN_TOP_0.25` research gate. They also should not be treated
as resolved for broad conditional analysis or runtime use.

Rank-gate evidence packet:

- Path: `artifacts/training/m20/20260505T212518Z/research_labels/vol_scaled/rank_gate_evidence_packet/`
- Evidence status: `RESEARCH_CONFIRMED_RANK_GATE`
- Runtime status: `NOT_RUNTIME_READY`
- Promotion status: `NOT_PROMOTABLE`
- Original locked test coverage/precision/lift: `0.002498` / `0.347458` /
  `1.841966`
- Prior-year confirmation coverage/precision/lift: `0.002496` / `0.512821` /
  `2.185905`
- Prev-prev-year confirmation coverage/precision/lift: `0.002497` /
  `0.522876` / `2.262976`
- Disable-gap exposure in locked selected-row evidence: `0`
- Blockers: sparse rank-gate coverage, calendar-slice non-overlap, disable gaps
  unconfirmed for general conditional analysis, no profitability evidence, and
  not runtime-ready.

The packet is documentation/research evidence only and does not authorize
runtime selector behavior, registry writes, promotion, policy simulation,
trading/backtest, model retraining, or profit claims.

Rank-gate economics diagnostics:

```powershell
python .\scripts\simulate_m20_rank_gate_economics.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
```

This writes `research_labels/vol_scaled/rank_gate_economics/` and compares
`CONDITION_THEN_TOP_0.25` against `GLOBAL_TOP_0.25`, `GLOBAL_TOP_1`,
`GLOBAL_TOP_5`, and `NO_GATE` using existing labels and predictions only.

Locked gate economics proxy results:

- Original locked test: selected `118`, coverage `0.002498`, precision
  `0.347458`, lift `1.841966`, net value proxy `-0.180157`.
- Prior-year confirmation: selected `156`, coverage `0.002496`, precision
  `0.512821`, lift `2.185905`, net value proxy `0.189677`.
- Prev-prev-year confirmation: selected `153`, coverage `0.002497`, precision
  `0.522876`, lift `2.262976`, net value proxy `-0.019105`.

The recommendation is `KEEP_RESEARCH_ONLY_RANK_GATE_ECONOMICS_CANDIDATE`.
Because net proxies are mixed and derived from label artifacts, this remains
`NOT_BACKTEST`, `NOT_PROFIT_EVIDENCE`, and `FORWARD_RETURN_PROXY_LIMITED`.

Rank-gate net-proxy diagnostics:

```powershell
python .\scripts\diagnose_m20_rank_gate_net.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
```

This writes `research_labels/vol_scaled/rank_gate_net_diagnostics/` and explains
the mixed net-proxy signs for locked `CONDITION_THEN_TOP_0.25` using existing
selected rows, fee labels, prediction rows, and training-frame features only.

Locked gate net-proxy decomposition:

- Original locked test: selected `118`, true positives `41`, false positives
  `77`, precision `0.347458`, net value proxy `-0.180157`.
- Prior-year confirmation: selected `156`, true positives `80`, false positives
  `76`, precision `0.512821`, net value proxy `0.189677`.
- Prev-prev-year confirmation: selected `153`, true positives `80`, false
  positives `73`, precision `0.522876`, net value proxy `-0.019105`.

The diagnostic records symbol, time, probability-bin, volatility, range, volume,
MACD, momentum, top-negative, and top-positive selected-row tables. It is
`RESEARCH_ONLY`, `NET_PROXY_MIXED`, `NOT_PNL`, `NO_RUNTIME`, `NO_REGISTRY`,
`NO_PROMOTION`, `NO_PROFIT_CLAIM`, and `SPARSE_SELECTION`.

Rank-gate tail/condition concentration analysis:

```powershell
python .\scripts\analyze_m20_rank_gate_tail.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
```

This writes `research_labels/vol_scaled/rank_gate_tail_analysis/` and checks
whether mixed net proxies are driven by a few tail rows, symbol/time
concentration, or condition buckets.

Key result:

- Original locked test: net proxy `-0.180157`; worst-5 selected rows contribute
  `-0.083755`.
- Prior-year confirmation: net proxy `0.189677`; worst-5 selected rows
  contribute `-0.490083`.
- Prev-prev-year confirmation: net proxy `-0.019105`; worst-5 selected rows
  contribute `-0.278726`.
- Negative net windows: `original_locked_test`,
  `prev_prev_year_confirmation`.
- Unstable concentration slices flagged: `9`.

The recommendation is
`REVIEW_TAIL_CONCENTRATION_BEFORE_ANY_POLICY_OR_STRATEGY_STEP`. This is
`TAIL_DIAGNOSTIC_ONLY`, `NOT_PNL`, `NO_PROFIT_CLAIM`, and still has no runtime,
registry, promotion, policy simulation, trading/backtest, or model-retrain
effect.

Rank-gate tail-risk filter simulation:

```powershell
python .\scripts\simulate_m20_rank_gate_tail_filter.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
```

This writes `research_labels/vol_scaled/rank_gate_tail_filter/` and tests
exploratory filters on the locked `CONDITION_THEN_TOP_0.25` selected rows:
exclude high range, exclude high volatility, exclude unstable concentration
slices, exclude negative symbol/time buckets, selected-row probability cutoffs,
and simple combos.

Result:

- Filters evaluated: `10`.
- Recommendation: `NO_STABLE_TAIL_FILTER_FOUND`.
- Strict high-volatility and negative-bucket exclusions collapse at least one
  window to `0` selected rows.
- The best non-empty exploratory filters still have negative net proxy in at
  least one window.

This is labeled `FILTER_SIM_ONLY`, `TAIL_RISK_FILTER_TEST`, `NOT_BACKTEST`,
`NOT_PNL`, `NO_PROFIT_CLAIM`, and does not change runtime, registry,
promotion, policy simulation, trading/backtest, or model-retrain behavior.

M20 decision memo:

```powershell
python .\scripts\write_m20_decision_memo.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
```

This writes `research_labels/vol_scaled/m20_decision_memo/` with a compact
evidence table and next-fork list. Current adjudication:

- Decision: `PAUSE_RANK_GATE_AS_STANDALONE_PATH`.
- Statuses: `RESEARCH_SIGNAL_CONFIRMED`, `ECONOMICS_NOT_STABLE`,
  `NO_STABLE_TAIL_FILTER`, `NOT_RUNTIME_READY`, `NOT_PROMOTABLE`.
- Positive signal is not buried: fee-exceedance logistic ranking remains
  confirmed as research signal.
- Blocking fact: sparse rank-gate economics are unstable and no stable tail
  filter was found.
- Next forks: richer strategy-family modules using the gate as optional filter,
  AutoGluon member prediction export, different horizon/label, or package M20
  as infrastructure-positive but profitability-negative research.

The memo is `RESEARCH_ONLY_DECISION_MEMO`, not a runtime change, registry write,
promotion, backtest, model retrain, long run, or profit claim.

M20 strategy-family scaffold:

```powershell
python .\scripts\design_m20_strategy_families.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
```

This writes `research_labels/vol_scaled/strategy_family_scaffold/` as design-only
research scaffolding. Families:

- `momentum_breakout`
- `range_mean_reversion`
- `volatility_expansion`
- `abstention_hold`

The rank gate is recorded as `OPTIONAL_FILTER_ONLY`; it does not decide
LONG/SHORT or execute anything. The scaffold records required signals, feature
requirements, and next experiments, but it does not add runtime, registry,
promotion, trading/backtest, model-retrain, long-run, or profit-claim behavior.

M20 momentum_breakout research diagnostic:

```powershell
python .\scripts\analyze_m20_momentum_breakout.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
```

This writes `research_labels/vol_scaled/momentum_breakout_research/` and checks
setup frequency, fee-exceedance positive rate, lift versus base, rank-gate
overlap, symbol/time slices, and original/prior/prev-prev stability. Current
diagnostic candidates:

- `realized_vol_high`: lift `1.531055` original, `1.500163` prior-year,
  `1.758930` prev-prev-year.
- `range_high`: lift `1.469422` original, `1.464049` prior-year, `1.682872`
  prev-prev-year.
- `volume_high`: lift `1.394658` original, `1.377996` prior-year, `1.670476`
  prev-prev-year.

This remains `RESEARCH_ONLY`, `DIAGNOSTIC_ONLY`, `NOT_BACKTEST`, and has no
runtime, registry, promotion, trading/backtest, model-retrain, long-run, or
profit-claim effect.

M20 gate+momentum combo diagnostic:

```powershell
python .\scripts\analyze_m20_gate_momentum_combo.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
```

This writes `research_labels/vol_scaled/gate_momentum_combo/` and compares
momentum-only setups, locked `CONDITION_THEN_TOP_0.25`, gate-and-momentum,
gate-or-momentum, and gate-then-momentum top-k policies.

Current result:

- Recommendation: `NO_INCREMENTAL_COMBO_EDGE_OVER_PAUSED_GATE`.
- `GATE_AND_MOMENTUM` and `GATE_THEN_MOMENTUM_TOPK` are equivalent to the
  paused locked gate on selected rows.
- Momentum-only setups have fee-label lift but broad negative net proxies.

This remains `COMBO_DIAGNOSTIC_ONLY`, `NOT_BACKTEST`, `NOT_PNL`, and has no
runtime, registry, promotion, trading/backtest, model-retrain, long-run, or
profit-claim effect.

<!-- M20_RANGE_MEAN_REVERSION_RESEARCH -->
## Research-Only M20 Range Mean-Reversion Diagnostic
Command: python .\scripts\analyze_m20_range_mean_reversion.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
Outputs: manifest.json, report.json, report.md, setup_metrics.csv, by_run.csv, by_symbol.csv, by_time.csv, rank_gate_overlap.csv, recommendation.json.
Diagnostic-only. No runtime, registry, promotion, paper/live execution, trading/backtest, or profitability status change.

<!-- M20_VOLATILITY_EXPANSION_RESEARCH -->
## Research-Only M20 Volatility Expansion Diagnostic
Command: python .\scripts\analyze_m20_volatility_expansion.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
Outputs: manifest.json, report.json, report.md, setup_metrics.csv, by_run.csv, by_symbol.csv, by_time.csv, rank_gate_overlap.csv, recommendation.json.
Current recommendation: KEEP_VOLATILITY_EXPANSION_AS_RESEARCH_DIAGNOSTIC_CANDIDATE.
Strongest stable setup: vol_plus_range_high, with lift 1.610740 original, 1.602734 prior-year, and 1.892837 prev-prev-year.
Other stable setup candidates include realized_vol_high, range_high, volume_high, vol_plus_volume_high, range_plus_volume_high, abs_log_return_high, shock_continuation, and shock_reversal.
Diagnostic-only. No runtime, registry, promotion, paper/live execution, trading/backtest, model-retrain, long-run, PnL, or profitability status change.

<!-- M20_STRATEGY_FAMILY_ADJUDICATION -->
## Research-Only M20 Strategy-Family Adjudication
Command: python .\scripts\adjudicate_m20_strategy_families.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
Outputs: manifest.json, report.json, report.md, family_comparison.csv, setup_comparison.csv, rank_gate_overlap_summary.csv, recommended_next_experiments.csv, recommendation.json.
Current recommendation: TEST_VOLATILITY_EXPANSION_NEXT.
Family roles: volatility_expansion primary with best setup vol_plus_range_high, momentum_breakout secondary with best setup realized_vol_high, and range_mean_reversion watchlist with best setup macd_near_zero.
Existing artifacts only. No runtime, registry, promotion, paper/live execution, trading/backtest, model-retrain, long-run, PnL, or profitability status change.

<!-- M20_VOLATILITY_EXPANSION_DEEP_DIVE -->
## Research-Only M20 Volatility Expansion Deep Dive
Command: python .\scripts\analyze_m20_volatility_expansion_deep_dive.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
Outputs: manifest.json, report.json, report.md, setup_deep_metrics.csv, by_run.csv, by_symbol.csv, by_time.csv, rank_gate_overlap_deep.csv, condition_intersection_matrix.csv, recommendation.json.
Current recommendation: TEST_VOLATILITY_EXPANSION_COMBO_NEXT.
Primary setup: vol_plus_range_high, with lift 1.610740 original, 1.602734 prior-year, and 1.892837 prev-prev-year.
Secondary setups: vol_plus_volume_high, range_plus_volume_high, realized_vol_high, and range_high.
Existing artifacts only. Diagnostic-only. No runtime, registry, promotion, paper/live execution, trading/backtest, model-retrain, long-run, PnL, or profitability status change.

<!-- M20_VOLATILITY_COMBO_ECONOMICS -->
## Research-Only M20 Volatility Combo Economics
Command: python .\scripts\analyze_m20_volatility_combo_economics.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
Outputs: manifest.json, report.json, report.md, policy_metrics.csv, by_run.csv, by_symbol.csv, by_time.csv, tail_summary.csv, stability.csv, recommendation.json.
Current recommendation: TRY_VOLATILITY_AS_OPTIONAL_GATE_FILTER.
Gate-and-volatility variants are equivalent to the paused locked rank gate: min lift 1.841966, average net proxy -0.003195, and two negative-net windows.
Volatility-only setups preserve label lift, but all volatility-only policies have negative average net proxy across the three existing windows, so they are not an escalation path by themselves.
Existing artifacts only. Diagnostic-only. No runtime, registry, promotion, paper/live execution, trading/backtest, model-retrain, long-run, PnL, or profitability status change.

<!-- M20_ABSTENTION_HOLD_RESEARCH -->
## Research-Only M20 Abstention/HOLD Diagnostic
Command: python .\scripts\analyze_m20_abstention_hold.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
Outputs: manifest.json, report.json, report.md, hold_rule_metrics.csv, by_run.csv, by_symbol.csv, by_time.csv, avoided_loss_proxy.csv, missed_positive_proxy.csv, recommendation.json.
Current recommendation: KEEP_ABSTENTION_AS_RESEARCH_FILTER.
Watchlist prediction-time rule: HOLD_BROAD_UNSTABLE_VOLATILITY. It skipped 36 selected rows in the original window, avoided 0.115118 negative net proxy, missed 4 positives, and did not fire in the prior-year or prev-prev-year selected rows.
Oracle-only diagnostics HOLD_SELECTED_NEGATIVE_NET_PROXY and HOLD_SELECTED_BELOW_MEDIAN_NET_PROXY produce clean-looking avoided-loss results, but they use after-the-fact net proxy and are not implementable as runtime rules.
Existing artifacts only. Diagnostic-only. No runtime HOLD logic, registry, promotion, paper/live execution, trading/backtest, model-retrain, long-run, PnL, or profitability status change.

<!-- M20_RESEARCH_PATH_ADJUDICATION -->
## Research-Only M20 Path Adjudication
Command: python .\scripts\write_m20_research_path_adjudication.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
Outputs: manifest.json, research_path_adjudication.json, research_path_adjudication.md, evidence_rollup.csv, path_decisions.csv, next_actions.csv.
Historical decision: STOP_CURRENT_FILTER_CHAIN_AND_PLAN_SPECIALIST_EXPORT.
Historical next action: PLAN_ROW_LEVEL_SPECIALIST_PREDICTION_EXPORT, superseded by later specialist confirmation, economic adjudication, and generic strategy candidate factory work.
The adjudication preserves the confirmed rank-gate signal but records unstable economics, volatility combo instability, weak implementable abstention, and non-implementable oracle HOLD rules. Current filter-chain work is paused; later specialist prediction, economic adjudication, and strategy candidate factory artifacts supersede the old row-level specialist-prediction next action.
Existing artifacts only. Decision/tracking only. No runtime, registry, promotion, paper/live execution, trading/backtest, model-retrain, long-run, PnL, or profitability status change.

<!-- M20_SPECIALIST_PREDICTION_EXPORT_PLAN -->
## Research-Only M20 Specialist Prediction Export Plan
Command: python .\scripts\plan_m20_specialist_prediction_export.py --base-run-dir .\artifacts\training\m20\20260505T212518Z --fitted-models-dir .\artifacts\training\m20\20260405T023104Z\fitted_models --previous-run-dir .\artifacts\training\m20\20260427T112021Z
Outputs: manifest.json, specialist_prediction_export_plan.json, specialist_prediction_export_plan.md, candidate_export_targets.csv, required_prediction_schema.json, manual_export_commands.md, post_export_analysis_commands.md, blockers.csv.
Current recommendation: ADD_LIGHTWEIGHT_PREDICTION_EXPORT_HOOK_FIRST.
The plan identifies 14 NHITS/PatchTST row-level prediction targets. Existing 20260427 OOF specialist rows may be sanitized for conditional analysis, while fitted-model candidates still need a clean per-specialist export path. Blockers: LONG_RUNS_MANUAL_ONLY, PER_SPECIALIST_EXPORT_HOOK_NOT_CONFIRMED, and AUTOGLUON_MEMBER_PREDICTIONS_MISSING.
Planning only. Codex must not launch long prediction exports. No export, score-only rerun, runtime, registry, promotion, paper/live execution, trading/backtest, model-retrain, long-run, PnL, or profitability status change.

<!-- M20_SPECIALIST_PREDICTION_EXPORT -->
## Research-Only M20 Existing Specialist Prediction Export
Command: python .\scripts\export_m20_existing_specialist_predictions.py --base-run-dir .\artifacts\training\m20\20260505T212518Z --previous-run-dir .\artifacts\training\m20\20260427T112021Z
Outputs: manifest.json, report.json, report.md, schema_audit.csv, predictions_neuralforecast_nhits_oof.csv, predictions_neuralforecast_patchtst_oof.csv.
Result: 472,306 sanitized specialist prediction rows from existing OOF artifacts only: 236,153 NHITS rows and 236,153 PatchTST rows. Future/net proxy fields are quarantined from prediction files.
Research-only. No score-only rerun, model retrain, runtime, registry, promotion, paper/live execution, trading/backtest, PnL, or profitability status change.

<!-- M20_SPECIALIST_CONDITIONAL_USEFULNESS -->
## Research-Only M20 Specialist Conditional Usefulness
Command: python .\scripts\analyze_m20_specialist_conditional_usefulness.py --base-run-dir .\artifacts\training\m20\20260505T212518Z --previous-run-dir .\artifacts\training\m20\20260427T112021Z
Outputs: manifest.json, report.json, report.md, model_metrics.csv, by_slice.csv, topk_metrics.csv, comparison.csv, recommendation.json.
Result: 472,234 joined fee-label rows from existing OOF artifacts only after 72 horizon/unlabeled rows were skipped. Best candidate: neuralforecast_patchtst with top-5% lift 1.743825, PR-AUC 0.138238, ROC-AUC 0.476149, and 15 conditional research slices. NHITS top-5% lift is 1.356309 with 2 conditional research slices. The original RUN_SPECIALIST_CONFIRMATION_EXPORT recommendation is historical and has been superseded by score-only confirmation, safe economic evaluation, cost-aware adjudication, and the generic strategy candidate factory.
Research-only. No score-only rerun, model retrain, runtime, registry, promotion, paper/live execution, trading/backtest, PnL, or profitability status change.

<!-- M20_SPECIALIST_CONFIRMATION_PLAN -->
## Research-Only M20 Specialist Confirmation Plan
Command: python .\scripts\plan_m20_specialist_confirmation.py --base-run-dir .\artifacts\training\m20\20260505T212518Z --previous-run-dir .\artifacts\training\m20\20260427T112021Z --fitted-models-dir .\artifacts\training\m20\20260405T023104Z\fitted_models
Outputs: manifest.json, specialist_confirmation_plan.json, specialist_confirmation_plan.md, target_slices.csv, required_export_schema.json, manual_commands.md, post_export_analysis_commands.md, blockers.csv.
Result: primary candidate neuralforecast_patchtst, secondary candidate neuralforecast_nhits, and 17 target slices for confirmation. The original confirmation-export-hook recommendation is historical; confirmation/export/economic-adjudication work has since completed and current M20 research should proceed through generic strategy candidate refinement, not one-off specialist reruns.
Planning only. Codex must not launch long prediction exports. No export, score-only rerun, model retrain, runtime, registry, promotion, paper/live execution, trading/backtest, PnL, or profitability status change.

<!-- M20_SPECIALIST_CONFIRMATION_EXPORT_HOOK -->
## Research-Only M20 Specialist Confirmation Export Hook
The training CLI supports `--export-specialist-predictions-only` for `--score-only` runs. The PowerShell helper supports dry-run planning with `-ScoreOnly`, `-ParquetDir`, and `-ExportSpecialistPredictionsOnly`.
Manual dry-run example: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_m20_training.ps1 -DryRun -ScoreOnly artifacts/training/m20/20260405T023104Z/fitted_models -ParquetDir exports/feature_ohlc_for_colab -ExportSpecialistPredictionsOnly -ConfirmationWindowStart 2024-04-02T11:30:00Z -ConfirmationWindowEnd 2025-04-02T11:30:00Z -ConfirmationTag confirm_prev_year
When Arash manually launches a score-only confirmation run, the hook writes sanitized NHITS/PatchTST row-level prediction files under research_labels/vol_scaled/specialist_predictions/. No long run was launched by Codex. No runtime, registry, promotion, paper/live execution, trading/backtest, model-retrain, PnL, or profitability status change.

<!-- M20_STRATEGY_CANDIDATE_FACTORY -->
## Research-Only M20 Strategy Candidate Factory

Command: python .\scripts\build_m20_strategy_candidates.py --source-run-dir .\artifacts\training\m20\20260506T054337Z

Writes `research_labels/vol_scaled/strategy_candidate_factory/` from existing artifacts only. The reusable factory evaluates MACD momentum, RSI mean-reversion/overextension, range compression, volatility state, return/reversal, and volume context candidates with one generic path. It joins labels and safe economic outcomes by `symbol`, `interval_begin`, and optional `fold_index`, emits candidate rows, metrics, symbol/time slices, feature-family audit rows, decisions, and recommendation artifacts, and preserves `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, `NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, and `NO_PROFIT_CLAIM`.

<!-- M20_STRATEGY_CANDIDATE_REFINEMENT -->
## Research-Only M20 Strategy Candidate Refinement

Command: python .\scripts\analyze_m20_strategy_candidate_refinement.py --source-run-dir .\artifacts\training\m20\20260506T054337Z

Writes `research_labels/vol_scaled/strategy_candidate_refinement/` from existing strategy candidate artifacts only. The analyzer evaluates symbol, month, quarter, volatility-bucket, range-bucket, and volume-bucket refinements generically, then emits refined candidate metrics, slice diagnostics, tail-loss diagnostics, sample-size diagnostics, candidate decisions, next actions, and recommendation artifacts. Current result: 15 candidates, 438 slices, all candidates `REFINE_OR_WATCHLIST_NEGATIVE_ECONOMICS`, recommendation `REFINE_STRATEGY_CANDIDATE_DEFINITIONS`.

<!-- M20_STRATEGY_SLICE_POLICY_EVALUATOR -->
## Research-Only M20 Strategy Slice Policy Evaluator

Command: python .\scripts\analyze_m20_strategy_slice_policy.py --source-run-dir .\artifacts\training\m20\20260506T054337Z

Writes `research_labels/vol_scaled/strategy_slice_policy_evaluator/` from existing strategy refinement artifacts only. The evaluator converts refined symbol, time, regime, and volume slices into generic policy candidates and writes policy metrics, policy candidates, by-symbol/time diagnostics, tail risk, candidate decisions, next actions, and recommendation artifacts. Current result: 438 policy candidates, 434 `SLICE_POLICY_ECONOMICS_NEGATIVE`, 4 `SLICE_POLICY_LOW_SAMPLE`, recommendation `REFINE_STRATEGY_CANDIDATE_DEFINITIONS`.

<!-- M20_STRATEGY_MODEL_FACTORY_PLAN -->
## Research-Only M20 Strategy Model Factory Plan

Command: python .\scripts\plan_m20_strategy_model_factory.py --source-run-dir .\artifacts\training\m20\20260506T054337Z

Writes `research_labels/vol_scaled/strategy_model_factory_plan/` from existing strategy refinement and slice policy artifacts only. The artifact defines the reusable strategy-conditioned model factory contract, required inputs, forbidden leakage inputs, candidate inputs, blockers, and manual command policy. AutoGluon is treated as a model/tournament factory rather than a final model. Current blockers: `NO_APPROVED_STRATEGY_POLICY_CANDIDATE` and `MODEL_FACTORY_EXECUTION_NOT_APPROVED`; no training or scoring command is approved.

<!-- M20_RESEARCH_CANDIDATE_COMPARATOR -->
## Research-Only M20 Candidate Comparator

Command: python .\scripts\compare_m20_research_candidates.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z

Writes `research_labels/vol_scaled/research_candidate_comparator/` from existing strategy candidate, strategy slice policy, model-factory plan, and NeuralForecast specialist adjudication artifacts only. The comparator emits a unified scorecard, economics comparison, stability comparison, candidate decisions, next actions, and recommendation. Current result: 470 compared candidates, 451 `ECONOMICS_NEGATIVE_RESEARCH_ONLY`, 15 `BLOCKED_RESEARCH_ONLY`, 4 `WATCHLIST_LOW_SAMPLE_RESEARCH_ONLY`, recommendation `PAUSE_CURRENT_M20_CANDIDATE_PATHS`.

<!-- M20_RESEARCH_DASHBOARD -->
## Research-Only M20 Dashboard

Command: python .\scripts\write_m20_research_dashboard.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z

Writes `research_labels/vol_scaled/m20_research_dashboard/` from existing evidence artifacts only. The dashboard emits a static JSON/Markdown rollup, evidence index, decision timeline, open blockers, next actions, and recommendation. Current result: overall decision `PAUSE_CURRENT_M20_RESEARCH_PATHS`, recommendation `PAUSE_CURRENT_M20_RESEARCH_PATHS_AND_REFINE_STRATEGY_DEFINITIONS`, blockers `NO_POSITIVE_PROXY_RESEARCH_CANDIDATE` and `NO_RUNTIME_OR_PROMOTION_DECISION`.

<!-- M20_STRATEGY_CANDIDATE_REDESIGN_PLAN -->
## Research-Only M20 Strategy Candidate Redesign Plan

Command: python .\scripts\plan_m20_strategy_candidate_redesign.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z

Writes `research_labels/vol_scaled/strategy_candidate_redesign_plan/` from existing evidence and feature-schema artifacts only. The planner emits available/missing feature family audits, v2 candidate definition specs, a downstream candidate contract, blocked definitions, next actions, and recommendation. Current result: 8 candidate definitions, 6 ready for a future v2 factory, 2 blocked by missing `regime_label` and `adx_14`; recommendation `BUILD_GENERIC_V2_STRATEGY_CANDIDATES`.

<!-- M20_STRATEGY_CANDIDATE_V2_FACTORY -->
## Research-Only M20 Strategy Candidate V2 Factory

Command: python .\scripts\build_m20_strategy_candidates_v2.py --source-run-dir .\artifacts\training\m20\20260506T054337Z

Writes `research_labels/vol_scaled/strategy_candidate_v2_factory/` from redesign-plan definitions and safe training-frame features only. The factory generates ready v2 candidate rows, carries blocked definitions forward, joins labels and economic outcomes only after setup selection, and emits candidate metrics, symbol/time diagnostics, decisions, next actions, and recommendation artifacts. Current result: 8 definitions processed, 6 ready candidates generated 237,794 rows, 2 definitions blocked by missing `regime_label` and `adx_14`, all ready candidates economics-negative, recommendation `REFINE_OR_ADD_SAFE_FEATURES_FOR_V2_CANDIDATES`.

Enriched-source command: python .\scripts\build_m20_strategy_candidates_v2.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --research-feature-dir .\artifacts\training\m20\20260506T054337Z\research_labels\vol_scaled\research_feature_enrichment

With `--research-feature-dir`, the factory uses the separate enrichment artifact instead of mutating or rereading new features into `training_frame/`. Current enriched result: source mode `research_feature_enrichment`, 8 definitions ready, `regime_conditioned_momentum` and `trend_strength_filtered_momentum` newly unblocked, 387,036 candidate rows, all 8 candidates `V2_STRATEGY_CANDIDATE_ECONOMICS_NEGATIVE`, recommendation `REFINE_OR_ADD_SAFE_FEATURES_FOR_V2_CANDIDATES`.

<!-- M20_SAFE_FEATURE_AVAILABILITY -->
## Research-Only M20 Safe Feature Availability Audit

Command: python .\scripts\audit_m20_safe_feature_availability.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --regime-thresholds-path .\artifacts\regime\m8\20260320T165813Z\thresholds.json

Writes `research_labels/vol_scaled/safe_feature_availability/` from existing feature-frame metadata and M8 threshold artifacts only. The audit records feature sources, leakage-risk checks, blocked features, and recommendation artifacts without computing or appending new features. Current result: `regime_label` is safe-computable from fixed M8 thresholds and existing `realized_vol_12`, `momentum_3`, `macd_line_12_26`; `adx_14` is safe-computable later from causal per-symbol OHLC history; recommendation `BUILD_M20_RESEARCH_FEATURE_ENRICHMENT_ARTIFACT`.

<!-- M20_RESEARCH_FEATURE_ENRICHMENT -->
## Research-Only M20 Feature Enrichment

Command: python .\scripts\build_m20_research_feature_enrichment.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --regime-thresholds-path .\artifacts\regime\m8\20260320T165813Z\thresholds.json

Writes `research_labels/vol_scaled/research_feature_enrichment/` as a separate research-only frame. The builder keeps original `training_frame/` files unchanged, adds `regime_label` using fixed M8 threshold provenance, computes causal per-symbol `adx_14` from OHLC history with warmup rows blank, and records lineage/leakage audits. Current result: 312,494 rows, features added `regime_label` and `adx_14`, blocked features 0, recommendation `RE_RUN_V2_STRATEGY_CANDIDATE_FACTORY_WITH_RESEARCH_FEATURES`.

<!-- M20_V2_REFINEMENT_RECOVERY -->
## Research-Only M20 V2 Refinement Recovery

Commands:
python .\scripts\plan_m20_v2_refinement.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
python .\scripts\build_m20_strategy_candidate_v2_refined_definitions.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
python .\scripts\build_m20_strategy_candidates_v2.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --research-feature-dir .\artifacts\training\m20\20260506T054337Z\research_labels\vol_scaled\research_feature_enrichment --redesign-plan-dir .\artifacts\training\m20\20260506T054337Z\research_labels\vol_scaled\strategy_candidate_v2_refined_definitions --output-name strategy_candidate_v2_refined_factory
python .\scripts\write_m20_reframe.py --source-run-dir .\artifacts\training\m20\20260506T054337Z

Writes a generic v2 refinement plan, predicate-spec refined definitions, refined candidate factory outputs, and a static M20 reframe artifact. Current result: 4 refined definitions, 118,116 candidate rows, all 4 candidates `V2_STRATEGY_CANDIDATE_ECONOMICS_NEGATIVE`; M20 remains `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, `NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, and `NO_PROFIT_CLAIM`. The reframe records M20 as context-aware decision selection and recommends `DESIGN_RESEARCH_ONLY_DECISION_POLICY_EVALUATOR`.

<!-- M20_DECISION_POLICY_RECOVERY -->
## Research-Only M20 Decision-Policy Recovery

Commands:
python .\scripts\audit_m20_policy_inputs.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z
python .\scripts\evaluate_m20_decision_policies.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z
python .\scripts\audit_m20_policy_validation.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
python .\scripts\build_m20_trading_aware_labels.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
python .\scripts\evaluate_m20_decision_policies.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z --trading-aware-label-dir .\artifacts\training\m20\20260506T054337Z\research_labels\vol_scaled\trading_aware_labels --output-name trading_aware_policy_eval
python .\scripts\plan_m20_shadow_observer.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --policy-eval-dir .\artifacts\training\m20\20260506T054337Z\research_labels\vol_scaled\trading_aware_policy_eval

The recovery path first audits policy-input availability, then evaluates bounded generic TAKE/HOLD policies from existing OOF predictions, refined candidate events, and safe economic outcomes. It adds validation/search-breadth audit artifacts, builds research-only trading-aware labels for calibration diagnostics, reruns the same evaluator with those labels, and writes a shadow-observer plan artifact. Current result: 70 generic policies evaluated before and after trading-aware labels; no adequate positive proxy policy; shadow observer recommendation `PAUSE_M20_POLICY_ROUTE_AND_REDESIGN_INPUTS`.

These tools do not train, score, change runtime behavior, write registry or promotion state, change trading/backtest logic, or make profit claims. Policy selection blocks future/net/economic outcome columns as inputs; those columns are outcome diagnostics only.

<!-- M20_INPUT_REDESIGN_RECOVERY -->
## Research-Only M20 Input Redesign Recovery

Commands:
python .\scripts\analyze_m20_input_failures.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
python .\scripts\build_m20_research_input_catalogue.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z
python .\scripts\plan_m20_input_redesign.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
python .\scripts\build_m20_redesigned_research_inputs.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
python .\scripts\evaluate_m20_decision_policies.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z --research-input-dir .\artifacts\training\m20\20260506T054337Z\research_labels\vol_scaled\m20_redesigned_research_inputs --label-column fee_plus_slippage_exceedance_6 --output-name m20_redesigned_policy_eval
python .\scripts\write_m20_input_redesign_decision.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --policy-eval-dir .\artifacts\training\m20\20260506T054337Z\research_labels\vol_scaled\m20_redesigned_policy_eval

This path explains why the candidate/policy routes failed, catalogues available and missing research inputs, recovers safe 6/12-candle research labels from enriched OHLCV rows, and reruns the generic policy evaluator using those redesigned labels for diagnostics. Current result: `fee_plus_slippage_exceedance_6` and `fee_plus_slippage_exceedance_12` are available, but the policy rerun remains economics-negative. Final decision: `M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY`.

The recovered labels are research outcomes only. They are not runtime inputs, not training-frame mutations, not backtest evidence, not promotion evidence, and not profit claims.

<!-- M20_FINAL_RESEARCH_SUMMARY -->
## Final M20 Negative Research Result

Command:
python .\scripts\write_m20_final_research_summary.py --source-run-dir .\artifacts\training\m20\20260506T054337Z

Writes `research_labels/vol_scaled/m20_final_research_summary/` as the terminal M20 negative-result artifact. It rolls up the specialist route, enriched/refined candidate routes, policy route, trading-aware label route, and input-redesign route. Current terminal decision: `M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY`; status `RESEARCH_ONLY_NEGATIVE_RESULT`; recommendation `KEEP_M20_PAUSED_AS_NEGATIVE_RESULT_AND_MOVE_TO_PLATFORM_MATURITY`.

This freezes the current M20 route as negative research evidence. Future work should move to platform maturity or data-upgrade planning, not more one-off M20 candidate, policy, threshold, label, or input tweaks. No runtime, registry, promotion, trading, backtest, or profit claim exists.

<!-- M9_REGIME_INTEGRATION_AUDIT -->
## M9 Regime Integration Audit

Command:
python .\scripts\audit_m9_regime_integration.py

Writes `artifacts/platform_maturity/m9/regime_integration_audit/` as the
audit-first platform-maturity artifact. It records M8 threshold availability,
the checked-in M9 signal policy, `/regime`, `/signal`, freshness, decision-trace,
risk-schema, and M20-pause documentation surfaces.

Current result: `M9_REGIME_INTEGRATION_CONSOLIDATED`; gap count `0`;
recommendation `PROCEED_TO_M10_RISK_INTERFACE_AUDIT`; next required action
`AUDIT_M10_RISK_INTERFACE_WITH_REGIME_CONTEXT`.

The audit adds a canonical `RegimeContext` contract only. It does not reopen M20,
does not change runtime behavior, and creates no promotion, backtest, trading, or
profit claim.
