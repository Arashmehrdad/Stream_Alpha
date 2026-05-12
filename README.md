# Stream Alpha

Stream Alpha is a local-first crypto trading research and operator stack built around Docker Compose, Redpanda Community Edition, PostgreSQL, FastAPI, Streamlit, and Python services.

The project started as a market-data and feature pipeline, then grew into a bounded trading system with explicit authority layers for prediction, risk, execution, reliability, explainability, adaptation, ensemble behavior, and guarded continual learning.

This README is a cleaned canonical guide for the accepted **Second Foundation v2** baseline, with honest status notes for the currently implemented M19-M21 surfaces.

## Current Status

Accepted milestones and current higher-foundation status:

* M1 through M7
* M8 foundation
* M9 regime-aware live signal foundation
* M10 risk engine foundation
* M11 execution abstraction and shadow execution foundation
* M12 guarded live maturity pass
* M13 reliability and recovery foundation
* M14 explainability and decision trace foundation
* M15 operator console foundation
* M16 deployment and environment foundation
* M17 operational alerting foundation
* M18 usability and strategy evaluation foundation
* M19 bounded adaptation now has runtime-generated persisted drift/performance truth, without invented active adaptive profiles or promotions beyond real persisted state
* M20 dynamic ensemble surfaces and persisted profile truth exist, and the current active roster is now aligned to the real registry-backed AutoGluon generalist only, but the candidate ecosystem remains narrow and not economically accepted yet
* M21 guarded continual learning now has runtime-generated persisted drift-cap truth, without invented active profiles, experiments, or promotions beyond real persisted state

Second Foundation v2 is not honestly complete through M21 yet. The repo now exposes truthful M19-M21 runtime surfaces without overclaiming stronger model diversity, profile activity, or promotion evidence than currently exists.

## What Stream Alpha Is

Stream Alpha is:

* local-first
* inspectable
* explicit about control boundaries
* honest about blocked states and degraded states
* designed for paper, shadow, and guarded live operation

Stream Alpha is not:

* a black-box autonomous trading bot
* an RL system
* a news or sentiment engine
* an unrestricted online learning system
* a cloud-native orchestration platform

## Core Architecture

Canonical runtime path:

`Feed -> Ingestion Health + Replay Buffer -> Features -> Freshness Guards -> Regime Engine -> Model Stack -> Calibration Layer -> Risk Engine -> Signal Engine -> Execution Layer -> Order State Store / Audit Logs / Metrics Store / Drift Monitor / Incident Timeline -> Operator Console / Alerts / Daily Summaries / Promotion and Evaluation Registry -> M18 Evaluation Surfaces -> M19 Adaptive Surfaces -> M20 Dynamic Ensemble Surfaces -> M21 Continual Learning Surfaces`

Core execution modes:

* `backtest`
* `paper`
* `shadow`
* `live`

The current mode should remain visible in the API, dashboard, logs, artifacts, and workflow outputs.

## Authority Boundaries

These boundaries are part of the accepted design and should be treated as stable until a later foundation explicitly changes them.

* **M4** remains authoritative for prediction and signal generation.
* **M10** remains authoritative for risk approval, modification, and blocking.
* **M11** remains authoritative for execution abstraction and order lifecycle truth.
* **M12** remains authoritative for guarded live controls.
* **M13** remains authoritative for reliability, freshness, and recovery truth.
* **M14** remains authoritative for explainability and canonical decision traces.
* **M15** remains authoritative for operator-console trust and usability.
* **M16** remains authoritative for deployment profiles, startup validation, and reproducible stack startup.
* **M17** remains authoritative for operational alerting, incident timeline, startup safety, and daily operations summaries.
* **M18** remains authoritative for usability and strategy evaluation, divergence reporting, promotion evidence, and paper-to-live degradation truth.
* **M19** remains authoritative for bounded adaptation, recalibration, drift truth, and explicit rollback of adaptive state.
* **M20** remains authoritative for dynamic ensemble behavior, regime-aware weighting, agreement truth, and controlled ensemble promotion and rollback.
* **M21** adds guarded continual-learning workflow on top of those accepted boundaries without taking authority away from them.

## Milestone Summary

### M1 to M8: foundational data and offline intelligence

* **M1** ingests Kraken public WebSocket v2 market data and writes normalized raw rows into PostgreSQL through Redpanda.
* **M2** finalizes OHLC candles safely, computes features from finalized candles only, and writes canonical rows into `feature_ohlc`.
* **M3** builds the offline training pipeline, evaluates baselines and learned models with purged expanding walk-forward validation, and saves training artifacts.
* **M4** serves the accepted inference API from saved artifacts and canonical feature rows.
* **M5** adds the minimum correct paper-trading engine using authoritative M4 signals.
* **M6** adds the first read-only Streamlit operator dashboard.
* **M7** adds local-first challenger retraining, promotion, registry, and rollback without automatic retraining loops.
* **M8** adds the offline regime workflow and deterministic regime thresholds.

### M9 to M12: runtime decision maturity

* **M9** adds regime-aware live signal logic on top of M4. The audit-first
  consolidation path now writes a deterministic regime integration audit before
  any further platform-maturity work.
* **M10** adds the authoritative pre-trade risk and sizing layer.
* **M11** adds execution abstraction and explicit order lifecycle truth for `paper` and `shadow`.
* **M12** adds guarded Alpaca live execution with explicit runtime arming, whitelisting, tiny notional limits, reconciliation truth, and fail-closed behavior.

### M13 to M17: operational trust and operator visibility

* **M13** adds reliability, freshness, lag detection, breaker state, recovery truth, and canonical system health aggregation.
* **M14** adds explainability, canonical decision traces, rationale artifacts, and explicit linkage across risk, execution, ledger, and position rows.
* **M15** upgrades the dashboard into a proper operator console with market, signals, trades, risk, health, models, and incidents views.
* **M16** adds deployment profiles, startup validation, runtime-profile selection, and PowerShell helpers.
* **M17** adds operational alerting, incident timeline, startup safety, and daily operations summaries.

### M18 to M21: evaluation, bounded intelligence, and guarded learning

* **M18** adds usability and strategy evaluation truth, divergence reporting, and realistic promotion evidence.
* **M19** now writes runtime-generated persisted drift/performance truth. It should be read as bounded adaptation evidence, not as proof of stronger adaptive profile or promotion activity than the real persisted state supports.
* **M20** is paused as an honest negative research result. The terminal decision
  is `M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY`; it has no runtime, registry,
  promotion, trading, backtest, or profit claim authority.
* **M21** now writes runtime-generated persisted drift-cap truth. Guarded workflow/read surfaces exist, but the repo does not invent active continual-learning profiles, experiments, or promotions beyond real persisted truth.

## Model stack after AutoGluon batch

### Present and trained now

* `autogluon_tabular`
* real promoted AutoGluon artifacts now exist and load through the current registry-backed runtime path
* simple baselines may still exist for evaluation and comparison, but they are not the intended primary production model stack

### Legacy historical artifacts still present for backward inspection

* `logistic_regression`
* `hist_gradient_boosting`

### Subsumed indirectly through AutoGluon

* AutoGluon-managed sublearners may exist inside `autogluon_tabular` artifacts, but they do not count as separately registry-addressable model families today

AutoGluon now receives credit in this repo because there is proven end-to-end training, promotion, registry, and runtime loading for real `autogluon_tabular` artifacts. That credit does not imply economic acceptance or a broad specialist roster yet.

### Still genuinely missing for M20 specialist purposes

* real `TREND_SPECIALIST` candidate families with committed trained artifacts and runtime-usable registry truth
* real `RANGE_SPECIALIST` candidate families with committed trained artifacts and runtime-usable registry truth
* enough specialist breadth to claim meaningful regime-slice value rather than a minimal active roster

### Referenced in design but not actually trained / registry-backed / runtime-usable

* NeuralForecast NHITS
* NeuralForecast NBEATSx
* NeuralForecast TFT
* NeuralForecast PatchTST
* River
* other advanced specialist families mentioned in design or research targeting hooks without real committed trained artifacts and runtime-usable registry truth

### Not counted separately even though AutoGluon is now real

* XGBoost
* LightGBM
* CatBoost
* tabular MLP-like learners managed under a real AutoGluon Tabular workflow

### Impact on M19

* M19 now has real runtime-generated persisted drift/performance truth.
* M19 should still be judged as bounded and evidence-led rather than strong autonomous adaptation, because the upstream trained model pool remains limited and no stronger adaptive profile/promotion state has been invented.

### Impact on M20

* M20 should now be judged from real candidate breadth, role diversity, regime-slice value, and economics after costs, not from missing umbrella-model plumbing.
* One real registry-backed, runtime-usable AutoGluon family is now present and does provide honest generalist candidate truth.
* The active runtime roster is now aligned to that current generalist only, rather than the stale archived-logistic roster that existed before this pass.
* That does not yet create real trend or range specialists, and the current promoted AutoGluon proof remains negative after costs, so M20 should still not be described as richly specialist, strongly diverse, or operationally strong.

### Impact on M21

* M21 now has real runtime-generated persisted drift-cap truth.
* M21 should not be described as strong continual learning yet, because guarded persisted truth exists without a broad, proven, runtime-usable specialist candidate ecosystem or invented promotion activity.

## High-Level Repository Layout

This is the important structure, not a brittle full tree dump.

```text
app/
  adaptation/
  common/
  continual_learning/
  ensemble/
  explainability/
  features/
  inference/
  ingestion/
  regime/
  reliability/
  trading/
  training/

configs/
  adaptation.yaml
  continual_learning.yaml
  explainability.yaml
  paper_trading.yaml
  reliability.yaml
  regime.m8.json
  regime_signal_policy.json
  training.m3.json
  training.m7.json

artifacts/
  adaptation/
  continual_learning/
  explainability/
  live/
  operations/
  paper_trading/
  rationale/
  registry/
  reliability/
  regime/
  training/

apps and entry surfaces:
- app.inference
- app.features.main
- app.ingestion.main
- app.regime
- app.training
- dashboards/streamlit_app.py
- scripts/run_paper_trader.py
```

Generated local artifacts such as training checkpoints and Colab parquet exports
are intentionally not source-controlled. Keep `checkpoints/`, `exports/`,
`Datasets/`, and runtime `artifacts/` local unless a later batch promotes a
specific artifact through the registry path with explicit proof.

## Documentation

The project documentation lives in [docs/index.md](docs/index.md).

Useful entry points:

* [Getting started](docs/getting-started.md)
* [Architecture](docs/architecture.md)
* [Docker](docs/docker.md)
* [Runtime vs training](docs/runtime-vs-training.md)
* [Operations runbook](docs/operations-runbook.md)
* [Troubleshooting](docs/troubleshooting.md)

Optional local docs build:

```powershell
python -m pip install -r requirements-docs.txt
python -m mkdocs build
python -m mkdocs serve
```

Dependency entrypoints are split by purpose:

* `requirements-runtime.txt` is the Docker service/runtime dependency set and
  keeps the smallest proven AutoGluon runtime set for the current registry
  champion.
* `requirements-training.txt` layers research/training dependencies on top of
  runtime dependencies.
* `requirements.txt` remains the compatibility superset for local development.

The shared Docker app image uses the runtime dependency file and a BuildKit pip
cache mount. If image/cache growth gets noisy locally, stop the stack before
cleanup and keep named volumes intact:

```powershell
docker compose --profile paper --env-file .env down --remove-orphans
docker builder prune -af --filter "until=24h"
docker image prune -af
docker compose --profile paper --env-file .env up -d
```

Do not run `docker system prune --volumes` unless you intentionally want to
delete local Postgres and Redpanda data.

## Core Services

* `redpanda`: local event broker
* `redpanda-console`: topic and cluster inspection UI
* `postgres`: canonical local datastore
* `producer`: Kraken public market-data ingestion service
* `features`: finalized-candle feature builder
* `inference`: FastAPI service for health, predict, signal, freshness, reliability, adaptation, ensemble, and continual-learning read surfaces
* `paper-trader`: execution runner for paper, shadow, and guarded live operation
* `dashboard`: Streamlit operator console

## Deployment and Runtime Profiles

M16 introduced one-command local startup helpers and runtime profiles.

Profiles:

* `dev`
* `paper`
* `shadow`
* `live`

Truthful startup scope:

* blank-clone `dev` works after copying `.env.example` to `.env`
* `paper`, `shadow`, and `live` require local model and regime artifacts, or a registry-backed champion
* `live` additionally requires `.env.secrets`, explicit runtime arming, and guarded startup checks

Key scripts:

* `./scripts/start-stack.ps1 -Profile dev`
* `./scripts/start-stack.ps1 -Profile paper`
* `./scripts/start-stack.ps1 -Profile shadow`
* `./scripts/start-stack.ps1 -Profile live`

## Local M7 Training

## Local M20 Specialist Status

Use the read-only status helper to inspect the newest M20 training or score-only
artifact without changing model behavior, promotion state, or runtime rosters:

* `./scripts/status_m20_training.ps1`

The status helper prints the latest artifact directory, execution mode, progress
state, incumbent version, specialist verdicts, and any obvious blocker such as a
missing `summary.json`.

For research-only post-processing of a completed M20 run into simple BUY/HOLD
threshold candidates on the saved winner-model OOF rows, use:

* `python .\scripts\analyze_m20_policy_candidates.py --run-dir .\artifacts\training\m20\<run_id>`

That helper defaults to the newest completed M20 artifact when `--run-dir` is
omitted, sweeps global probability thresholds against the saved winner-model
OOF predictions, compares each candidate against the simple `prob_up >= 0.50`
baseline policy, and writes deterministic `policy_eval/` artifacts into the run
directory:

* `policy_candidates.csv`
* `policy_report.json`
* `policy_report.md`

If regime labels are already present in `oof_predictions.csv`, the saved report
also includes regime-conditional summaries. The report flags zero-trade or
low-coverage candidates, fold concentration, and simple higher-cost/slippage
scenarios so threshold "wins" are not mistaken for promotion evidence.

The current completed score-only artifact
`artifacts/training/m20/20260427T112021Z` can be post-processed by this helper,
but the saved policy report remains non-promotable research truth: the tested
threshold family takes zero trades on the winning PatchTST OOF rows.

To diagnose that zero-trade result without changing thresholds, labels, runtime
behavior, or promotion state, run:

* `python .\scripts\diagnose_m20_oof_signal.py --run-dir .\artifacts\training\m20\<run_id>`

The diagnostic writes `policy_eval/diagnostics/` artifacts showing the OOF
filter funnel, score quantiles, threshold crossing counts, cost-scenario counts,
label balance, and honesty flags. On `20260427T112021Z`, the PatchTST `prob_up`
scores are tightly clustered around `0.398436`, so thresholds `0.50` through
`0.90` have zero crossings before cost; lower diagnostic thresholds cross before
cost but still have zero positive after-cost rows under the tested fee/slippage
scenarios. That points the next recovery batch toward calibration/score mapping
analysis or candidate rejection before any label-generation work.

The recovery branch also includes research-only trading-aware label helpers for
triple-barrier events, fee/slippage exceedance, incumbent meta-labels, and label
diagnostics. They are offline utilities only and are not wired into runtime
inference, registry promotion, or live/paper execution.

To generate deterministic research label artifacts for a completed M20 run, use:

* `python .\scripts\generate_m20_research_labels.py --run-dir .\artifacts\training\m20\<run_id>`

For `20260427T112021Z`, this writes `research_labels/` artifacts for
return-proxy triple-barrier labels, fee-exceedance labels, incumbent meta-labels,
label diagnostics, and slice distributions. The real artifact lacks OHLC price
path and volatility columns, so the batch uses the saved `future_return_3`
column with a fixed-bps fallback and marks the labels `LABELS_NOT_TRAINING_READY`.
This is a target-space diagnostic only; it does not train a model or make M20
promotable.

To gate those labels before any training batch, run:

* `python .\scripts\analyze_m20_label_readiness.py --run-dir .\artifacts\training\m20\<run_id>`

For `20260427T112021Z`, the readiness report marks triple-barrier and
fee-exceedance targets as coarse research-ready, blocks meta-labeling because all
meta-labels are zero and no default entry signal fires, and keeps the fixed-bps
labels research-only. Tiny baselines are emitted only as feasibility diagnostics
and are not comparable to the runtime incumbent.

To audit volatility sources and, where possible, generate volatility-scaled
research labels:

* `python .\scripts\audit_m20_volatility_sources.py --run-dir .\artifacts\training\m20\<run_id>`
* `python .\scripts\generate_m20_research_labels.py --run-dir .\artifacts\training\m20\<run_id> --use-volatility`

For `20260427T112021Z`, volatility columns such as `realized_vol_12` are present
in the saved feature manifests but not row-aligned in `oof_predictions.csv`, so
the audit computes a past-looking research volatility proxy from ordered returns.
Fixed-bps labels are retained for comparison. This remains research-only and
does not train models or alter runtime behavior.

To train only tiny research baseline diagnostics on the volatility-scaled
triple-barrier labels, run:

* `python .\scripts\train_m20_research_baseline.py --run-dir .\artifacts\training\m20\<run_id>`

For `20260427T112021Z`, this writes `research_labels/vol_scaled/baselines/`
artifacts. The best tiny diagnostic baseline is `logistic_regression_tiny`
with balanced accuracy `0.372175`, above majority/random diagnostics but still
low on positive and negative recall. The labels come from a no-lookahead
research volatility proxy because manifest volatility columns are not
row-aligned in OOF predictions. These baselines are not runtime-comparable, not
promotable, and are not written to the registry.

To audit available row-aligned research features for those labels, run:

* `python .\scripts\build_m20_research_feature_matrix.py --run-dir .\artifacts\training\m20\<run_id>`

For `20260427T112021Z`, the audit selects `oof_predictions.csv`, filters it to
the winning PatchTST rows, confirms `symbol + interval_begin` alignment, and
writes `research_labels/vol_scaled/feature_matrix/` artifacts. The emitted
matrix matches all `236117` label rows, but only four safe OOF score/signal
features survive leakage screening: `y_pred`, `prob_up`, `confidence`, and
`long_trade_taken`. No stronger model training is performed by this batch.

To attempt a market-feature training-frame export for a completed M20 run, use:

* `python .\scripts\export_m20_training_frame_features.py --run-dir .\artifacts\training\m20\<run_id>`

For `20260427T112021Z`, this writes a blocker under
`research_labels/vol_scaled/training_frame_export/`: the manifests preserve the
configured OHLC-derived feature names, but the completed artifact does not
contain a row-level market feature frame with `symbol + interval_begin` keys.
Only OOF score/policy outputs are row-aligned today. The blocker includes the
required future export schema for the next M20 scoring/training run and does
not train a model or change runtime behavior.

Future M20 runs can opt in to preserving that row-level market-feature frame:

* `python -m app.training --config .\configs\training.m20.json --export-training-frame`
* `.\scripts\start_m20_training.ps1 -DryRun -ExportTrainingFrame`

The hook writes `training_frame/` artifacts inside the active run directory and
excludes score outputs, labels, future returns, barrier metadata, realized
outcomes, and post-event columns. The flag is off by default.

Evidence note: the first export-enabled score-only run
`artifacts/training/m20/20260505T200658Z` completed model scoring too late for
the hook to persist `training_frame/` before interruption/timeout. The next
recovery patch should move the export earlier or add an export-only path before
any market-feature baseline is trained.

Follow-up evidence: export-only mode now writes the frame before model scoring.
Run `artifacts/training/m20/20260505T212518Z` contains `training_frame/` with
`236153` rows, fold `4`, and `22` configured OHLC-derived market features.
Folds `0` through `3` are recorded as skipped by the recent-window filter, and
model scoring was skipped.

That export-only run can now generate research labels directly from
`training_frame/`:

* `python .\scripts\generate_m20_research_labels.py --run-dir .\artifacts\training\m20\20260505T212518Z --source training-frame --use-volatility`

The generated `research_labels/vol_scaled/` artifacts use row-aligned
`realized_vol_12` and `close_price`. Meta-labeling is not applicable for this
export-only run because no OOF/default prediction signals exist. The readiness
gate marks fee-exceedance labels ready by coarse class-balance checks and
triple-barrier labels not ready because the neutral/HOLD rate remains high. No
model training, runtime inference, registry authority, promotion, paper/live
execution, thresholds, NeuralForecast behavior, or roster behavior changed.

The next research-only batch trains tiny fee-exceedance baselines on those
market features:

* `python .\scripts\train_m20_fee_exceedance_baseline.py --run-dir .\artifacts\training\m20\20260505T212518Z`

The current result is a feasibility diagnostic only: `logistic_regression_tiny`
beats always-negative and fixed-seed random diagnostics on balanced accuracy and
average precision, but it is not runtime-comparable, not promotable, not written
to the registry, and not profitability evidence.

Conditional usefulness analysis is available for the fee-exceedance baseline:

* `python .\scripts\analyze_m20_conditional_usefulness.py --run-dir .\artifacts\training\m20\20260505T212518Z`

The current report analyzes the available deterministic test prediction sample
and classifies slices by symbol, time, volatility, momentum, RSI, MACD, volume,
range, and simple regime-lite buckets. Slice findings are strategy-ensemble
inputs only and require confirmation on another fold/window before any runtime
use.

Full-test confirmation is available after exporting full split predictions:

* `python .\scripts\train_m20_fee_exceedance_baseline.py --run-dir .\artifacts\training\m20\20260505T212518Z --export-full-predictions`
* `python .\scripts\analyze_m20_conditional_usefulness.py --run-dir .\artifacts\training\m20\20260505T212518Z --prediction-source full-test`

The full-test report uses `47229` test rows and writes
`research_labels/vol_scaled/conditional_usefulness_full_test/`. It remains
research-only, single recent-fold only, not runtime-comparable, not promotable,
and not profitability evidence.

M20 also has a research-only model/member audit:

* `python .\scripts\audit_m20_model_members.py --run-dir .\artifacts\training\m20\20260505T212518Z --previous-run-dir .\artifacts\training\m20\20260427T112021Z --fitted-models-dir .\artifacts\training\m20\20260405T023104Z\fitted_models`

The audit treats AutoGluon as a model factory / ensemble manager rather than one
indivisible model. In the inspected M20 paths, AutoGluon member metadata and
member-level predictions were not present, so the ledger records a future export
requirement instead of inventing member evidence. Existing weak candidates are
retained for conditional review.

## Historical backfill and feature replay

To import the full local Kraken downloadable OHLCVT CSV dataset into the real `raw_ohlc` table, replay the same live feature logic into `feature_ohlc`, and persist a training-readiness artifact, run:

* `.\scripts\import_kraken_ohlcvt.ps1`

That wrapper defaults to the extracted local dataset root at:

* `.\Datasets\master_q4`

and imports the authoritative 5-minute files for:

* `BTC/USD`
* `ETH/USD`
* `SOL/USD`

To replay features and refresh the readiness report from already imported raw CSV truth without touching `raw_ohlc` again, run:

* `.\scripts\import_kraken_ohlcvt.ps1 -ReplayOnly`

Important honesty note:

* Kraken's downloadable OHLCVT CSV files do not include VWAP in the current local format, while the real `raw_ohlc` contract does require it.
* The importer keeps the real `raw_ohlc -> feature_ohlc -> training` path and records this explicitly in `import_operation.json`.
* When CSV VWAP is missing, imported rows use `close_price` as the persisted VWAP fallback so the existing feature contract can be replayed without inventing a second raw schema.

To backfill the real Kraken 5-minute raw history into `raw_ohlc`, replay the same live feature logic into `feature_ohlc`, and persist a training-readiness artifact, run:

* `python -m app.ingestion.backfill_ohlc --symbols BTC/USD ETH/USD SOL/USD --start 2026-03-31T11:50:00Z --end 2026-04-02T11:50:00Z --training-config .\configs\training.m7.json`

To replay features and refresh the readiness report from already persisted raw history without calling Kraken again, run:

* `python -m app.ingestion.backfill_ohlc --symbols BTC/USD ETH/USD SOL/USD --start 2026-03-31T11:50:00Z --end 2026-04-02T11:50:00Z --training-config .\configs\training.m7.json --skip-raw-backfill`

Each run writes a readiness artifact bundle under:

* `artifacts/training/data_readiness/<run_id>/`

Important files in that bundle are:

* `readiness_report.json`
* `symbol_coverage.csv`
* `gap_summary.csv`
* `summary.md`
* `backfill_operation.json`
* `import_operation.json` for local Kraken CSV imports

Minimum historical readiness before rerunning AutoGluon experiments means:

* `ready_for_training` is `true` for the configured walk-forward split
* configured symbols have real rows in both `raw_ohlc` and `feature_ohlc`
* the report's gap warnings are understood and acceptable for the experiment window

Kraken's public OHLC REST endpoint only returns the most recent 720 entries, so older gaps cannot be recovered on demand from that source once they fall outside that window. The readiness artifact keeps those gaps explicit instead of pretending they were filled.

For local AutoGluon challenger training, use the repo-native operator flow instead of raw Docker profile reasoning:

* `./scripts/prepare_m7_training.ps1`
* `./scripts/start_m7_training.ps1`

The prepare script checks the checked-in M7 config, AutoGluon availability, PostgreSQL reachability, and `feature_ohlc` readiness. If the training source is missing or not yet large enough for the configured walk-forward split, it starts the existing `dev` stack path to populate features and then prints the next recommended command.

The prepare script also reports whether optional `fastai` breadth is actually usable for AutoGluon, not just installed. Missing or broken `fastai` breadth is still not treated as a blocker for the authoritative M7 AutoGluon path.

The start script runs the same readiness checks, forces the local training process onto a repo-local same-drive temp root for AutoGluon and Ray, then invokes the authoritative training command `python -m app.training --config .\configs\training.m7.json`. During the run it shows a PowerShell progress heartbeat with elapsed time, the configured AutoGluon time budget, the latest discovered model, and the current best model when available. After the run it prints the newest artifact directory plus the `summary.json` winner and acceptance flags to inspect.

After a completed M7 run, use the post-training threshold research helper to inspect cost-aware long-only thresholds and regime blocks without changing runtime policy:

* `./scripts/analyze_m7_thresholds.ps1`

That script defaults to the newest M7 artifact, evaluates the winner model's OOF `prob_up` thresholds under cost-aware long-only policies, and writes a `threshold_analysis/` folder into the run directory with JSON, CSV, and Markdown summaries.

To compare explicit named research candidates instead of ad hoc threshold sweeps, use:

* `./scripts/evaluate_m7_policy_candidates.ps1`

That script defaults to the newest completed M7 artifact, evaluates the bounded built-in named candidates against the winner model's OOF predictions, and writes a `policy_candidate_analysis/` folder into the run directory. The candidate set now includes a small explicit regime-routing ablation family such as RANGE-only, TREND_UP-only, and TREND_DOWN/HIGH_VOL-blocked variants. This remains research support only and does not change production inference or promotion behavior.

To judge whether those named candidates look robust across multiple completed M7 runs instead of one-run luck, use:

* `./scripts/evaluate_m7_policy_candidates_multi_run.ps1`

That script scans completed runs under `artifacts/training/m7`, excludes incomplete runs, skips legacy runs whose `oof_predictions.csv` schema is too old for regime-aware cost analysis, aggregates candidate performance across analyzable runs, and writes a stable `_analysis/policy_candidates/` summary under the M7 artifact root. This is still research support only and does not change runtime or promotion behavior.

To replay a bounded shortlist of named M7 research candidates as simple long-only proxy ledgers on a completed run, use:

* `./scripts/evaluate_m7_policy_replay.ps1`

That script defaults to the newest completed M7 artifact, replays the built-in shortlist in time order using the saved OOF proxy fields, and writes a `policy_replay_analysis/` folder into the run directory with cumulative net, drawdown, and trade-ledger outputs. It stays research-only and does not change production runtime or promotion behavior.

To judge whether those replay paths hold up across multiple completed M7 runs, use:

* `./scripts/evaluate_m7_policy_replay_multi_run.ps1`

That script scans completed runs under `artifacts/training/m7`, excludes incomplete runs, skips legacy runs whose OOF schema is too old for replay analysis, aggregates cumulative-path metrics across analyzable runs, and writes a stable `_analysis/policy_replay/` summary under the M7 artifact root. This remains research-only and does not change runtime or promotion behavior.

For a bounded stronger-config research pass, use:

* `./scripts/run_m7_research_experiments.ps1`

That runner discovers the small checked-in M7 AutoGluon research config set under `configs/training.m7.research.*.json`, runs each config through the existing Windows-safe M7 training path, evaluates the existing named policy candidates after each completed run, and writes a compact `_analysis/research_experiments/` summary under `artifacts/training/m7`. This remains research support only and does not change runtime or promotion behavior.

For completed-run data, regime, and opportunity diagnostics, use:

* `./scripts/analyze_m7_data_regime.ps1`

That script defaults to the newest completed M7 artifact, analyzes label balance, opportunity density, regime routing, and fold drift from the saved run artifacts, and writes a `data_regime_diagnostics/` folder into the run directory. It stays research-only and does not change runtime or promotion behavior.

For research-only forward paper observation of top M7 policy candidates while the normal paper system runs, use:

* `./scripts/show_live_policy_challengers.ps1`

That script reads the research-only challenger scoreboard written under `artifacts/paper_trading/paper/research/policy_challengers/`, compares the bounded candidate shortlist against the active production policy on the same observed paper rows, and prints cumulative proxy, drawdown, trade-count, and sparse-routing warnings. It is observer-only: it does not route execution, place extra trades, or change the active runtime policy.

## Linux VPS paper observation

To run the normal paper stack plus the research-only live challenger sidecar on a Linux VPS for forward observation, use:

* `./scripts/deploy_paper_vps.ps1`
* `./scripts/status_paper_vps.ps1`
* `./scripts/show_live_policy_challengers_vps.ps1`
* `./scripts/stop_paper_vps.ps1`

The deploy path reads VPS connection settings from the local root `.env`, supports the existing legacy aliases already used in this repo, uploads only the bounded paper/runtime deployment set, writes a sanitized remote `.env`, starts the Docker Compose `paper` profile, and keeps challenger scoring observer-only through the normal paper runner path. It does not train on the VPS and it does not change live behavior.

* `./scripts/stop-stack.ps1`
* `./scripts/reset-state.ps1`
* `./scripts/prune-runtime-artifacts.ps1 -RetentionDays 14`

## Quick Start

### 1. Prepare the environment

```powershell
Copy-Item .env.example .env
```

### 2. Start the core local stack

```powershell
docker compose up --build -d
```

### 3. Follow the main service logs

```powershell
docker compose logs -f producer
docker compose logs -f features
```

### 4. Run the inference API

Set explicit artifact paths if you want overrides. Otherwise, inference resolves the current promoted champion and latest regime thresholds automatically when configured to do so.

```powershell
python -m app.inference
```

### 5. Run the trader once

```powershell
python scripts\run_paper_trader.py --config configs\paper_trading.yaml --once
```

### 6. Run the Streamlit operator console

```powershell
$env:POSTGRES_HOST = "127.0.0.1"
$env:INFERENCE_API_BASE_URL = "http://127.0.0.1:8000"
streamlit run dashboards/streamlit_app.py
```

## Core API Surfaces

### Base runtime endpoints

* `GET /health`
* `GET /latest-features?symbol=BTC/USD`
* `GET /predict?symbol=BTC/USD`
* `GET /signal?symbol=BTC/USD`
* `GET /regime?symbol=BTC/USD`
* `GET /freshness?symbol=BTC/USD`
* `GET /reliability/system`
* `GET /metrics`

### M17 operational endpoints

* `GET /alerts/active`
* `GET /alerts/timeline`
* `GET /operations/daily-summary`
* `GET /operations/startup-safety`

### M19 adaptation endpoints

* `GET /adaptation/summary`
* `GET /adaptation/drift`
* `GET /adaptation/performance`
* `GET /adaptation/profiles`
* `GET /adaptation/promotions`

### M20 and M21 read surfaces

* read-only ensemble and continual-learning status is surfaced through prediction, signal, health, traces, and dashboard snapshots
* `GET /continual-learning/summary`
* `GET /continual-learning/experiments`
* `GET /continual-learning/profiles`
* `GET /continual-learning/drift-caps`
* `GET /continual-learning/promotions`
* `GET /continual-learning/events`

## M21 Guarded Continual-Learning Workflow

M21 currently exposes a **manual, guarded, evidence-based** continual-learning workflow surface. The runtime now persists drift-cap truth, but active profiles, experiments, and promotions should only be treated as present when they exist in real persisted state.

It is designed to be:

* shadow-first
* measurable
* reversible
* explicit about block reasons
* explicit about rollback targets

It is not designed to do:

* uncontrolled live self-retraining
* in-place mutation of production model artifacts
* hidden autonomous promotion

### M21 approved candidate types

* `CALIBRATION_OVERLAY`
* `INCREMENTAL_SHADOW_CHALLENGER`

Only `CALIBRATION_OVERLAY` may become `LIVE_ELIGIBLE` in M21.
`INCREMENTAL_SHADOW_CHALLENGER` remains shadow-only in M21.

### Guarded promotion rules

Promotion is blocked when any of these are true:

* target profile does not exist
* operator confirmation is missing
* latest matching drift-cap status is `BREACHED`
* health is not `HEALTHY`
* freshness is not `FRESH`
* candidate policy disallows the requested stage
* an `INCREMENTAL_SHADOW_CHALLENGER` requests `LIVE_ELIGIBLE`

Successful promotion:

* activates the target profile only within its exact scope
* supersedes only the incumbent active profile in that exact scope
* writes explicit decision truth
* writes explicit event truth
* preserves rollback visibility

Blocked promotion:

* returns explicit `blocked=true`
* writes persisted `HOLD` decision truth when profile truth exists
* writes persisted `PROMOTION_BLOCKED` event truth when profile truth exists

### Guarded rollback rules

Rollback remains explicit and scoped:

* there must be an active profile for the supplied scope
* that profile must expose an explicit `rollback_target_profile_id`
* blocked rollback outcomes remain visible and persisted when profile truth exists

Successful rollback:

* restores the explicit rollback target
* writes persisted `ROLLBACK` decision truth
* writes persisted `ROLLBACK_APPLIED` event truth

### M21 workflow endpoints

* `POST /continual-learning/promotions/promote-profile`
* `POST /continual-learning/promotions/rollback-active-profile`

Blocked business-rule outcomes still return `200` with explicit workflow truth.

### PowerShell example: promote profile

```powershell
$body = @'
{
  "decision_id": "promote:cl-profile-1:20260322T120000Z",
  "profile_id": "cl-profile-1",
  "requested_promotion_stage": "PAPER_APPROVED",
  "summary_text": "Manual guarded promotion after reviewing M21 evidence.",
  "reason_codes": [
    "OPERATOR_REVIEWED_EVIDENCE",
    "MANUAL_M21_PROMOTION"
  ],
  "operator_confirmed": true
}
'@

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/continual-learning/promotions/promote-profile `
  -ContentType 'application/json' `
  -Body $body
```

### PowerShell example: rollback active profile

```powershell
$body = @'
{
  "decision_id": "rollback:cl-profile-1:20260322T120500Z",
  "execution_mode": "paper",
  "symbol": "BTC/USD",
  "regime_label": "TREND_UP",
  "summary_text": "Manual guarded rollback to the explicit prior M21 profile.",
  "operator_confirmed": true
}
'@

Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/continual-learning/promotions/rollback-active-profile `
  -ContentType 'application/json' `
  -Body $body
```

## Execution Modes and Live Safety

### Paper

* authoritative M4 signals
* authoritative M10 risk
* local state mutation
* no broker dependency

### Shadow

* authoritative M4 signals
* authoritative M10 risk
* execution truth and order lifecycle audit without live broker submission

### Live

* explicit runtime arming required
* explicit config gating required
* tiny notional limits
* whitelisting required
* reconciliation truth required
* fail-closed health gating

Guarded live still does not mean fully autonomous live trading. Broker-truth mismatch, degraded reliability, or unresolved reconciliation should block live submission.

## Operator Console

The Streamlit operator console now reflects the accepted **M15 through M18** foundations plus the currently available M19-M21 runtime truth surfaces.

Main views:

* Market
* Signals
* Trades
* Risk
* Health
* Models
* Incidents

Key operator surfaces include:

* mode and safety banners
* market freshness and lag views
* signal context and decision traces
* risk rationale and blocked-trade visibility
* reliability status and alerts
* adaptation status
* continual-learning workflow status
* workflow event visibility for promotion and rollback

## Validation

### Local validation remains primary

Run the full local validation path when doing serious milestone work:

```powershell
python -m pytest
python -m pylint app dashboards tests scripts\run_paper_trader.py
```

### Baseline GitHub Actions validation

A minimal validation workflow now runs on:

* `push`
* `pull_request`

Purpose:

* protect the accepted v2 baseline
* run a fast lint and pytest slice
* avoid pretending to be full end-to-end production validation

Deeper repository-backed checks still remain part of local validation.

## Artifacts and Local Evidence

Important artifact roots:

* `artifacts/training/`
* `artifacts/registry/`
* `artifacts/regime/`
* `artifacts/paper_trading/`
* `artifacts/reliability/`
* `artifacts/rationale/`
* `artifacts/live/`
* `artifacts/adaptation/`
* `artifacts/continual_learning/`
* `artifacts/operations/`

Important local truth sources:

* PostgreSQL tables for canonical runtime and workflow state
* local artifacts for operator evidence and reports
* dashboard and API surfaces that read from those truths without inventing parallel hidden state

## Environment Variables

Copy `.env.example` to `.env` before starting the stack.

Important examples:

* `APP_NAME`
* `LOG_LEVEL`
* `POSTGRES_HOST`
* `POSTGRES_PORT`
* `POSTGRES_DB`
* `POSTGRES_USER`
* `POSTGRES_PASSWORD`
* `INFERENCE_API_BASE_URL`
* `INFERENCE_MODEL_PATH`
* `INFERENCE_REGIME_THRESHOLDS_PATH`
* `INFERENCE_REGIME_SIGNAL_POLICY_PATH`
* `DASHBOARD_REFRESH_SECONDS`
* `APCA_API_KEY_ID`
* `APCA_API_SECRET_KEY`
* `ALPACA_BASE_URL`
* `STREAMALPHA_ENABLE_LIVE`
* `STREAMALPHA_LIVE_CONFIRM`

The full authoritative environment-variable table should continue to live in `.env.example` and may be expanded there over time.

## Known Limits

Current intentional limits include:

* no RL
* no sentiment or news integration
* no unrestricted online learning
* no automatic live self-retraining
* no portfolio optimizer
* no hidden orchestration layer
* no broker-fill import pretending to be canonical when reconciliation is unresolved
* no high-availability production deployment claim

## References

* Kraken WebSocket v2 trade docs
* Kraken WebSocket v2 OHLC docs
* Kraken WebSocket v2 status and heartbeat docs
* Redpanda quick-start documentation

## Closeout

Second Foundation v2 is mostly aligned through accepted M18, with active but still limited M19-M21 runtime truth surfaces.

That means:

* prediction, risk, execution, reliability, explainability, operator visibility, adaptation, ensemble behavior, and continual-learning workflow all have explicit accepted boundaries
* continual learning is manual and guarded
* continual-learning changes are measurable and reversible
* there is no uncontrolled live self-retraining
* GitHub Actions baseline validation now protects the accepted v2 branch, while local-first validation remains primary

The honest current limitation is that the old sklearn models now remain legacy-only historical artifacts, while the authoritative path has only one real `autogluon_tabular` family and that proof is still negative after costs. M19/M21 should therefore be judged from the real persisted truth now generated at runtime rather than stronger profile, promotion, or specialist-model diversity that does not yet exist.

M20 research now has a manual-only confirmation plan for the first full-test signal-positive fee-exceedance logistic baseline. The plan records the target slices, success rules, expected artifacts, and follow-up commands under `artifacts/training/m20/20260505T212518Z/research_labels/vol_scaled/confirmation_plan/`, while explicitly keeping the evidence single-recent-fold only, not runtime-comparable, not promotable, and not profitability evidence. AutoGluon member analysis remains blocked until member-level metadata or predictions are exported.

The M20 training CLI also supports a safe manual-only confirmation window override for future score-only/export-only research runs: `--confirmation-window-start`, `--confirmation-window-end`, and `--confirmation-tag`. The override is off by default, validates timestamps and tag safety, records metadata when used, and still requires Arash to launch any long confirmation run manually.

The first manual confirmation artifact, `artifacts/training/m20/20260506T054337Z`, has now been processed through the short research pipeline. Its prior-year fee-exceedance baseline confirmed the original strongest slices (`momentum=flat`, `range=low`, `symbol=BTC/USD`, `macd=positive`, `volume=low`) as strongly confirmed research signals, while the April 2026 disable slices were outside the confirmation window and remain untested coverage gaps. This supports continuing research-only strategy-selector design, but it still does not create runtime, registry, promotion, live/paper, threshold, NeuralForecast, roster, or profitability evidence.

That evidence now feeds a research-only selector design artifact, `fee_exceedance_gate_v0_research`, under `artifacts/training/m20/20260505T212518Z/research_labels/vol_scaled/strategy_selector_design/`. It is an opportunity-gate design for future simulations only, not runtime selector logic, not promotable, not registry-backed, and not a trading or profitability claim. PatchTST/NHITS remain conditionally unknown, and AutoGluon member predictions remain missing.

The first selector simulation is also research-negative for the broad selector rules: global logistic top-5% ranking stayed much sharper, while the weighted confirmed-slice selector admitted nearly all rows and fell back to base-rate precision. The fee-exceedance score remains useful as a ranker, but the selector needs narrower held-out tuning or another confirmation-window simulation before any strategy-family design work.

A rank-gated selector evaluation now compares global top-k fee-exceedance ranking with condition-then-rank policies at 1%, 2%, 5%, and 10% coverage under `artifacts/training/m20/20260505T212518Z/research_labels/vol_scaled/rank_gated_selector/`. This remains research-only and does not implement runtime selector behavior. The strongest stable result still comes from rank gating around the logistic score; it is not a backtest, not promotable, not registry-backed, and not profitability evidence.

Nested held-out rank-gate tuning now selects rank-gate parameters on validation only and evaluates locked parameters on original test plus the prior-year confirmation run. The selected research policy is `CONDITION_THEN_TOP_0.25`, with original test lift `1.841966` and confirmation lift `2.185905`. This is still research-only, not runtime, not a backtest, not promotable, and not a profit claim.

The next M20 comparable confirmation-window run is a manual execution break. Codex must not launch it. Codex should print the PowerShell commands, stop, and wait for Arash to run them in a separate PowerShell terminal and paste back the resulting run directory and metrics. `CONDITION_THEN_TOP_0.25` remains research-only until that separate comparable window is processed; there is still no runtime, registry, promotion, policy simulation, trading/backtest, model-retrain, or profit-claim change.

Arash manually completed the next export-only confirmation run outside Codex at `artifacts/training/m20/20260506T063818Z`, covering `2023-04-02T11:30:00Z` through `2024-04-02T11:30:00Z` with tag `confirm_prev_prev_year`. The short post-export research pipeline has now produced labels, readiness, fee-exceedance baseline predictions, conditional usefulness, slice comparison, and nested rank-gate metrics. The locked `CONDITION_THEN_TOP_0.25` confirmation result selected `153` rows with coverage `0.002497`, precision `0.522876`, lift `2.262976`, recall `0.005652`, false positives `73`, average probability `0.998087`, and disable-gap exposure `0`. The comparison still has missing slices, so the candidate remains research-only and still creates no runtime, registry, promotion, policy simulation, trading/backtest, model-retrain, or profit-claim change.

The missing comparison slices are now adjudicated as `CALENDAR_SLICE_NON_OVERLAP`: `month=2025-11`, `month=2025-12`, `month=2026-04`, `quarter=2025Q4`, and `quarter=2026Q2` are absent because the confirmation window is a different calendar period. For locked `CONDITION_THEN_TOP_0.25`, the current evidence records `DISABLE_GAP_NO_SELECTED_EXPOSURE_IN_LOCKED_GATE`; the general blocker remains `DISABLE_GAP_STILL_UNCONFIRMED_FOR_GENERAL_CONDITIONAL_ANALYSIS`.

A lightweight research-only rank-gate evidence packet is available at `artifacts/training/m20/20260505T212518Z/research_labels/vol_scaled/rank_gate_evidence_packet/`. It records `RESEARCH_CONFIRMED_RANK_GATE`, runtime status `NOT_RUNTIME_READY`, and promotion status `NOT_PROMOTABLE`. Window metrics are: original locked test coverage/precision/lift `0.002498` / `0.347458` / `1.841966`; prior-year confirmation `0.002496` / `0.512821` / `2.185905`; prev-prev-year confirmation `0.002497` / `0.522876` / `2.262976`. Blockers remain sparse rank-gate coverage, calendar-slice non-overlap, disable gaps unconfirmed for general conditional analysis, no profitability evidence, and not runtime-ready.

An offline rank-gate economics diagnostic is also available at `artifacts/training/m20/20260505T212518Z/research_labels/vol_scaled/rank_gate_economics/`. It compares `CONDITION_THEN_TOP_0.25` with global top-k and no-gate baselines using existing fee labels only. The locked gate keeps lift above `1.84` across the three windows, but net return proxies are mixed (`-0.180157`, `0.189677`, `-0.019105`), so the result stays `RESEARCH_ONLY`, `NOT_BACKTEST`, `NOT_PROFIT_EVIDENCE`, `NO_RUNTIME`, `NO_REGISTRY`, and `NO_PROMOTION`.

A follow-up net-proxy diagnostic is available at `artifacts/training/m20/20260505T212518Z/research_labels/vol_scaled/rank_gate_net_diagnostics/`. It decomposes the locked gate by run, symbol, month/quarter, probability bin, volatility/range/volume/MACD/momentum buckets, and tail rows. It preserves `NET_PROXY_MIXED`, `NOT_PNL`, `NO_PROFIT_CLAIM`, sparse selection, and no runtime/registry/promotion effect.

A tail/condition concentration analysis is also available at `artifacts/training/m20/20260505T212518Z/research_labels/vol_scaled/rank_gate_tail_analysis/`. It finds negative net-proxy windows in the original and prev-prev-year confirmations and flags unstable tail/condition concentration, so `CONDITION_THEN_TOP_0.25` stays research-only, not PnL, not a backtest, not promotable, and not runtime-ready.

A research-only tail-risk filter sweep is available at `artifacts/training/m20/20260505T212518Z/research_labels/vol_scaled/rank_gate_tail_filter/`. It tests high-range/high-volatility/unstable-bucket exclusions and probability cutoffs on the locked selected rows, but the result is `NO_STABLE_TAIL_FILTER_FOUND`; it is an exploratory test-split sweep only, not runtime logic, not PnL, not a backtest, and not profitability evidence.

The M20 research decision memo is available at `artifacts/training/m20/20260505T212518Z/research_labels/vol_scaled/m20_decision_memo/`. It records `RESEARCH_SIGNAL_CONFIRMED`, `ECONOMICS_NOT_STABLE`, `NO_STABLE_TAIL_FILTER`, `NOT_RUNTIME_READY`, and `NOT_PROMOTABLE`. The memo pauses the rank-gate as a standalone path, while preserving forks for richer strategy-family modules, AutoGluon member prediction export, alternate horizons/labels, and packaging M20 as infrastructure-positive but profitability-negative research.

A research-only strategy-family scaffold is available at `artifacts/training/m20/20260505T212518Z/research_labels/vol_scaled/strategy_family_scaffold/`. It defines design-only families for `momentum_breakout`, `range_mean_reversion`, `volatility_expansion`, and `abstention_hold`; the rank gate is only an optional filter input, and no runtime strategy, registry, promotion, trading/backtest, model-retrain, or profit-claim behavior is added.

The first momentum_breakout diagnostic is available at `artifacts/training/m20/20260505T212518Z/research_labels/vol_scaled/momentum_breakout_research/`. It keeps `realized_vol_high`, `range_high`, and `volume_high` as research diagnostic setup candidates across the original/prior/prev-prev windows, but it is setup-frequency and fee-label lift analysis only: no runtime, registry, promotion, trading/backtest, model-retrain, or profit claim.

The gate+momentum combo diagnostic is available at `artifacts/training/m20/20260505T212518Z/research_labels/vol_scaled/gate_momentum_combo/`. It shows `GATE_AND_MOMENTUM` and `GATE_THEN_MOMENTUM_TOPK` are equivalent to the paused locked gate on selected rows, so the recommendation is `NO_INCREMENTAL_COMBO_EDGE_OVER_PAUSED_GATE`; momentum-only setups have label lift but broad negative net proxies. This remains diagnostic-only and non-runtime.

<!-- M20_RANGE_MEAN_REVERSION_RESEARCH -->
M20 range_mean_reversion research diagnostic:
Command: python .\scripts\analyze_m20_range_mean_reversion.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
Writes research_labels/vol_scaled/range_mean_reversion_research/. Research-only diagnostic: no runtime, registry, promotion, trading/backtest, model-retrain, long-run, or profit-claim behavior is added.

<!-- M20_VOLATILITY_EXPANSION_RESEARCH -->
M20 volatility_expansion research diagnostic:
Command: python .\scripts\analyze_m20_volatility_expansion.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
Writes research_labels/vol_scaled/volatility_expansion_research/. The result keeps volatility-expansion as a research diagnostic candidate, with the strongest stable setup `vol_plus_range_high` at minimum lift `1.602734` across the original/prior/prev-prev windows. This is fee-label/setup diagnostics only: no runtime, registry, promotion, trading/backtest, model-retrain, long-run, PnL, or profit-claim behavior is added.

<!-- M20_STRATEGY_FAMILY_ADJUDICATION -->
M20 strategy-family adjudication packet:
Command: python .\scripts\adjudicate_m20_strategy_families.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
Writes research_labels/vol_scaled/strategy_family_adjudication/. Current recommendation is `TEST_VOLATILITY_EXPANSION_NEXT`: `volatility_expansion` is primary, `momentum_breakout` is secondary, and `range_mean_reversion` stays watchlist. This uses existing artifacts only and remains research-only: no runtime, registry, promotion, trading/backtest, model-retrain, long-run, PnL, or profit-claim behavior is added.

<!-- M20_VOLATILITY_EXPANSION_DEEP_DIVE -->
M20 volatility_expansion deep dive:
Command: python .\scripts\analyze_m20_volatility_expansion_deep_dive.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
Writes research_labels/vol_scaled/volatility_expansion_deep_dive/. Current recommendation is `TEST_VOLATILITY_EXPANSION_COMBO_NEXT`; primary setup `vol_plus_range_high` has lift `1.610740` original, `1.602734` prior-year, and `1.892837` prev-prev-year. This remains diagnostic-only and uses existing artifacts only; no runtime, registry, promotion, trading/backtest, model-retrain, long-run, PnL, or profit-claim behavior is added.

<!-- M20_VOLATILITY_COMBO_ECONOMICS -->
M20 volatility combo economics diagnostic:
Command: python .\scripts\analyze_m20_volatility_combo_economics.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
Writes research_labels/vol_scaled/volatility_combo_economics/. Current recommendation is `TRY_VOLATILITY_AS_OPTIONAL_GATE_FILTER`: gate-and-volatility is equivalent to the paused locked rank gate, while volatility-only setups preserve label lift but show negative average net proxy across all three windows. This remains research-only and not runtime, registry, promotion, trading/backtest, PnL, or profit evidence.

<!-- M20_ABSTENTION_HOLD_RESEARCH -->
M20 abstention/HOLD research diagnostic:
Command: python .\scripts\analyze_m20_abstention_hold.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
Writes research_labels/vol_scaled/abstention_hold_research/. Current recommendation is `KEEP_ABSTENTION_AS_RESEARCH_FILTER`: `HOLD_BROAD_UNSTABLE_VOLATILITY` is watchlist only, while the strongest avoided-loss rules use after-the-fact net-proxy oracle information and are not implementable. This remains research-only and not runtime HOLD logic, registry, promotion, trading/backtest, PnL, or profit evidence.

<!-- M20_RESEARCH_PATH_ADJUDICATION -->
M20 research path adjudication:
Command: python .\scripts\write_m20_research_path_adjudication.py --base-run-dir .\artifacts\training\m20\20260505T212518Z
Writes research_labels/vol_scaled/m20_research_path_adjudication/. Current decision is `STOP_CURRENT_FILTER_CHAIN_AND_PLAN_SPECIALIST_EXPORT`; next action is `PLAN_ROW_LEVEL_SPECIALIST_PREDICTION_EXPORT`. Rank-gate signal remains real, but current filter/volatility/abstention paths do not provide stable implementable economics. No runtime, registry, promotion, trading/backtest, model-retrain, long-run, PnL, or profit-claim behavior is added.

<!-- M20_SPECIALIST_PREDICTION_EXPORT_PLAN -->
M20 specialist prediction export plan:
Command: python .\scripts\plan_m20_specialist_prediction_export.py --base-run-dir .\artifacts\training\m20\20260505T212518Z --fitted-models-dir .\artifacts\training\m20\20260405T023104Z\fitted_models --previous-run-dir .\artifacts\training\m20\20260427T112021Z
Writes research_labels/vol_scaled/specialist_prediction_export_plan/. Current recommendation is `ADD_LIGHTWEIGHT_PREDICTION_EXPORT_HOOK_FIRST`. It identifies 14 NHITS/PatchTST row-level prediction targets, including existing 20260427 OOF candidates to sanitize/analyze and fitted-model candidates that still need export. Blockers remain `LONG_RUNS_MANUAL_ONLY`, `PER_SPECIALIST_EXPORT_HOOK_NOT_CONFIRMED`, and `AUTOGLUON_MEMBER_PREDICTIONS_MISSING`. No export, score-only rerun, runtime, registry, promotion, trading/backtest, model-retrain, PnL, or profit-claim behavior is added.

<!-- M20_SPECIALIST_PREDICTION_EXPORT -->
M20 existing specialist prediction export:
Command: python .\scripts\export_m20_existing_specialist_predictions.py --base-run-dir .\artifacts\training\m20\20260505T212518Z --previous-run-dir .\artifacts\training\m20\20260427T112021Z
Writes research_labels/vol_scaled/specialist_predictions/. It sanitizes existing OOF rows only: 236,153 `neuralforecast_nhits` rows and 236,153 `neuralforecast_patchtst` rows. Future/net proxy columns are quarantined from prediction outputs. No score-only rerun, model retrain, runtime, registry, promotion, trading/backtest, PnL, or profit-claim behavior is added.

<!-- M20_SPECIALIST_CONDITIONAL_USEFULNESS -->
M20 specialist conditional usefulness:
Command: python .\scripts\analyze_m20_specialist_conditional_usefulness.py --base-run-dir .\artifacts\training\m20\20260505T212518Z --previous-run-dir .\artifacts\training\m20\20260427T112021Z
Writes research_labels/vol_scaled/specialist_conditional_usefulness/. Existing OOF rows only: 472,234 joined fee-label rows after skipping 72 horizon/unlabeled rows. Best candidate is `neuralforecast_patchtst` with top-5% lift `1.743825`, PR-AUC `0.138238`, ROC-AUC `0.476149`, and 15 conditional research slices; `neuralforecast_nhits` has top-5% lift `1.356309` and 2 slices. The original `RUN_SPECIALIST_CONFIRMATION_EXPORT` recommendation is now superseded by score-only confirmation, safe economic evaluation, cost-aware adjudication, and the generic strategy candidate factory. No score-only rerun, model retrain, runtime, registry, promotion, trading/backtest, PnL, or profit-claim behavior is added.

<!-- M20_SPECIALIST_CONFIRMATION_PLAN -->
M20 specialist confirmation plan:
Command: python .\scripts\plan_m20_specialist_confirmation.py --base-run-dir .\artifacts\training\m20\20260505T212518Z --previous-run-dir .\artifacts\training\m20\20260427T112021Z --fitted-models-dir .\artifacts\training\m20\20260405T023104Z\fitted_models
Writes research_labels/vol_scaled/specialist_confirmation_plan/. Primary candidate is `neuralforecast_patchtst`, secondary is `neuralforecast_nhits`, and 17 target slices are listed for manual confirmation. The original `ADD_SPECIALIST_CONFIRMATION_EXPORT_HOOK_FIRST` recommendation is historical; the confirmation/export/economic-adjudication path has since completed and the current next step is generic strategy candidate refinement. No export, score-only rerun, model retrain, runtime, registry, promotion, trading/backtest, PnL, or profit-claim behavior is added.

<!-- M20_SPECIALIST_CONFIRMATION_EXPORT_HOOK -->
M20 specialist confirmation export hook:
The training CLI now accepts research-only `--export-specialist-predictions-only` during `--score-only` runs, and `scripts/start_m20_training.ps1 -DryRun` supports `-ScoreOnly`, `-ParquetDir`, and `-ExportSpecialistPredictionsOnly`. The hook writes sanitized NHITS/PatchTST row-level prediction files under `research_labels/vol_scaled/specialist_predictions/` when Arash manually launches a score-only confirmation run. No long run was launched by Codex, and no runtime, registry, promotion, trading/backtest, model-retrain, PnL, or profit-claim behavior is added.

<!-- M20_SPECIALIST_CONFIRMATION_ADJUDICATION -->
M20 specialist confirmation adjudication:
Command: python .\scripts\write_m20_specialist_confirmation_adjudication.py --confirmation-run-dir .\artifacts\training\m20\20260507T135017Z --original-run-dir .\artifacts\training\m20\20260505T212518Z
Writes research_labels/vol_scaled/specialist_confirmation_adjudication/ from existing specialist conditional-usefulness artifacts only. The adjudication records PatchTST as `CONFIRMED_SELECTIVE_RANK_SLICE_RESEARCH_CANDIDATE` and NHITS as `SECONDARY_WATCHLIST_OR_WEAKER_CANDIDATE` unless stronger evidence exists. Overall status remains `RESEARCH_ONLY_NOT_PROMOTABLE` with `NO_RUNTIME_EFFECT`, `NOT_PROMOTABLE`, and `NO_PROFIT_CLAIM`. Next blocker is economic/policy evaluation, not another raw scoring run.

<!-- M20_SPECIALIST_EDGE_EVALUATOR -->
M20 generic specialist edge evaluator:
Command: python .\scripts\analyze_m20_specialist_edge.py --prediction-run-dir .\artifacts\training\m20\20260507T135017Z --label-source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-source score_only_confirmation
Writes `research_labels/vol_scaled/specialist_edge_evaluator/` from existing prediction and label artifacts only. The tool discovers `predictions_{model}_{source}.csv` files unless `--models` is supplied, joins by `symbol` and `interval_begin` with model-aware label support, evaluates top-k fractions and probability thresholds, and emits model, top-k, threshold, symbol, time, decision, next-action, report, manifest, and recommendation artifacts. It is generic across models and keeps status `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, and `NO_PROFIT_CLAIM`.

<!-- M20_COST_AWARE_SPECIALIST_POLICY_EVALUATOR -->
M20 cost-aware specialist policy evaluator:
Command: python .\scripts\analyze_m20_cost_aware_specialist_policy.py --prediction-run-dir .\artifacts\training\m20\20260507T135017Z --label-source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-source score_only_confirmation
Writes `research_labels/vol_scaled/cost_aware_specialist_policy_evaluator/` from existing artifacts only. The reusable evaluator applies identical top-k, threshold, edge-slice, symbol, month, and quarter diagnostic policy logic across discovered/requested models. It now supports `--economic-outcome-dir` and defaults to the label-source run's `economic_outcome_artifacts/`; economic metrics are computed only from safe economic outcome or label/evaluation artifacts. Outputs remain research-only with `NO_RUNTIME_EFFECT`, `NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, `NO_PROFIT_CLAIM`, and `NOT_BACKTEST`.

<!-- M20_ECONOMIC_OUTCOME_ARTIFACTS -->
M20 economic outcome artifacts:
Command: python .\scripts\build_m20_economic_outcomes.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z
Writes `research_labels/vol_scaled/economic_outcome_artifacts/` from safe evaluation/label/training-frame artifacts only. The builder keys rows by `symbol` and `interval_begin`, carries optional `fold_index`, and emits `future_return`, `gross_value_proxy`, `net_value_proxy`, fees, slippage, horizon, `long_only` direction, fee-exceedance label context, and triple-barrier label context when available. These artifacts are research/evaluation-only and do not change prediction exports, runtime, registry, promotion, trading, backtests, or profitability claims.

<!-- M20_COST_AWARE_POLICY_ADJUDICATION -->
M20 cost-aware policy adjudication:
Command: python .\scripts\write_m20_cost_aware_policy_adjudication.py --prediction-run-dir .\artifacts\training\m20\20260507T135017Z
Writes `research_labels/vol_scaled/cost_aware_policy_adjudication/` from existing cost-aware specialist policy outputs only. Current adjudication records PatchTST and NHITS as `SIGNAL_CONFIRMED_ECONOMICS_NEGATIVE`, pauses the NeuralForecast specialist policy path, and points next work to `MOVE_TO_GENERIC_STRATEGY_CONDITIONED_CANDIDATE_FACTORY`. No runtime, registry, promotion, training, scoring, backtest, trading, or profit claim behavior is changed.

<!-- M20_STRATEGY_CANDIDATE_FACTORY -->
M20 strategy-conditioned candidate factory:
Command: python .\scripts\build_m20_strategy_candidates.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
Writes `research_labels/vol_scaled/strategy_candidate_factory/` from existing training-frame, label, and safe economic-outcome artifacts only. The reusable factory evaluates MACD, RSI, range, volatility, return/reversal, and volume setup families with identical candidate-event logic, joins safe labels/economics by `symbol`, `interval_begin`, and optional `fold_index`, and keeps every output `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, `NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, and `NO_PROFIT_CLAIM`. It does not change runtime, registry, promotion, training, scoring, prediction exports, backtests, trading, or profit claims.

<!-- M20_STRATEGY_CANDIDATE_REFINEMENT -->
M20 strategy candidate refinement:
Command: python .\scripts\analyze_m20_strategy_candidate_refinement.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
Writes `research_labels/vol_scaled/strategy_candidate_refinement/` from existing strategy candidate and training-frame artifacts only. The analyzer applies one generic refinement pass across symbol, month, quarter, volatility, range, and volume slices, with sample-size and tail-loss diagnostics. Current result keeps all strategy candidates research-only and recommends `REFINE_STRATEGY_CANDIDATE_DEFINITIONS`; no runtime, registry, promotion, training, scoring, prediction export, backtest, trading, or profit claim behavior is changed.

<!-- M20_STRATEGY_SLICE_POLICY_EVALUATOR -->
M20 strategy slice policy evaluator:
Command: python .\scripts\analyze_m20_strategy_slice_policy.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
Writes `research_labels/vol_scaled/strategy_slice_policy_evaluator/` from existing strategy refinement artifacts only. The evaluator scores generic symbol, time, regime, and volume slice policies without building one-off strategy tools. Current result: 438 policy candidates, 434 economics-negative and 4 low-sample, recommendation `REFINE_STRATEGY_CANDIDATE_DEFINITIONS`. Outputs remain research-only with no runtime, registry, promotion, training, scoring, prediction export, backtest, trading, or profit-claim behavior.

<!-- M20_STRATEGY_MODEL_FACTORY_PLAN -->
M20 strategy-conditioned model factory plan:
Command: python .\scripts\plan_m20_strategy_model_factory.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
Writes `research_labels/vol_scaled/strategy_model_factory_plan/` as a design-only contract for a future reusable model factory. AutoGluon or any future model family is treated as a tournament/model factory, not a final universal model. Current blockers are `NO_APPROVED_STRATEGY_POLICY_CANDIDATE` and `MODEL_FACTORY_EXECUTION_NOT_APPROVED`; no training, scoring, runtime, registry, promotion, backtest, trading, or profit-claim behavior is changed.

<!-- M20_RESEARCH_CANDIDATE_COMPARATOR -->
M20 research candidate comparator:
Command: python .\scripts\compare_m20_research_candidates.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z
Writes `research_labels/vol_scaled/research_candidate_comparator/` from existing generic strategy and NeuralForecast specialist adjudication artifacts only. It compares model, strategy, policy, and model-factory candidate inputs under one research-only evidence contract. Current result: 470 candidates, no positive proxy research candidate, recommendation `PAUSE_CURRENT_M20_CANDIDATE_PATHS`, next action `REFINE_STRATEGY_CANDIDATE_DEFINITIONS`.

<!-- M20_RESEARCH_DASHBOARD -->
M20 research dashboard:
Command: python .\scripts\write_m20_research_dashboard.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z
Writes `research_labels/vol_scaled/m20_research_dashboard/` as a static handoff artifact from existing evidence only. Current decision: `PAUSE_CURRENT_M20_RESEARCH_PATHS`; recommendation `PAUSE_CURRENT_M20_RESEARCH_PATHS_AND_REFINE_STRATEGY_DEFINITIONS`; blockers `NO_POSITIVE_PROXY_RESEARCH_CANDIDATE` and `NO_RUNTIME_OR_PROMOTION_DECISION`. No runtime, registry, promotion, training, scoring, backtest, trading, or profit-claim behavior is changed.

<!-- M20_STRATEGY_CANDIDATE_REDESIGN_PLAN -->
M20 strategy candidate redesign plan:
Command: python .\scripts\plan_m20_strategy_candidate_redesign.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z
Writes `research_labels/vol_scaled/strategy_candidate_redesign_plan/` as a design-only v2 candidate-definition artifact. It proposes multi-condition, volatility-adjusted, range/volatility/volume composite, lower-turnover, abstention/HOLD, and tail-risk-aware definitions from existing safe features, while blocking regime-conditioned and trend-strength definitions until `regime_label` and `adx_14`-style features exist. It does not evaluate candidates, train, score, change runtime behavior, or claim profit.

<!-- M20_STRATEGY_CANDIDATE_V2_FACTORY -->
M20 strategy candidate v2 factory:
Command: python .\scripts\build_m20_strategy_candidates_v2.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
Writes `research_labels/vol_scaled/strategy_candidate_v2_factory/` from the redesign plan and safe training-frame features only. Current result: 8 definitions processed, 6 ready definitions generated 237,794 candidate rows, 2 definitions remain blocked by missing `regime_label` and `adx_14`, and all ready candidates are economics-negative under the current safe net-proxy artifact. Recommendation: `REFINE_OR_ADD_SAFE_FEATURES_FOR_V2_CANDIDATES`. No runtime, registry, promotion, training, scoring, prediction export, backtest, trading, or profit-claim behavior is changed.

Enriched-source command: python .\scripts\build_m20_strategy_candidates_v2.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --research-feature-dir .\artifacts\training\m20\20260506T054337Z\research_labels\vol_scaled\research_feature_enrichment
When `--research-feature-dir` is provided, the factory reads `research_features.csv`, `research_feature_columns.json`, and `blocked_features.csv` from the enrichment artifact while preserving the default training-frame mode. Current enriched result: 8 definitions processed, `regime_conditioned_momentum` and `trend_strength_filtered_momentum` newly unblocked, 387,036 candidate rows, all 8 candidates economics-negative, recommendation `REFINE_OR_ADD_SAFE_FEATURES_FOR_V2_CANDIDATES`.

<!-- M20_SAFE_FEATURE_AVAILABILITY -->
M20 safe feature availability audit:
Command: python .\scripts\audit_m20_safe_feature_availability.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --regime-thresholds-path .\artifacts\regime\m8\20260320T165813Z\thresholds.json
Writes `research_labels/vol_scaled/safe_feature_availability/` from existing M20 feature-frame and M8 threshold artifacts only. Current result: `regime_label` and `adx_14` are not already in the M20 frame, but both are marked safe-computable for a future research-only enrichment artifact; no feature engineering, runtime, registry, promotion, training, scoring, backtest, trading, or profit-claim behavior is changed.

<!-- M20_RESEARCH_FEATURE_ENRICHMENT -->
M20 research feature enrichment:
Command: python .\scripts\build_m20_research_feature_enrichment.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --regime-thresholds-path .\artifacts\regime\m8\20260320T165813Z\thresholds.json
Writes `research_labels/vol_scaled/research_feature_enrichment/` without mutating `training_frame/`. Current result: 312,494 enriched research rows, added `regime_label` from fixed M8 thresholds and causal per-symbol `adx_14`, blocked features 0, recommendation `RE_RUN_V2_STRATEGY_CANDIDATE_FACTORY_WITH_RESEARCH_FEATURES`. No runtime, registry, promotion, training, scoring, backtest, trading, or profit-claim behavior is changed.

<!-- M20_V2_REFINEMENT_RECOVERY -->
M20 v2 refinement recovery:
Commands:
python .\scripts\plan_m20_v2_refinement.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
python .\scripts\build_m20_strategy_candidate_v2_refined_definitions.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
python .\scripts\build_m20_strategy_candidates_v2.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --research-feature-dir .\artifacts\training\m20\20260506T054337Z\research_labels\vol_scaled\research_feature_enrichment --redesign-plan-dir .\artifacts\training\m20\20260506T054337Z\research_labels\vol_scaled\strategy_candidate_v2_refined_definitions --output-name strategy_candidate_v2_refined_factory
python .\scripts\write_m20_reframe.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
Writes `v2_refinement_plan/`, `strategy_candidate_v2_refined_definitions/`, `strategy_candidate_v2_refined_factory/`, and `m20_reframe/`. Current refined result: 4 refined definitions, 118,116 candidate rows, all 4 candidates economics-negative. M20 is reframed as research-only context-aware decision selection; next action `DESIGN_RESEARCH_ONLY_DECISION_POLICY_EVALUATOR`. No runtime, registry, promotion, training, scoring, backtest, trading, or profit claim behavior is changed.

<!-- M20_DECISION_POLICY_RECOVERY -->
M20 decision-policy recovery:
Commands:
python .\scripts\audit_m20_policy_inputs.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z
python .\scripts\evaluate_m20_decision_policies.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z
python .\scripts\audit_m20_policy_validation.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
python .\scripts\build_m20_trading_aware_labels.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
python .\scripts\evaluate_m20_decision_policies.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z --trading-aware-label-dir .\artifacts\training\m20\20260506T054337Z\research_labels\vol_scaled\trading_aware_labels --output-name trading_aware_policy_eval
python .\scripts\plan_m20_shadow_observer.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --policy-eval-dir .\artifacts\training\m20\20260506T054337Z\research_labels\vol_scaled\trading_aware_policy_eval

Writes `policy_input_availability_audit/`, `decision_policy_eval/`, `policy_validation_audit/`, `trading_aware_labels/`, `trading_aware_policy_eval/`, `trading_aware_policy_validation_audit/`, and `shadow_adaptation_observer_plan/`. Current result: policy inputs are available, 70 generic decision policies were evaluated, trading-aware labels wrote 312,485 rows, label-aware policy evaluation still found no adequate positive proxy policy, and the shadow observer plan recommends `PAUSE_M20_POLICY_ROUTE_AND_REDESIGN_INPUTS`. All artifacts remain `RESEARCH_ONLY`, `NO_RUNTIME_EFFECT`, `NOT_BACKTEST`, `NOT_RUNTIME_READY`, `NOT_PROMOTABLE`, and `NO_PROFIT_CLAIM`.

<!-- M20_INPUT_REDESIGN_RECOVERY -->
M20 input-redesign recovery:
Commands:
python .\scripts\analyze_m20_input_failures.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
python .\scripts\build_m20_research_input_catalogue.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z
python .\scripts\plan_m20_input_redesign.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
python .\scripts\build_m20_redesigned_research_inputs.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
python .\scripts\evaluate_m20_decision_policies.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --prediction-run-dir .\artifacts\training\m20\20260507T135017Z --research-input-dir .\artifacts\training\m20\20260506T054337Z\research_labels\vol_scaled\m20_redesigned_research_inputs --label-column fee_plus_slippage_exceedance_6 --output-name m20_redesigned_policy_eval
python .\scripts\write_m20_input_redesign_decision.py --source-run-dir .\artifacts\training\m20\20260506T054337Z --policy-eval-dir .\artifacts\training\m20\20260506T054337Z\research_labels\vol_scaled\m20_redesigned_policy_eval

Writes `m20_input_failure_analysis/`, `m20_research_input_catalogue/`, `m20_input_redesign_plan/`, `m20_redesigned_research_inputs/`, `m20_redesigned_policy_eval/`, and `m20_input_redesign_decision/`. The blocked 6/12-candle labels are now safe-computable research outcomes from existing enriched OHLCV features: `fee_plus_slippage_exceedance_6` has 312,476 usable rows and `fee_plus_slippage_exceedance_12` has 312,458 usable rows. The generic policy rerun remains economics-negative; final decision is `M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY`. No runtime, registry, promotion, training, scoring, backtest, trading, or profit claim behavior is changed.

<!-- M20_FINAL_RESEARCH_SUMMARY -->
M20 final research summary:
Command: python .\scripts\write_m20_final_research_summary.py --source-run-dir .\artifacts\training\m20\20260506T054337Z
Writes `research_labels/vol_scaled/m20_final_research_summary/` from existing M20 evidence only. Current terminal decision: `M20_POLICY_ROUTE_PAUSED_NO_POSITIVE_PROXY`; status `RESEARCH_ONLY_NEGATIVE_RESULT`; project route recommendation `KEEP_M20_PAUSED_AS_NEGATIVE_RESULT_AND_MOVE_TO_PLATFORM_MATURITY`. The negative result is intentional preserved evidence, not an implementation failure. No runtime, registry, promotion, training, scoring, backtest, trading, or profit claim behavior is changed.

<!-- M9_REGIME_INTEGRATION_AUDIT -->
M9 regime integration audit:
Command: python .\scripts\audit_m9_regime_integration.py
Writes `artifacts/platform_maturity/m9/regime_integration_audit/`.
Current audit result: `M9_REGIME_INTEGRATION_CONSOLIDATED`; gap count `0`;
recommendation `PROCEED_TO_M10_RISK_INTERFACE_AUDIT`. The audit adds a canonical
`RegimeContext` contract and confirms existing M9 read surfaces without changing
runtime, registry, promotion, trading, backtest, training, scoring, or M20 status.

<!-- M10_RISK_INTERFACE_AUDIT -->
M10 risk interface audit:
Command: python .\scripts\audit_m10_risk_interface.py
Writes `artifacts/platform_maturity/m10/risk_interface_audit/`.
Current audit result: `M10_RISK_INTERFACE_CONSOLIDATED`; gap count `0`;
recommendation `PROCEED_TO_M11_EXECUTION_INTERFACE_AUDIT`; next required action
`AUDIT_M11_EXECUTION_INTERFACE_WITH_RISK_AUTHORITY`. The audit confirms the M10
risk configuration contract, pure risk engine, RegimeContext/readiness boundary,
risk persistence, decision-trace enrichment, runner authority, and downstream
order-request surface without changing runtime, registry, promotion, trading,
backtest, training, scoring, or M20 status.

<!-- M11_EXECUTION_INTERFACE_AUDIT -->
M11 execution interface audit:
Command: python .\scripts\audit_m11_execution_interface.py
Writes `artifacts/platform_maturity/m11/execution_interface_audit/`.
Current audit result: `M11_EXECUTION_INTERFACE_CONSOLIDATED`; gap count `0`;
recommendation `PROCEED_TO_M12_GUARDED_LIVE_AUDIT`; next required action
`AUDIT_M12_GUARDED_LIVE_CONTROLS`. The audit confirms deterministic order
requests, idempotency keys, paper/shadow/live adapters, lifecycle persistence,
M10-to-M11 risk authority boundaries, and guarded-live submit gates without
changing runtime, registry, promotion, execution, trading, backtest, training,
scoring, or M20 status.
