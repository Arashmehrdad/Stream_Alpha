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

* **M9** adds regime-aware live signal logic on top of M4.
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
* **M20** still exposes ensemble surfaces and persisted profile truth, and the active profile is now a current AutoGluon-only generalist roster. It must still be judged on real candidate breadth, role diversity, regime-slice value, and economics after costs. On those terms it remains active but weak.
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

For local AutoGluon challenger training, use the repo-native operator flow instead of raw Docker profile reasoning:

* `./scripts/prepare_m7_training.ps1`
* `./scripts/start_m7_training.ps1`

The prepare script checks the checked-in M7 config, AutoGluon availability, PostgreSQL reachability, and `feature_ohlc` readiness. If the training source is missing or not yet large enough for the configured walk-forward split, it starts the existing `dev` stack path to populate features and then prints the next recommended command.

The prepare script also reports whether optional `fastai` breadth is actually usable for AutoGluon, not just installed. Missing or broken `fastai` breadth is still not treated as a blocker for the authoritative M7 AutoGluon path.

The start script runs the same readiness checks, forces the local training process onto a repo-local same-drive temp root for AutoGluon and Ray, then invokes the authoritative training command `python -m app.training --config .\configs\training.m7.json`. During the run it shows a PowerShell progress heartbeat with elapsed time, the configured AutoGluon time budget, the latest discovered model, and the current best model when available. After the run it prints the newest artifact directory plus the `summary.json` winner and acceptance flags to inspect.

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
