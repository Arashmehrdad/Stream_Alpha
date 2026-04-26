# Model Runtime

## Inference Service

The inference service runs `python -m app.inference` and creates a FastAPI app from `app/inference/main.py`.

It reads:

- Runtime metadata from `app/runtime/config.py`.
- Model metadata and registry entries through `app/training/registry.py`.
- Model artifacts through `joblib`.
- Feature rows from PostgreSQL.
- Regime runtime artifacts through `app/regime/live.py`.
- Startup validation from `artifacts/runtime/startup_report.json`.

## Model Loading

The visible runtime loader validates that the deserialized artifact is a dictionary with required keys:

- `model_name`
- `trained_at`
- `feature_columns`
- `expanded_feature_names`
- `model`

The loaded model must expose `predict_proba`.

If the model is a pretrained forecaster wrapper, the code validates its contract before runtime use.

## Current Champion Runtime Assumptions

`requirements-runtime.txt` confirms runtime support for:

- `autogluon.tabular==1.5.0`
- `catboost`
- `lightgbm`
- `xgboost`

`PLANS.md` records the current runtime as an AutoGluon-backed weak ensemble/profile state. Do not treat training metadata, comments, or config hooks as proof of a stronger model. A model counts only when it has real artifacts, registry/runtime discoverability, and runtime usability.

## Health Response

`GET /health` returns `HealthResponse`. Important fields include:

- `status`
- `runtime_profile`
- `execution_mode`
- `startup_validation_passed`
- `model_loaded`
- `model_name`
- `model_artifact_path`
- `regime_loaded`
- `database`
- `health_overall_status`
- `ensemble_status`
- `ensemble_roster_status`

## Failure Modes

| Failure | Likely cause | Operator action |
| --- | --- | --- |
| `model_loaded=false` | Missing or invalid model artifact | Check startup report and model registry paths |
| 500 from `/predict` | Artifact schema mismatch | Compare `feature_columns` against `feature_ohlc` schema |
| 404 from feature endpoints | No feature row for symbol/time | Check `features` service and `feature_ohlc` rows |
| 503 from signal/freshness | Database unavailable | Check `streamalpha-postgres` health and logs |
| Startup validation failed | Missing profile, config, live flags, model, or regime artifact | Inspect `artifacts/runtime/startup_report.json` |

