# Runtime vs Training

## Runtime Dependencies

`requirements-runtime.txt` is intentionally slim. It is used by `docker/app.Dockerfile` and contains the packages needed by the app services:

- Kafka and PostgreSQL clients: `aiokafka`, `asyncpg`.
- API and dashboard: `fastapi`, `uvicorn`, `streamlit`, `httpx`, `websockets`.
- Core ML/runtime packages: `numpy`, `scikit-learn`, `PyYAML`.
- AutoGluon runtime support: `autogluon.tabular==1.5.0`.
- Current confirmed backend packages: `catboost`, `lightgbm`, `xgboost`.

The Docker app image copies only runtime code, configs, dashboards, and scripts needed by the services.

## Training Dependencies

`requirements-training.txt` includes `-r requirements-runtime.txt` and adds heavier training and research packages:

- `autogluon.tabular[all]==1.5.0`
- `chronos-forecasting`
- `lightning`
- `neuralforecast`
- `sktime`
- `timesfm`
- `pytest`
- `pylint`
- `autopep8`
- `paramiko`

`requirements.txt` currently points to `requirements-training.txt` as the local development compatibility superset.

## Docker Image Slimming

`docker/app.Dockerfile` uses BuildKit syntax and a pip cache mount:

```dockerfile
# syntax=docker/dockerfile:1.7
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-runtime.txt
```

`docker-compose.yml` reuses one shared app image tag:

```yaml
image: streamalpha-app:latest
```

This reduces duplicate image tags for `config-check`, `producer`, `features`, `inference`, `trader`, and `dashboard`.

## Practical Rule

- Use runtime dependencies for Docker services and inference validation.
- Use training dependencies for model fitting, score-only runs, research experiments, and training tests.
- Do not add training-only packages to `requirements-runtime.txt` unless a runtime artifact cannot load without them.

