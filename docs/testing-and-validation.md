# Testing and Validation

## Dependency Checks

```powershell
python -m pip install `
  --dry-run `
  --no-deps `
  -r requirements-runtime.txt

python -m pip install `
  --dry-run `
  --no-deps `
  -r requirements-training.txt
```

## Focused Runtime Tests

```powershell
python -m pytest `
  tests/test_inference_model_loader.py `
  tests/test_inference_api.py `
  tests/test_inference_service.py `
  -q
```

## Focused Training Tests

```powershell
python -m pytest `
  tests/test_training_service.py `
  tests/test_training_specialist_verdicts.py `
  tests/test_training_neuralforecast.py `
  tests/test_training_registry.py `
  tests/test_training_compare.py `
  -q
```

## Docker Validation

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  config --services

docker compose `
  --profile paper `
  --env-file .env `
  build config-check inference

docker compose `
  --profile paper `
  --env-file .env `
  up -d config-check inference dashboard trader
```

## Health Validation

```powershell
(Invoke-WebRequest `
  -UseBasicParsing `
  http://127.0.0.1:8000/health).Content
```

Expected pass criteria:

- HTTP request succeeds.
- `status` is `ok`.
- `startup_validation_passed` is true.
- `model_loaded` is true.
- `database` is healthy.

## Documentation Validation

Install docs dependencies from the docs dependency file:

```powershell
python -m pip install `
  -r requirements-docs.txt
```

Run a local Markdown link check if a repo checker exists. This repository does not currently include a named Markdown link-check script, but the docs-hardening batch used a small local Python check over README and `docs/*.md`.

Run the ASCII check used for docs hardening when practical. The docs-hardening batch used a local Python check over `mkdocs.yml`, `README.md`, and `docs/*.md`.

Build the MkDocs site:

```powershell
python -m mkdocs build
```
