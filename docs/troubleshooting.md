# Troubleshooting

## Docker Disk Full

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| C: drive fills during Docker builds | Docker Desktop with WSL2 stores engine data by default under `C:\Users\[USERNAME]\AppData\Local\Docker\wsl` | Run safe Docker cleanup and consider moving Docker Desktop storage to another drive |
| `docker system df` shows large build cache | BuildKit cache retained dependency layers | Run `docker builder prune -af` if freeing disk matters more than fast rebuilds |
| Windows disk does not shrink after prune | Docker Desktop WSL VHDX did not compact automatically | Stop Docker Desktop and WSL before compacting the VHDX |

## Docker VHDX Not Shrinking

Symptom: Docker reports less data, but `docker_data.vhdx` remains large.

Likely cause: WSL virtual disks keep allocated size until compacted.

Docker Desktop with WSL2 stores engine data by default under:

```text
C:\Users\[USERNAME]\AppData\Local\Docker\wsl
```

Fix:

1. Stop Docker Desktop.
2. Run `wsl --shutdown`.
3. Compact the Docker VHDX with the Windows method available on the machine.
4. Restart Docker Desktop.

Do not manually delete `docker_data.vhdx`.

## Move Docker Desktop Storage

Docker Desktop storage can be moved from:

```text
Settings -> Resources -> Advanced -> Disk image location
```

The disk image location controls where Docker stores the Linux volume containing containers and images.

Moving the disk image location is the preferred long-term fix when Docker images and build cache should grow on another drive.

## Missing Python Package

Symptom: a container or local command fails with `ModuleNotFoundError`.

Likely cause: package is missing from the correct dependency file.

Fix:

- Runtime service dependency: add to `requirements-runtime.txt`.
- Training/research dependency: add to `requirements-training.txt`.
- Rebuild Docker only when runtime dependencies changed.

## Model Fails To Load

Symptom: `/health` reports `model_loaded=false` or startup validation fails.

Likely cause: missing registry entry, missing artifact, invalid artifact payload, or missing backend package.

Fix:

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  logs config-check inference
```

Then inspect:

```powershell
Get-Content artifacts/runtime/startup_report.json
```

## Compose Env Interpolation Warnings

Symptom: Compose prints warnings about unset variables.

Likely cause: `.env` is missing variables used by `docker-compose.yml`.

Fix:

```powershell
Copy-Item .env.example .env
```

Then set at least:

```text
STREAMALPHA_RUNTIME_PROFILE=paper
STREAMALPHA_TRADING_CONFIG_PATH=configs/paper_trading.paper.yaml
```

## Redpanda Connection Issues

Symptom: producer or features cannot connect to Kafka.

Likely cause: Redpanda is not healthy, wrong bootstrap server, or port conflict.

Fix:

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  ps redpanda

docker compose `
  --profile paper `
  --env-file .env `
  logs redpanda producer features
```

## Postgres Connection Issues

Symptom: inference returns 503, feature service cannot write, or trader cannot start.

Likely cause: Postgres not healthy, wrong credentials, or port conflict.

Fix:

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  ps postgres

docker compose `
  --profile paper `
  --env-file .env `
  logs postgres
```

## Port Conflicts

Symptom: Compose cannot bind ports.

Likely cause: another local service uses a default port.

Fix: change the matching `.env` port variable:

- `POSTGRES_EXTERNAL_PORT`
- `REDPANDA_CONSOLE_PORT`
- `REDPANDA_EXTERNAL_KAFKA_PORT`
- `REDPANDA_EXTERNAL_ADMIN_PORT`
- `REDPANDA_EXTERNAL_SCHEMA_REGISTRY_PORT`
- `REDPANDA_EXTERNAL_PANDAPROXY_PORT`

For inference and dashboard ports, update `docker-compose.yml` only in a deliberate config batch.
