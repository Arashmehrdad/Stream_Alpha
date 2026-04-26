# Docker

## Compose Profiles

`docker-compose.yml` defines these profiles:

- `dev`
- `paper`
- `shadow`
- `live`

All profiles include `redpanda`, `redpanda-console`, `postgres`, `config-check`, `producer`, and `features`. The deployed profiles also include `inference`, `trader`, and `dashboard`.

## Build and Run

Start the paper stack:

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  up -d --build
```

Build selected app services:

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  build config-check inference dashboard trader producer features
```

Stop the stack while preserving volumes:

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  down --remove-orphans
```

## Shared App Image

The app services share `streamalpha-app:latest` through the `x-streamalpha-app-build` anchor in `docker-compose.yml`.

This keeps separate containers and service commands while reducing duplicate image tags.

## Safe Cleanup

Do not delete volumes unless intentionally resetting PostgreSQL and Redpanda data.

Safe cleanup sequence:

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  down --remove-orphans

docker builder prune -af

docker image prune -af

docker compose `
  --profile paper `
  --env-file .env `
  up -d
```

Do not run:

```powershell
docker system prune --volumes
```

unless the goal is to delete local Postgres and Redpanda data.

## Inspect Disk Usage

```powershell
docker system df -v

docker image ls -a `
  --format "table {{.Repository}}`t{{.Tag}}`t{{.ID}}`t{{.Size}}"
```

Docker Desktop with WSL2 stores engine data by default under:

```text
C:\Users\[USERNAME]\AppData\Local\Docker\wsl
```

Docker Desktop storage can be moved from:

```text
Settings -> Resources -> Advanced -> Disk image location
```

The disk image location controls where Docker stores the Linux volume containing containers and images.

If C: space does not return after cleanup, the Docker WSL2 VHDX may not have compacted automatically. Stop Docker Desktop and WSL before compacting the VHDX. Do not manually delete `docker_data.vhdx`.

## Rebuild After Dependency Changes

After changing `requirements-runtime.txt`:

```powershell
docker compose `
  --profile paper `
  --env-file .env `
  build config-check inference dashboard trader producer features

docker compose `
  --profile paper `
  --env-file .env `
  up -d --force-recreate
```
