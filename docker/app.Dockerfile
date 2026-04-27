# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/workspace

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements-runtime.txt

COPY app ./app
COPY configs ./configs
COPY dashboards ./dashboards
COPY scripts ./scripts

CMD ["python", "-m", "app.ingestion.main"]
