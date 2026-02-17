FROM python:3.12-slim-bookworm

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY benchmarks/pyproject.toml benchmarks/uv.lock ./

RUN uv sync --frozen --no-dev

COPY benchmarks /usr/src/app/benchmarks
COPY qa /usr/src/app/qa

WORKDIR /usr/src/app
