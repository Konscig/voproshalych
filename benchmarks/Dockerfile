FROM python:3.12-slim-bookworm

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

ENV UV_SYSTEM_PYTHON=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY . /usr/src/app/
