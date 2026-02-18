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

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY models /usr/src/app/models
COPY utils /usr/src/app/utils
COPY *.py /usr/src/app/
COPY README.md Makefile /usr/src/app/

WORKDIR /usr/src/app

CMD ["tail", "-f", "/dev/null"]
