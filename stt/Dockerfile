FROM python:3.12-slim-bookworm

RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

EXPOSE 8000

WORKDIR /stt

ENV UV_SYSTEM_PYTHON=1

ADD *.toml *.lock ./
RUN uv sync --frozen

ADD am.py .
RUN uv run am.py

ADD app.py .

CMD uv run fastapi dev --host 0.0.0.0
