FROM python:3.10.16-slim-bullseye

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

WORKDIR /chatbot

ENV UV_SYSTEM_PYTHON=1

ADD *.toml *.lock ./

RUN uv sync --frozen

COPY *.py .

CMD uv run main.py
