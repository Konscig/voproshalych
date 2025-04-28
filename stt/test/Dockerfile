FROM python:3.12-slim-bookworm

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

WORKDIR /app

ENV UV_SYSTEM_PYTHON=1

ADD *.toml *.lock /app/
RUN uv sync --frozen

COPY *.py .

CMD uv run telegram-bot.py
# CMD uv run vk-bot.py
