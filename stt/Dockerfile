FROM python:3.12.10-slim-bookworm

RUN apt-get update && \
    apt-get install -y ffmpeg git curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

EXPOSE 8000

WORKDIR /stt

ENV UV_SYSTEM_PYTHON=1
ENV HUGGINGFACE_HUB_CACHE=/huggingface_cache
ENV HF_HOME=/huggingface_cache

ADD *.toml *.lock *.md ./
RUN uv sync --frozen

ADD download_model.py .
ARG HF_TOKEN
RUN --mount=type=cache,target=/huggingface_cache,sharing=locked \
    HF_TOKEN=$HF_TOKEN uv run python download_model.py

ADD am.py .
ADD app.py .

CMD ["uv", "run", "fastapi", "dev", "--host", "0.0.0.0"]
