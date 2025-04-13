FROM python:3.12-slim-bookworm

RUN apt-get update && \
    apt-get install -y ffmpeg git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

EXPOSE 8000

WORKDIR /app

ADD *.toml *.lock /app/
RUN uv venv --seed &&\
    uv sync --frozen

ADD gigaam.py /app/
RUN uv run gigaam.py

ADD app.py /app/

CMD uv run fastapi dev --host 0.0.0.0
