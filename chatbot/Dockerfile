FROM python:3.10.16-slim-bullseye

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ADD *.toml *.lock /usr/src/app/
RUN uv venv --seed &&\
    uv sync --frozen

COPY . .

CMD uv run main.py
