FROM python:3.10.16-slim-bookworm

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y apt-transport-https curl cron

RUN crontab -l | { cat; echo "5 18 * * * curl -X POST http://localhost:8080/reindex/"; } | crontab -

COPY requirements.txt .

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_SYSTEM_PYTHON=1
RUN uv pip install -r requirements.txt

COPY save_models.py .
RUN python save_models.py

COPY . /usr/src/app/
