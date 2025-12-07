FROM python:3.10-slim-bookworm

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y apt-transport-https curl cron && apt-get install -y ffmpeg git

RUN crontab -l | { cat; echo "5 18 * * * curl -X POST http://localhost:8080/reindex/"; } | crontab -

RUN pip install --upgrade pip
COPY ./requirements.txt .
RUN pip install -r requirements.txt

COPY ./save_models.py .
RUN python save_models.py

COPY . /usr/src/app/

EXPOSE 8080
