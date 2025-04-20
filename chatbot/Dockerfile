FROM python:3.10.16-slim-bullseye

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

WORKDIR /chatbot

# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1
ENV UV_SYSTEM_PYTHON=1

# ADD *.toml *.lock .python-version ./
ADD *.toml *.lock ./
# RUN uv venv --seed &&\
#     uv sync --frozen
RUN uv sync --frozen

COPY *.py .

CMD uv run main.py
# CMD python main.py
