FROM python:3.12-slim-bookworm

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_SYSTEM_PYTHON=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl ca-certificates gnupg lsb-release && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    chmod a+r /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
    "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y --no-install-recommends docker-ce-cli && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

# Загружаем модель эмбеддингов во время сборки
COPY save_models.py ./
RUN uv run python save_models.py

# Сохраняем кэш HuggingFace в образе
ENV HF_HOME=/usr/src/app/.cache/huggingface
RUN mkdir -p /usr/src/app/.cache/huggingface

COPY models /usr/src/app/models
COPY utils /usr/src/app/utils
COPY *.py /usr/src/app/
COPY README.md Makefile /usr/src/app/

# Устанавливаем переменную окружения для использования локальной модели
ENV EMBEDDING_MODEL_PATH=/usr/src/app/saved_models/multilingual-e5-large-wikiutmn

WORKDIR /usr/src/app

CMD ["tail", "-f", "/dev/null"]
