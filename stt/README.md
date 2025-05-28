# Микросервис STT

## Установка

### Локальная установка

1. Инициализация проекта и создание виртуального окружения:
```sh
uv init
uv venv --seed
```

2. Установка зависимостей:
```sh
uv add dotenv ffmpeg-python requests aiogram httpx
uv pip install git+https://github.com/salute-developers/GigaAM.git
```

3. Для разработки с FastAPI:
```sh
uv init --app
uv run fastapi dev
```

## Тестирование

### Запуск тестов

Запуск всех тестов:
```sh
uv run pytest
```

### Установка тестовых зависимостей

Для установки всех зависимостей, включая тестовые:
```sh
uv pip install -e ".[test]"
```

## API Эндпоинты

### POST /transcribe/
Преобразует аудиофайл в текст.

**Запрос:**
- Метод: POST
- Content-Type: multipart/form-data
- Тело: аудиофайл (имя поля формы: 'file')

**Ответ:**
```json
{
    "status": "success",
    "transcription": "Распознанный текст"
}
```

**Пример использования curl:**
```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/transcribe/' \
  -H 'accept: application/json' \
  -F 'file=@/путь/к/вашему/аудиофайлу'
```

### GET /
Эндпоинт проверки работоспособности.

**Ответ:**
```json
{
    "message": "hello, world!"
}
```

## Развертывание в Docker

### Сборка образов

Сборка сервиса:
```sh
docker buildx build -t webmasha/stt-gam:0.1.0 --platform linux/arm64,linux/amd64 --push .
```

### Запуск с Docker

Запуск сервиса:
```sh
docker run --name stt -it --rm -p 8000:8000 webmasha/stt-gam:0.1.0 uv run fastapi dev --host 0.0.0.0
```
