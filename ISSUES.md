# Проблемы и решения при запуске бенчмарков RAG

## Решённые проблемы

### 1. Локальный .env файл
**Проблема:** Файл `.env` не существовал в подмодуле, скрипты загружали `.env.docker`

**Решение:** Создать `.env` в подмодуле и изменить все скрипты:
- `benchmarks/generate_embeddings.py` → загрузка `.env`
- `benchmarks/generate_dataset.py` → загрузка `.env`
- `benchmarks/run_comprehensive_benchmark.py` → загрузка `.env`

### 2. Относительные импорты в qa
**Проблема:** `from config import Config` не работало при импорте `qa.main`

**Решение:** Изменить на `from qa.config import Config`:
- `qa/main.py`
- `qa/confluence_retrieving.py`

### 3. Путь к модели эмбеддингов
**Проблема:** Хардкод `saved_models/multilingual-e5-large-wikiutmn`

**Решение:** Использовать `Config.EMBEDDING_MODEL_PATH`

### 4. LLM-судья для Tier 1
**Проблема:** LLMJudge инициализировался даже для Tier 1

**Решение:** Передавать `judge=None` для Tier 1

## Нерешённые проблемы

### 1. Подключение к PostgreSQL из хоста

**Статус:** ⚠️ НЕ РЕШЕНО

**Проблема:**
- Подключение из хоста к PostgreSQL в контейнере Docker выдаёт ошибку
- `FATAL: role "postgres" does not exist` при попытке подключиться к `127.0.0.1:5432`
- Подключение изнутри контейнера работает корректно

**Диагностика:**
- ✅ Контейнер `virtassist-db` запущен и здоров
- ✅ PostgreSQL 14.4 слушает на всех адресах (`listen_addresses = '*'`)
- ✅ Пользователь `postgres` существует в контейнере
- ✅ База `virtassist` существует
- ✅ `pg_hba.conf` разрешает подключения с `127.0.0.1/32`
- ✅ Пинг до `127.0.0.1` работает
- ✅ Подключение изнутри контейнера к `127.0.0.1` работает
- ❌ Подключение с хоста не работает

**Испробованные решения:**
1. ✅ Создать `.env` в подмодуле
2. ✅ Остановить локальную PostgreSQL (launchctl unload)
3. ✅ Изменить `POSTGRES_HOST` на `127.0.0.1:5432`
4. ✅ Пересоздать контейнер PostgreSQL
5. ✅ Пересоздать все контейнеры docker-compose
6. ✅ Попробовать `127.0.0.1`
7. ✅ Попробовать `localhost`
8. ✅ Попробовать `172.18.0.1` (container IP)
9. ✅ Попробовать `db` (Docker hostname)
10. ✅ Попробовать `0.0.0.1` и `0.0.0.1:5432`
11. ✅ Попробовать `::1` (IPv6)

**Текущее состояние:**
- `.env` содержит `POSTGRES_HOST=127.0.0.1:5432`
- URI формируется корректно: `postgresql://postgres:postgres@127.0.0.1:5432/virtassist`
- Но подключение не работает

**Возможные причины:**
- Проблема с Docker Network Bridge
- Конфликт с IPv6/IPv4
- Проблема с psycopg2 в виртуальном окружении uv
- Firewall или проблема с портом

**Необходимые действия:**
1. ⚠️ Требуется дополнительная диагностика сети Docker
2. ⚠️ Возможно использовать Docker Desktop вместо Docker CLI
3. ⚠️ Проверить конфигурацию macOS firewall
4. ⚠️ Попробовать использовать Unix socket вместо TCP/IP

## Рабочие решения

### Tier 1 (Retrieval Accuracy) - работает! ✅

**Команда:**
\`\`\`bash
docker compose up -d
# Перейти в Submodules/voproshalych
uv run --directory . python benchmarks/run_comprehensive_benchmark.py --tier 1 --dataset benchmarks/data/final_test_dataset.json --skip-checks
\`\`\`

**Метрики:**
- hit_rate@1
- hit_rate@5
- hit_rate@10
- MRR

### Альтернативный способ запуска

Если проблемы с сетью Docker продолжаются, можно использовать PostgreSQL внутри Docker:

**Из контейнера:**
\`\`\`bash
docker exec -it virtassist-db psql -U postgres -d virtassist
\`\`\`

**Или запускать скрипты внутри контейнера:**
\`\`\`bash
docker exec virtassist-db python /path/to/script.py
\`\`\`

