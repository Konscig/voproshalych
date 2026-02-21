# Smoke-сценарий для benchmarks (Контейнеризация с Docker)

Полный цикл проверки: synthetic, manual и real-users режимы с использованием `docker-compose.benchmarks.yml`.

## Предварительные требования

- Docker и Docker Compose установлены
- `.env.docker` файл настроен с переменными окружения
- Доступен PostgreSQL в сети Docker

## 0) Поднять контейнеры

### Вариант A: Базовый стек (рекомендуется для бенчмарков)

```bash
cd Submodules/voproshalych/benchmarks
make COMPOSE_FILE=../docker-compose.benchmarks.yml up
```

Или напрямую:
```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml up -d --build
```

**Запускаются:** db, db-migrate, benchmarks

### Вариант B: С API сервисами (qa, adminpanel)

```bash
cd Submodules/voproshalych/benchmarks
make COMPOSE_FILE=../docker-compose.benchmarks.yml up-api
```

Или напрямую:
```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml --profile api up -d --build
```

**Запускаются:** + qa, adminpanel

### Вариант C: Полный стек

```bash
cd Submodules/voproshalych/benchmarks
make COMPOSE_FILE=../docker-compose.benchmarks.yml up-full
```

Или напрямую:
```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml --profile full up -d --build
```

**Запускаются:** Все сервисы (db, qa, chatbot, adminpanel, max, benchmarks)

### Проверка статуса

Дождитесь здорового состояния всех сервисов:

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml ps
```

Или через Makefile:
```bash
cd Submodules/voproshalych/benchmarks
make COMPOSE_FILE=../docker-compose.benchmarks.yml ps
```

Ожидайте пока все сервисы будут в статусе `healthy` или `running`.

## 1) Проверка .env.docker

Убедитесь что `.env.docker` настроен:

```bash
cd Submodules/voproshalych
cat .env.docker
```

Обязательные переменные:
- `MISTRAL_API`, `MISTRAL_MODEL`
- `BENCHMARKS_JUDGE_API_KEY` (или `JUDGE_API`)
- `EMBEDDING_MODEL_PATH` (путь к модели или используется HF)

## 2) Подготовка БД

### Вариант A: Загрузка дампа

```bash
cd Submodules/voproshalych/benchmarks
make COMPOSE_FILE=../docker-compose.benchmarks.yml load-dump
```

Или напрямую:
```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml exec -T benchmarks uv run python benchmarks/load_database_dump.py \
  --dump benchmarks/data/dump/virtassist_backup_20260213.dump
```

### Вариант B: Только удаление таблиц

```bash
cd Submodules/voproshalych/benchmarks
make COMPOSE_FILE=../docker-compose.benchmarks.yml drop-tables
```

Или напрямую:
```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml exec -T benchmarks uv run python benchmarks/load_database_dump.py \
  --drop-tables-only
```

**Важно:** При загрузке дампа таблицы автоматически очищаются перед загрузкой.

## 3) Генерация эмбеддингов

### Варианты генерации

```bash
cd Submodules/voproshalych

# Вариант A: Эмбеддинги для чанков (документов) — используется для RAG
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --chunks

# Вариант B: Эмбеддинги для всех вопросов
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --all

# Вариант C: Эмбеддинги для вопросов с оценкой 5 (только полезные)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --score 5

# Вариант D: Эмбеддинги для вопросов с оценкой 1
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --score 1

# Вариант E: Перезаписать существующие эмбеддинги (любой из выше)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --all --overwrite

# Вариант F: Проверить покрытие эмбеддингами
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --check-coverage
```

Или через Makefile:
```bash
cd Submodules/voproshalych/benchmarks

# Эмбеддинги для чанков (рекомендуется)
make COMPOSE_FILE=../docker-compose.benchmarks.yml generate-embeddings

# Проверить покрытие
make COMPOSE_FILE=../docker-compose.benchmarks.yml check-coverage
```

**Примечание:** Эмбеддинги будут кэшироваться в volume `model-cache`.

## 4) Synthetic dataset + benchmark

### Генерация датасета

```bash
cd Submodules/voproshalych

# ===========================================
# Режим 1: Synthetic — вопросы из чанков через LLM
# ===========================================

# Вариант A: Тестовый режим (5 вопросов) — проверить что работает
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode synthetic --max-questions 5

# Вариант B: Ограниченный режим (20 вопросов) — текущая команда
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode synthetic --max-questions 20

# Вариант C: Полный режим — ВСЕ чанки с эмбеддингами
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode synthetic --max-questions 10000

# ===========================================
# Режим 2: Из реальных вопросов пользователей
# ===========================================

# Вариант A: Тестовый режим (5 вопросов)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions --max-questions 5

# Вариант B: Ограниченный режим (20 вопросов) — текущая команда
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions --max-questions 20

# Вариант C: Полный режим — ВСЕ реальные вопросы пользователей
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions --max-questions 10000

# ===========================================
# Режим 3: Только вопросы с оценкой 5 (полезные)
# ===========================================

# Вариант A: Тестовый режим (5 вопросов)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions-score-5 --max-questions 5

# Вариант B: Ограниченный режим (20 вопросов) — текущая команда
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions-score-5 --max-questions 20

# Вариант C: Полный режим — ВСЕ вопросы с оценкой 5
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions-score-5 --max-questions 10000

# ===========================================
# Режим 4: Экспорт для ручной аннотации
# ===========================================

docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode export-annotation --output benchmarks/data/dataset_realq_20260221_135841.json
```

### Запуск benchmark

```bash
cd Submodules/voproshalych

# ===========================================
# Тестовый режим (быстрая проверка)
# ===========================================

# Synthetic, 5 вопросов, все tier
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic --limit 5

# Synthetic, 10 вопросов — текущая команда
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic --limit 10

# ===========================================
# Полный режим (все вопросы из датасета)
# ===========================================

# Synthetic, все вопросы, все tier (финальный замер)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic

# Synthetic, все вопросы, только tier 0-1 (без judge, быстрее)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier 1 --mode synthetic

# Synthetic, все вопросы, полный pipeline с judge (медленно, но полноценно)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic
```

### Makefile варианты

```bash
cd Submodules/voproshalych/benchmarks

# Генерация датасета (использует --max-questions 20 по умолчанию)
make COMPOSE_FILE=../docker-compose.benchmarks.yml generate-dataset

# Запуск бенчмарков (использует --limit 10 по умолчанию)
make COMPOSE_FILE=../docker-compose.benchmarks.yml run-benchmarks

# Полный замер (без лимитов)
make COMPOSE_FILE=../docker-compose.benchmarks.yml run-benchmarks-full
```

## 5) Manual dataset + benchmark (ручная аннотация)

Рекомендуемый workflow для ручной аннотации:

```bash
cd Submodules/voproshalych

# 1. Экспорт существующего датасета для аннотации
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode export-annotation --output benchmarks/data/dataset_for_annotation.json

# 2. Отредактируйте файл вручную - добавьте метки качества
# Поля для заполнения: is_question_ok, is_answer_ok, is_chunk_ok, notes

# 3. Запуск бенчмарка на аннотированном датасете

# Вариант A: Тестовый режим (5 вопросов)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode manual \
  --manual-dataset benchmarks/data/dataset_for_annotation.json \
  --limit 5

# Вариант B: Полный режим — все вопросы из файла аннотации
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode manual \
  --manual-dataset benchmarks/data/dataset_for_annotation.json
```

Также можно использовать `--mode manual` с любым JSON файлом в новом формате.

## 6) Real users benchmark

```bash
cd Submodules/voproshalych

# ===========================================
# Тестовый режим
# ===========================================

# Вариант A: 10 вопросов, score 5
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 10 --top-k 10

# Вариант B: 50 вопросов, score 5 — текущая команда
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 50 --top-k 10

# ===========================================
# Полный режим
# ===========================================

# Вариант A: ВСЕ вопросы с оценкой 5
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 10000 --top-k 10

# Вариант B: ВСЕ вопросы с оценкой 1 (негативные)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 1 --real-limit 10000 --top-k 10

# Вариант C: Последние вопросы (по дате создания)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-latest --real-limit 100 --top-k 10
```

## 7) Проверка результатов через дашборд

```bash
cd Submodules/voproshalych/benchmarks
make COMPOSE_FILE=../docker-compose.benchmarks.yml run-dashboard
```

Или напрямую:
```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml run --rm -p 7860:7860 benchmarks uv run python benchmarks/run_dashboard.py
```

Дашборд доступен по адресу: `http://localhost:7860`

Ожидаемо:
- в `benchmarks/reports/` появляются свежие `rag_benchmark_*.json/.md`;
- в `benchmarks/reports/benchmark_runs.json` появляются новые записи;
- в дашборде видны метрики и графики по новым запускам.

## 8) Просмотр логов

Для просмотра логов сервиса benchmarks:

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml logs -f benchmarks
```

Или через Makefile:
```bash
cd Submodules/voproshalych/benchmarks
make COMPOSE_FILE=../docker-compose.benchmarks.yml logs
```

Для просмотра логов всех сервисов:
```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml logs -f
```

## 9) Проверка volumes

Проверьте что данные сохраняются в volumes:

```bash
docker volume inspect model-cache benchmarks-reports benchmarks-data
```

## 10) Остановка контейнеров

```bash
cd Submodules/voproshalych/benchmarks
make COMPOSE_FILE=../docker-compose.benchmarks.yml down
```

Или напрямую:
```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml down
```

Для остановки с удалением volumes:
```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml down -v
```

## Полный pipeline через Makefile

```bash
cd Submodules/voproshalych/benchmarks

# Управление стеком
make COMPOSE_FILE=../docker-compose.benchmarks.yml up
make COMPOSE_FILE=../docker-compose.benchmarks.yml ps

# Подготовка БД
make COMPOSE_FILE=../docker-compose.benchmarks.yml load-dump

# Генерация эмбеддингов
make COMPOSE_FILE=../docker-compose.benchmarks.yml generate-embeddings

# Генерация датасета и запуск бенчмарков
make COMPOSE_FILE=../docker-compose.benchmarks.yml generate-dataset
make COMPOSE_FILE=../docker-compose.benchmarks.yml run-benchmarks

# Запуск дашборда
make COMPOSE_FILE=../docker-compose.benchmarks.yml run-dashboard

# Логи и статус
make COMPOSE_FILE=../docker-compose.benchmarks.yml logs
make COMPOSE_FILE=../docker-compose.benchmarks.yml ps

# Остановка
make COMPOSE_FILE=../docker-compose.benchmarks.yml down
```

## Полный pipeline напрямую (без Makefile)

```bash
cd Submodules/voproshalych

# ===========================================
# Подготовка
# ===========================================

# Поднять стек
docker compose -f docker-compose.benchmarks.yml up -d --build

# Проверить статус
docker compose -f docker-compose.benchmarks.yml ps

# Подготовка БД
docker compose -f docker-compose.benchmarks.yml exec -T benchmarks uv run python benchmarks/load_database_dump.py \
  --dump benchmarks/data/dump/virtassist_backup_20260213.dump

# ===========================================
# Генерация эмбеддингов
# ===========================================

# Эмбеддинги для чанков (обязательно для RAG)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --chunks

# Эмбеддинги для вопросов с оценкой 5 (для бенчмарков)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --score 5

# Проверка покрытия
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --check-coverage

# ===========================================
# Генерация датасета — ВАРИАНТЫ
# ===========================================

# Тестовый режим (5 вопросов, быстро проверить)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode synthetic --max-questions 5

# Ограниченный режим (20 вопросов)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode synthetic --max-questions 20

# Полный режим (все чанки)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode synthetic --max-questions 10000

# Из реальных вопросов (ограниченно)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions --max-questions 20

# Из вопросов с оценкой 5 (полный)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions-score-5 --max-questions 10000

# ===========================================
# Запуск бенчмарков — ВАРИАНТЫ
# ===========================================

# Тестовый режим (5 вопросов)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic --limit 5

# Ограниченный режим (10 вопросов)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic --limit 10

# Полный режим (все вопросы из датасета)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic

# Real users (ограниченно)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 50 --top-k 10

# Real users (полный)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 10000 --top-k 10

# ===========================================
# Дашборд и остановка
# ===========================================

# Запуск дашборда
docker compose -f docker-compose.benchmarks.yml run --rm -p 7860:7860 benchmarks uv run python benchmarks/run_dashboard.py

# Остановка
docker compose -f docker-compose.benchmarks.yml down
```

## Устранение проблем

### Проблема: "Контейнер не запускается"

Проверьте логи:
```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml logs benchmarks
```

Проверьте зависимости:
```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml ps
```

### Проблема: "Нет соединения с БД"

Убедитесь что сервис `db` healthy:
```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml ps
```

Проверьте переменные окружения в `.env.docker`:
```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml exec benchmarks env | grep POSTGRES
```

### Проблема: "Модели не кэшируются"

Проверьте volume `model-cache`:
```bash
docker volume ls | grep model-cache
```

Кэш хранится в `model-cache:/root/.cache/huggingface`.

### Проблема: "Нет отчётов"

Проверьте volume `benchmarks-reports`:
```bash
docker run --rm -v benchmarks-reports:/data alpine ls -lh /data
```

## Мониторинг ресурсов

Для мониторинга ресурсов контейнеров:
```bash
docker stats
```

Для мониторинга только сервиса benchmarks:
```bash
docker stats virtassist-benchmarks
```

## Пересборка образа

Если нужно пересобрать образ benchmarks:
```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml build --no-cache benchmarks
```

Для пересборки с очисткой всех кэшей:
```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml build --no-cache --pull benchmarks
docker system prune -f
```

## Поведение контейнера benchmarks

**CMD в Dockerfile:** `tail -f /dev/null`

**Почему это нужно:**
- Контейнер benchmarks предназначен для ручного выполнения команд через `docker compose exec`
- Без CMD контейнер бы завершался сразу после старта
- `tail -f /dev/null` поддерживает контейнер в состоянии `running` бесконечно

**Использование:**
- Для запуска команд: `docker compose exec benchmarks uv run python ...`
- Для запуска дашборда: `docker compose run --rm -p 7860:7860 benchmarks uv run python benchmarks/run_dashboard.py`

**Важно:** Контейнер не является сервисом-демоном, а интерактивной средой для выполнения задач.

## Политики перезапуска сервисов

| Сервис | Policy | Поведение |
|--------|--------|-----------|
| db | unless-stopped | Автоматический перезапуск |
| qa | unless-stopped | Автоматический перезапуск |
| chatbot | unless-stopped | Автоматический перезапуск |
| adminpanel | unless-stopped | Автоматический перезапуск |
| max | unless-stopped | Автоматический перезапуск |
| **benchmarks** | **unless-stopped** | **Контейнер остаётся running для exec команд** |

**Важно:** Сервис `benchmarks` имеет `restart: unless-stopped`, что позволяет:
- Контейнеру оставаться в состоянии `running` после запуска
- Выполнять команды через `docker compose exec`
- Запускать дашборд через `make run-dashboard`
