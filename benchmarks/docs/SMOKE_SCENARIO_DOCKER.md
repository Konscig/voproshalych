# Smoke-сценарий для benchmarks (Docker режим)

Полный цикл проверки: synthetic, manual и real-users режимы с использованием `docker-compose.benchmarks.yml`.

## Предварительные требования

- Docker и Docker Compose установлены
- `.env.docker` файл настроен с переменными окружения
- Доступен PostgreSQL в сети Docker

## 0) Поднять контейнеры

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml up -d --build
```

**Запускаются:** db, db-migrate, benchmarks

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
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml exec -T benchmarks uv run python benchmarks/load_database_dump.py \
  --dump benchmarks/data/dump/virtassist_backup_20260213.dump
```

**Важно:** При загрузке дампа таблицы автоматически очищаются перед загрузкой.

### Вариант B: Только удаление таблиц

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml exec -T benchmarks uv run python benchmarks/load_database_dump.py \
  --drop-tables-only
```

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

# Вариант B: Ограниченный режим (20 вопросов)
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

# Вариант B: Ограниченный режим (20 вопросов)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions-score-5 --max-questions 20

# Вариант C: Полный режим — ВСЕ вопросы с оценкой 5
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions-score-5 --max-questions 10000

# ===========================================
# Режим 4: Экспорт для ручной аннотации
# ===========================================

docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode export-annotation --output benchmarks/data/dataset_for_annotation.json
```

Или через Makefile:
```bash
cd Submodules/voproshalych/benchmarks

# Генерация датасета (использует --max-questions 20 по умолчанию)
make COMPOSE_FILE=../docker-compose.benchmarks.yml generate-dataset
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
```

Или через Makefile:
```bash
cd Submodules/voproshalych/benchmarks

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

# Вариант B: 50 вопросов, score 5
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
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml run --rm -p 7860:7860 benchmarks uv run python benchmarks/run_dashboard.py
```

Или через Makefile:
```bash
cd Submodules/voproshalych/benchmarks
make COMPOSE_FILE=../docker-compose.benchmarks.yml run-dashboard
```

Дашборд доступен по адресу: `http://localhost:7860`

Ожидаемо:
- в `benchmarks/reports/` появляются свежие `rag_benchmark_*.json/.md`;
- в `benchmarks/reports/benchmark_runs.json` появляются новые записи;
- в дашборде видны метрики и графики по новым запускам.

## 8) Остановка контейнеров

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml down
```

Или через Makefile:
```bash
cd Submodules/voproshalych/benchmarks
make COMPOSE_FILE=../docker-compose.benchmarks.yml down
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
