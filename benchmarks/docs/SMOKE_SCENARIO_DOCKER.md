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

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml exec -T benchmarks uv run python benchmarks/load_database_dump.py \
  --dump benchmarks/data/dump/virtassist_backup_20260213.dump
```

**Важно:** При загрузке дампа таблицы автоматически очищаются перед загрузкой.

## 3) Генерация эмбеддингов

```bash
cd Submodules/voproshalych

# Эмбеддинги для чанков (документов) — используется для RAG
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --chunks

# Проверить покрытие
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --check-coverage
```

## 4) Synthetic dataset + benchmark

### Генерация датасета

```bash
cd Submodules/voproshalych

# Synthetic — вопросы из чанков через LLM
# Тестовый режим (5 вопросов)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode synthetic --max-questions 5

# Полный режим (все чанки с эмбеддингами)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode synthetic --max-questions 10000

# Из реальных вопросов пользователей
# Тестовый режим (5 вопросов)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions --max-questions 5

# Полный режим (все вопросы)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions --max-questions 10000

# Только вопросы с оценкой 5
# Тестовый режим (5 вопросов)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions-score-5 --max-questions 5

# Полный режим (все вопросы с оценкой 5)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions-score-5 --max-questions 10000

# Экспорт для ручной аннотации
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode export-annotation --output benchmarks/data/dataset_for_annotation.json
```

### Запуск benchmark

```bash
cd Submodules/voproshalych

# Тестовый режим (5 вопросов)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic --limit 5

# Полный режим (все вопросы)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic
```

## 5) Manual dataset + benchmark

```bash
cd Submodules/voproshalych

# Экспорт для аннотации
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py \
  --mode export-annotation --output benchmarks/data/dataset_for_annotation.json

# Запуск на аннотированном датасете (тестовый)
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode manual \
  --manual-dataset benchmarks/data/dataset_for_annotation.json \
  --limit 5
```

## 6) Real users benchmark

```bash
cd Submodules/voproshalych

# Тестовый режим
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 10 --top-k 10

# Полный режим
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 10000 --top-k 10
```

## 7) Дашборд

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml run --rm -p 7860:7860 benchmarks uv run python benchmarks/run_dashboard.py
```

Дашборд доступен по адресу: `http://localhost:7860`

## 8) Остановка

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml down
```

## Пересборка

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml build --no-cache benchmarks
```
