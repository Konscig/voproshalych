# Smoke-сценарий для benchmarks (Локальный режим)

Полный цикл проверки: synthetic, manual и real-users режимы без Docker.

## Предварительные требования

- Python 3.12+
- Установленный UV: `curl -LsSf https://astral.sh/uv | sh`
- PostgreSQL доступен локально или по сети
- Переменные окружения в `.env` или `.env.docker`

## 0) Установка зависимостей

```bash
cd Submodules/voproshalych
uv sync
```

## 1) Подготовка окружения

Создайте `.env.docker` файл:

```bash
cd Submodules/voproshalych
cp .env.docker.example .env.docker
```

Обязательные переменные:
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- `MISTRAL_API`, `MISTRAL_MODEL`
- `BENCHMARKS_JUDGE_API_KEY` (или `JUDGE_API`)
- `EMBEDDING_MODEL_PATH` (или используется HF)

## 2) Подготовка БД

```bash
cd Submodules/voproshalych
uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump
```

## 3) Генерация эмбеддингов

```bash
cd Submodules/voproshalych

# Эмбеддинги для чанков (документов) — используется для RAG
uv run python benchmarks/generate_embeddings.py --chunks

# Проверить покрытие
uv run python benchmarks/generate_embeddings.py --check-coverage
```

## 4) Synthetic dataset + benchmark

### Генерация датасета

```bash
cd Submodules/voproshalych

# Synthetic — вопросы из чанков через LLM
# Тестовый режим (5 вопросов)
uv run python benchmarks/generate_dataset.py \
  --mode synthetic --max-questions 5

# Полный режим (все чанки)
uv run python benchmarks/generate_dataset.py \
  --mode synthetic --max-questions 10000

# Из реальных вопросов
# Тестовый режим (5 вопросов)
uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions --max-questions 5

# Полный режим (все вопросы)
uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions --max-questions 10000

# Только вопросы с оценкой 5
# Тестовый режим (5 вопросов)
uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions-score-5 --max-questions 5

# Полный режим (все вопросы с оценкой 5)
uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions-score-5 --max-questions 10000

# Экспорт для ручной аннотации
uv run python benchmarks/generate_dataset.py \
  --mode export-annotation --output benchmarks/data/dataset_for_annotation.json
```

### Запуск benchmark

```bash
cd Submodules/voproshalych

# Тестовый режим (5 вопросов)
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic --limit 5

# Полный режим (все вопросы)
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic
```

## 5) Manual dataset + benchmark

```bash
cd Submodules/voproshalych

# Экспорт для аннотации
uv run python benchmarks/generate_dataset.py \
  --mode export-annotation --output benchmarks/data/dataset_for_annotation.json

# Запуск на аннотированном датасете (тестовый)
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode manual \
  --manual-dataset benchmarks/data/dataset_for_annotation.json \
  --limit 5
```

## 6) Real users benchmark

```bash
cd Submodules/voproshalych

# Тестовый режим
uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 10 --top-k 10

# Полный режим
uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 10000 --top-k 10
```

## 7) Дашборд

```bash
cd Submodules/voproshalych
uv run python benchmarks/run_dashboard.py
```

Дашборд доступен по адресу: `http://localhost:7860`

## Очистка

```bash
cd Submodules/voproshalych

# Очистить отчёты
rm -rf benchmarks/reports/*.json benchmarks/reports/*.md

# Очистить историю запусков
rm -f benchmarks/reports/benchmark_runs.json

# Очистить датасеты
rm -rf benchmarks/data/dataset_*.json
```
