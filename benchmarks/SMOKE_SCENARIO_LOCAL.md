# Smoke-сценарий для benchmarks (Локальное тестирование)

Полный цикл проверки: synthetic, manual и real-users режимы без Docker.

## Предварительные требования

- Python 3.12+
- Установленный UV: `curl -LsSf https://astral.sh/uv | sh`
- PostgreSQL доступен локально или по сети
- Переменные окружения в `.env` или `.env.docker`

## 0) Установка зависимостей

```bash
cd Submodules/voproshalych/benchmarks
make install-local
```

Или напрямую:
```bash
cd Submodules/voproshalych
uv sync
```

## 1) Подготовка окружения

Создайте или обновите `.env.docker` файл:

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

### Вариант A: Загрузка дампа

```bash
cd Submodules/voproshalych/benchmarks
make load-dump-local
```

Или напрямую:
```bash
cd Submodules/voproshalych
uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump
```

### Вариант B: Только удаление таблиц

```bash
cd Submodules/voproshalych/benchmarks
make drop-tables-local
```

Или напрямую:
```bash
cd Submodules/voproshalych
uv run python benchmarks/load_database_dump.py --drop-tables-only
```

**Важно:** При загрузке дампа таблицы автоматически очищаются перед загрузкой.

## 3) Генерация эмбеддингов

```bash
cd Submodules/voproshalych/benchmarks
make generate-embeddings-local
```

Или напрямую:
```bash
cd Submodules/voproshalych
uv run python benchmarks/generate_embeddings.py --chunks
uv run python benchmarks/generate_embeddings.py --check-coverage
```

## 4) Synthetic dataset + benchmark

```bash
cd Submodules/voproshalych/benchmarks
make generate-dataset-local
make run-benchmarks-local
```

Или напрямую:
```bash
cd Submodules/voproshalych
uv run python benchmarks/generate_dataset.py --max-questions 20
uv run python benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic --limit 10
```

## 5) Manual dataset + benchmark

Создайте файл `benchmarks/data/manual_dataset_smoke.json` с 3-5 записями:

```json
[
  {
    "question": "Ваш вопрос",
    "ground_truth_answer": "Ожидаемый ответ",
    "relevant_chunk_ids": [1, 2, 3]
  }
]
```

```bash
cd Submodules/voproshalych
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode manual \
  --manual-dataset benchmarks/data/manual_dataset_smoke.json \
  --limit 5
```

## 6) Real users benchmark

```bash
cd Submodules/voproshalych
uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 50 --top-k 10
```

## 7) Проверка результатов

```bash
cd Submodules/voproshalych/benchmarks
make run-dashboard-local
```

Или напрямую:
```bash
cd Submodules/voproshalych
uv run python benchmarks/run_dashboard.py
```

Дашборд доступен по адресу: `http://localhost:7860`

Ожидаемо:
- в `benchmarks/data/reports/` появляются свежие `rag_benchmark_*.json/.md`;
- в PostgreSQL таблице `benchmark_runs` появляются новые записи с `dataset_type`;
- в дашборде видны метрики и графики по новым запускам.

## Проверка файлов

После прохождения сценария проверьте:

```bash
cd Submodules/voproshalych
ls -lh benchmarks/data/reports/
ls -lh benchmarks/data/dataset_*.json
```

Должны быть:
- JSON отчёты с результатами бенчмарков
- Markdown отчёты
- Созданные датасеты (если генерировались)

## Полный pipeline через Makefile

```bash
cd Submodules/voproshalych/benchmarks

# Установка
make install-local

# Подготовка БД
make load-dump-local

# Генерация эмбеддингов
make generate-embeddings-local

# Генерация датасета и запуск бенчмарков
make generate-dataset-local
make run-benchmarks-local

# Запуск дашборда
make run-dashboard-local
```

## Полный pipeline напрямую (без Makefile)

```bash
cd Submodules/voproshalych

# Установка зависимостей
uv sync

# Подготовка БД
uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump

# Генерация эмбеддингов
uv run python benchmarks/generate_embeddings.py --chunks
uv run python benchmarks/generate_embeddings.py --check-coverage

# Генерация датасета
uv run python benchmarks/generate_dataset.py --max-questions 20

# Запуск бенчмарков
uv run python benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic --limit 10

# Запуск дашборда
uv run python benchmarks/run_dashboard.py
```

## Устранение проблем

### Проблема: "Не удалось подключиться к БД"

Проверьте `.env.docker` файл:
```bash
cd Submodules/voproshalych
cat .env.docker | grep POSTGRES
```

Убедитесь что PostgreSQL запущен:
```bash
psql -h localhost -U postgres -d postgres -c "SELECT 1"
```

### Проблема: "Ошибка импорта qa.config"

Убедитесь что вы находитесь в директории `Submodules/voproshalych` при запуске команд:
```bash
cd Submodules/voproshalych
uv run python benchmarks/generate_embeddings.py --chunks
```

### Проблема: "Нет эмбеддингов в БД"

Запустите генерацию эмбеддингов:
```bash
cd Submodules/voproshalych
uv run python benchmarks/generate_embeddings.py --chunks
```

## Очистка

Для очистки после тестирования:

```bash
cd Submodules/voproshalych

# Очистить отчёты
rm -rf benchmarks/data/reports/*.json benchmarks/data/reports/*.md

# Очистить датасеты
rm -rf benchmarks/data/dataset_*.json benchmarks/data/manual_dataset_*.json

# Очистить кэш UV
rm -rf .venv/
```
