# Smoke-сценарий для benchmarks (Локальное тестирование)

Полный цикл проверки: synthetic, manual и real-users режимы без Docker.

## Предварительные требования

- Python 3.12+
- Установленный UV: `curl -LsSf https://astral.sh/uv | sh`
- PostgreSQL доступен локально или по сети
- Переменные окружения в `.env` или `.env.local`

## 0) Установка зависимостей

```bash
cd Submodules/voproshalych/benchmarks
uv sync
```

## 1) Подготовка окружения

Создайте `.env` файл или убедитесь что переменные заданы:

```bash
cp .env.docker.example .env
```

Обязательные переменные:
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- `MISTRAL_API`, `MISTRAL_MODEL`
- `BENCHMARKS_JUDGE_API_KEY` (или `JUDGE_API`)
- `EMBEDDING_MODEL_PATH` (или используеться HF)

## 2) Подготовка БД

```bash
cd Submodules/voproshalych/benchmarks
uv run python load_database_dump.py --dump data/dump/virtassist_backup_20260213.dump
```

Если нужно только удалить таблицы без загрузки дампа:

```bash
uv run python load_database_dump.py --drop-tables-only
```

**Важно:** При загрузке дампа таблицы автоматически очищаются перед загрузкой.

## 3) Генерация эмбеддингов

```bash
cd Submodules/voproshalych/benchmarks
uv run python generate_embeddings.py --chunks
uv run python generate_embeddings.py --check-coverage
```

## 4) Synthetic dataset + benchmark

```bash
cd Submodules/voproshalych/benchmarks
uv run python generate_dataset.py --max-questions 20
uv run python run_comprehensive_benchmark.py --tier all --mode synthetic --limit 10
```

## 5) Manual dataset + benchmark

Создайте файл `benchmarks/data/manual_dataset_smoke.json` с 3-5 записями.

```bash
cd Submodules/voproshalych/benchmarks
uv run python run_comprehensive_benchmark.py \
  --tier all --mode manual \
  --manual-dataset data/manual_dataset_smoke.json \
  --limit 5
```

## 6) Real users benchmark

```bash
cd Submodules/voproshalych/benchmarks
uv run python run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 50 --top-k 10
```

## 7) Проверка результатов

```bash
cd Submodules/voproshalych/benchmarks
uv run python run_dashboard.py
```

Дашборд доступен по адресу: `http://localhost:7860`

Ожидаемо:
- в `data/reports/` появляются свежие `rag_benchmark_*.json/.md`;
- в PostgreSQL таблице `benchmark_runs` появляются новые записи с `dataset_type`;
- в дашборде видны метрики и графики по новым запускам.

## Проверка файлов

После прохождения сценария проверьте:

```bash
ls -lh data/reports/
ls -lh data/dataset_*.json
```

Должны быть:
- JSON отчёты с результатами бенчмарков
- Markdown отчёты
- Созданные датасеты (если генерировались)

## Быстрые команды через Makefile

```bash
cd Submodules/voproshalych/benchmarks

# Установка
make install

# Бенчмарки
make generate-embeddings
make generate-dataset
make run-benchmarks

# Дашборд
make run-dashboard
```

## Устранение проблем

### Проблема: "Не удалось подключиться к БД"

Проверьте `.env` файл:
```bash
cat .env | grep POSTGRES
```

Убедитесь что PostgreSQL запущен:
```bash
psql -h localhost -U postgres -d postgres -c "SELECT 1"
```

### Проблема: "Ошибка импорта qa.config"

Убедитесь что вы находитесь в директории `Submodules/voproshalych` или добавили её в PYTHONPATH:

```bash
cd Submodules/voproshalych
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Проблема: "Нет эмбеддингов в БД"

Запустите генерацию эмбеддингов:
```bash
cd Submodules/voproshalych/benchmarks
uv run python generate_embeddings.py --chunks
```

## Очистка

Для очистки после тестирования:

```bash
cd Submodules/voproshalych/benchmarks

# Очистить отчёты
rm -rf data/reports/*.json data/reports/*.md

# Очистить датасеты
rm -rf data/dataset_*.json data/manual_dataset_*.json

# Очистить кэш UV
rm -rf .venv/
```
