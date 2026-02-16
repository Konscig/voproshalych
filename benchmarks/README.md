# Модуль оценки качества RAG (Voproshalych)

Модуль `benchmarks/` предназначен для объективной оценки качества RAG-системы
по трём уровням: Retrieval, Generation и End-to-End.

## Что внутри

```text
benchmarks/
├── data/
│   └── dataset_YYYYMMDD_HHMMSS.json
├── reports/
│   ├── rag_benchmark_YYYYMMDD_HHMMSS.json
│   └── rag_benchmark_YYYYMMDD_HHMMSS.md
├── models/rag_benchmark.py
├── utils/
│   ├── embedding_generator.py
│   ├── llm_judge.py
│   └── report_generator.py
├── generate_embeddings.py
├── generate_dataset.py
├── run_comprehensive_benchmark.py
├── dashboard.py
├── run_dashboard.py
└── pyproject.toml
```

## Ключевые особенности

- Тестирование на реальной PostgreSQL (в Docker).
- LLM-as-a-Judge для Tier 2/3.
- Версионированные датасеты (`dataset_*.json`) без перезаписи истории.
- Сохранение результатов и в артефакты (`JSON/MD`), и в БД (`benchmark_runs`).
- Связь запуска с `dataset_file`, git-веткой, hash-коммита и автором запуска.
- Дашборд читает историю запусков из БД.

## Зависимости

Для модуля используется отдельный `benchmarks/pyproject.toml`.

```bash
cd Submodules/voproshalych/benchmarks

# Базовые зависимости модуля
uv sync

# Плюс зависимости дашборда
uv sync --extra dashboard
```

## Переменные окружения

Основные переменные берутся из `Submodules/voproshalych/.env` и
`Submodules/voproshalych/.env.docker` через `qa.config`.

Ключевые:

- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`,
  `POSTGRES_PASSWORD`
- `EMBEDDING_MODEL_PATH`
- `MISTRAL_MODEL`, `MISTRAL_API`
- `JUDGE_MODEL`, `JUDGE_API`
- `BENCHMARKS_JUDGE_API_KEY`, `BENCHMARKS_JUDGE_BASE_URL`,
  `BENCHMARKS_JUDGE_MODEL`

Дополнительно для metadata запуска бенчмарка:

- `BENCHMARK_GIT_BRANCH`
- `BENCHMARK_GIT_COMMIT_HASH`
- `BENCHMARK_RUN_AUTHOR`

## Полный рабочий цикл

### 1) Поднять инфраструктуру

```bash
cd Submodules/voproshalych
docker compose up -d --build
```

### 2) Сгенерировать эмбеддинги для чанков

```bash
cd Submodules/voproshalych
docker run --rm \
  --network voproshalych_chatbot-conn \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  virtassist/qa:latest \
  python benchmarks/generate_embeddings.py --chunks

docker run --rm \
  --network voproshalych_chatbot-conn \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  virtassist/qa:latest \
  python benchmarks/generate_embeddings.py --check-coverage
```

### 3) Сгенерировать датасет (по умолчанию 100 вопросов)

```bash
cd Submodules/voproshalych
docker run --rm \
  --network voproshalych_chatbot-conn \
  -v "$PWD:/workspace" \
  -w /workspace \
  virtassist/qa:latest \
  python benchmarks/generate_dataset.py --num-samples 100
```

Результат: новый versioned файл
`benchmarks/data/dataset_YYYYMMDD_HHMMSS.json`.

Опционально можно задать имя явно:

```bash
python benchmarks/generate_dataset.py --num-samples 100 --output benchmarks/data/dataset_custom.json
```

### 4) Запустить бенчмарк

```bash
cd Submodules/voproshalych
docker run --rm \
  --network voproshalych_chatbot-conn \
  -e BENCHMARK_GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)" \
  -e BENCHMARK_GIT_COMMIT_HASH="$(git rev-parse --short HEAD)" \
  -e BENCHMARK_RUN_AUTHOR="$(git config user.name)" \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  virtassist/qa:latest \
  python benchmarks/run_comprehensive_benchmark.py --tier all --limit 50
```

Опционально на конкретном датасете:

```bash
python benchmarks/run_comprehensive_benchmark.py --tier all --dataset benchmarks/data/dataset_YYYYMMDD_HHMMSS.json --limit 50
```

Важно: если `--dataset` не указан и дефолтный
`benchmarks/data/golden_dataset_synthetic.json` отсутствует, скрипт автоматически
возьмёт последний `benchmarks/data/dataset_*.json`.

### 5) Проверить артефакты

```bash
cd Submodules/voproshalych
ls benchmarks/reports
cat benchmarks/reports/rag_benchmark_*.md
```

## Что сохраняется после запуска

1. `benchmarks/reports/rag_benchmark_*.json`
2. `benchmarks/reports/rag_benchmark_*.md`
3. Запись в таблице БД `benchmark_runs`:
   - `timestamp`
   - `git_branch`
   - `git_commit_hash`
   - `run_author`
   - `dataset_file`
   - `tier_1_metrics`, `tier_2_metrics`, `tier_3_metrics`
   - `overall_status`

В JSON-артефакт также добавляются:

- `run_metadata`
- `overall_status`
- `dataset_file`

## Дашборд

Запуск:

```bash
cd Submodules/voproshalych/benchmarks
uv sync --extra dashboard
uv run python run_dashboard.py
```

Дашборд доступен на `http://localhost:7860` (если порт занят, Gradio выберет
следующий свободный).

Основные вкладки:

- `Latest Run` — сводная таблица ключевых метрик последнего запуска.
- `Metric History` — история отдельной метрики + baseline.
- `Tier Comparison` — сравнение Tier-метрик на одном графике + baseline.
- `Runs Registry` — плоский реестр запусков (Timestamp, Branch/Commit,
  T1/T2/T3 метрики).
- `Run Dataset` — просмотр датасета, связанного с выбранным запуском.
- `Справка` — формальные определения метрик и методологии LLM-as-a-Judge.

## CLI команды

### `generate_embeddings.py`

```bash
python benchmarks/generate_embeddings.py --all
python benchmarks/generate_embeddings.py --score 5
python benchmarks/generate_embeddings.py --score 1
python benchmarks/generate_embeddings.py --chunks
python benchmarks/generate_embeddings.py --check-coverage
python benchmarks/generate_embeddings.py --chunks --overwrite
```

### `generate_dataset.py`

```bash
python benchmarks/generate_dataset.py --num-samples 100
python benchmarks/generate_dataset.py --num-samples 100 --output benchmarks/data/dataset_custom.json
python benchmarks/generate_dataset.py --check-only --output benchmarks/data/dataset_custom.json
```

### `run_comprehensive_benchmark.py`

```bash
python benchmarks/run_comprehensive_benchmark.py --tier all
python benchmarks/run_comprehensive_benchmark.py --tier 1
python benchmarks/run_comprehensive_benchmark.py --tier 2
python benchmarks/run_comprehensive_benchmark.py --tier 3
python benchmarks/run_comprehensive_benchmark.py --tier all --limit 10
python benchmarks/run_comprehensive_benchmark.py --tier 1 --top-k 20
python benchmarks/run_comprehensive_benchmark.py --tier all --dataset benchmarks/data/dataset_custom.json
python benchmarks/run_comprehensive_benchmark.py --tier all --output-dir benchmarks/my_reports
python benchmarks/run_comprehensive_benchmark.py --tier all --skip-checks
```

## Архитектурная стратегия

Модуль `benchmarks` остаётся внутри `voproshalych`.

Почему:

- Глубокая связка с `qa.config` и SQLAlchemy-моделями (`Chunk`,
  `QuestionAnswer`, `BenchmarkRun`).
- Вынос в отдельный репозиторий усложнит поддержку и синхронизацию схем/логики.
- Текущая форма оптимальна по YAGNI.

Future-proof:

- При реальной потребности переиспользования модуль можно упаковать в отдельный
  Python package без изменения продуктового пайплайна.

## Troubleshooting

### `OperationalError: could not translate host name "db"`

Локальный запуск не видит docker hostname `db`.

Решение:

- Локально задайте в `.env`: `POSTGRES_HOST=localhost` (или IP Docker-хоста).
- Внутри Docker оставляйте `POSTGRES_HOST=db`.

### `ModuleNotFoundError: No module named 'openai'`

Установите зависимости в `benchmarks/.venv`:

```bash
cd Submodules/voproshalych/benchmarks
uv sync
```

### Дашборд не видит датасет для run

Проверьте, что `dataset_file` в `benchmark_runs` указывает на существующий файл
в `benchmarks/data/`.

### Порт дашборда занят

Если `7860` занят, Gradio стартует на следующем свободном порту (например,
`7861`).
