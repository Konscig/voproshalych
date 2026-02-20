# CLI Reference: Команды модуля бенчмарков

Полная документация CLI команд для работы с бенчмарками RAG-системы.

---

## Содержание

1. [load_database_dump.py](#load_database_dumppy)
2. [generate_embeddings.py](#generate_embeddingspy)
3. [generate_dataset.py](#generate_datasetpy)
4. [run_comprehensive_benchmark.py](#run_comprehensive_benchmarkpy)
5. [run_dashboard.py](#run_dashboardpy)

---

## load_database_dump.py

Загрузка дампа PostgreSQL или удаление таблиц.

**Файл:** `benchmarks/load_database_dump.py`

**Парсинг аргументов:** функция `main()`, строки 190-217

### Аргументы

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--dump` | str | None | Путь к файлу дампа `.dump` |
| `--dump-dir` | str | None | Директория с дампами для поиска последнего |
| `--drop-tables-only` | flag | False | Только удалить таблицы без загрузки дампа |

### Примеры

```bash
# Загрузить конкретный дамп
uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump

# Найти и загрузить последний дамп из директории
uv run python benchmarks/load_database_dump.py --dump-dir benchmarks/data/dump

# Только удалить таблицы
uv run python benchmarks/load_database_dump.py --drop-tables-only
```

---

## generate_embeddings.py

Генерация эмбеддингов для чанков и вопросов.

**Файл:** `benchmarks/generate_embeddings.py`

**Парсинг аргументов:** функция `main()`, строки 36-71

### Аргументы

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--all` | flag | False | Генерировать эмбеддинги для всех вопросов |
| `--score` | int | None | Генерировать для вопросов с оценкой (1 или 5) |
| `--chunks` | flag | False | Генерировать эмбеддинги для всех чанков |
| `--check-coverage` | flag | False | Проверить покрытие эмбеддингами |
| `--overwrite` | flag | False | Перезаписывать существующие эмбеддинги |

### Примеры

```bash
# Генерация для чанков
uv run python benchmarks/generate_embeddings.py --chunks

# Проверка покрытия
uv run python benchmarks/generate_embeddings.py --check-coverage

# Генерация для вопросов с оценкой 5
uv run python benchmarks/generate_embeddings.py --score 5

# Генерация для всех вопросов
uv run python benchmarks/generate_embeddings.py --all

# Перезапись существующих эмбеддингов
uv run python benchmarks/generate_embeddings.py --chunks --overwrite
```

---

## generate_dataset.py

Генерация синтетического датасета для бенчмарков.

**Файл:** `benchmarks/generate_dataset.py`

**Парсинг аргументов:** функция `main()`, строки 190-244

### Аргументы

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--max-questions` | int | 500 | Максимальное количество пар вопрос-ответ |
| `--num-samples` | int | None | Устаревший алиас для `--max-questions` |
| `--output` | str | None | Путь для сохранения (по умолчанию versioned) |
| `--check-only` | flag | False | Только проверить существующий датасет |
| `--skip-existing-dataset` | str | None | Путь к существующему датасету для дополнения |
| `--generation-attempts` | int | 3 | Попытки генерации для одного чанка |

### Примеры

```bash
# Генерация 20 вопросов
uv run python benchmarks/generate_dataset.py --max-questions 20

# Генерация с кастомным путём
uv run python benchmarks/generate_dataset.py --max-questions 300 --output benchmarks/data/custom_dataset.json

# Инкрементальное дополнение существующего датасета
uv run python benchmarks/generate_dataset.py --max-questions 500 --skip-existing-dataset benchmarks/data/dataset_20260216_124845.json

# Проверка существующего датасета
uv run python benchmarks/generate_dataset.py --check-only --output benchmarks/data/custom_dataset.json
```

---

## run_comprehensive_benchmark.py

Запуск комплексных бенчмарков RAG-системы.

**Файл:** `benchmarks/run_comprehensive_benchmark.py`

**Парсинг аргументов:** функция `main()`, строки 325-411

### Аргументы

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--tier` | str | "all" | Уровень бенчмарка: "1", "2", "3", "all" |
| `--mode` | str | "synthetic" | Режим: "synthetic", "manual", "real-users" |
| `--dataset` | str | "benchmarks/data/golden_dataset_synthetic.json" | Путь к датасету |
| `--manual-dataset` | str | None | Путь к manual датасету (для --mode manual) |
| `--limit` | int | None | Ограничение количества записей |
| `--top-k` | int | 10 | Количество результатов для поиска (Tier 1) |
| `--output-dir` | str | "benchmarks/reports" | Директория для результатов |
| `--skip-checks` | flag | False | Пропустить проверку prerequisites |
| `--real-score` | int | 5 | Фильтр score для real-users режима |
| `--real-limit` | int | 500 | Лимит вопросов для real-users режима |
| `--real-latest` | flag | False | Брать последние вопросы по created_at |

### Режимы

#### synthetic
Использует автоматически сгенерированный датасет.

```bash
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic \
  --dataset benchmarks/data/dataset_20260216_124845.json \
  --limit 50
```

#### manual
Использует экспертно размеченный датасет.

```bash
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode manual \
  --manual-dataset benchmarks/data/manual_dataset_20260217_101500.json
```

#### real-users
Использует реальные вопросы из таблицы `question_answer`.

```bash
uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users \
  --real-score 5 \
  --real-limit 500 \
  --top-k 10
```

### Tier-уровни

| Tier | Что проверяет | Метрики |
|------|---------------|---------|
| 0 | Intrinsic Embedding Quality | avg_intra_cluster_sim, avg_inter_cluster_dist, silhouette_score |
| 1 | Retrieval Accuracy | HitRate@K, MRR, NDCG@K, Recall@K, Precision@K |
| 2 | Generation Quality | avg_faithfulness, avg_answer_relevance, avg_rouge1_f, avg_rougeL_f, avg_bleu |
| 3 | End-to-End | avg_e2e_score, avg_semantic_similarity, avg_rouge1_f, avg_bleu |
| judge | LLM-as-a-Judge Quality | consistency_score, error_rate, avg_latency_ms |
| ux | User Experience Quality | cache_hit_rate, context_preservation, multi_turn_consistency |
| all | Все уровни | Все метрики |

---

## run_dashboard.py

Запуск интерактивного дашборда для просмотра результатов.

**Файл:** `benchmarks/run_dashboard.py`

**Парсинг аргументов:** файл `benchmarks/dashboard.py`, функция `main()`, строки 801-831

### Аргументы

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--port` | int | None | Порт для дашборда (по умолчанию первый свободный от 7860) |

### Примеры

```bash
# Запуск на автоматическом порту
uv run python benchmarks/run_dashboard.py

# Запуск на конкретном порту
uv run python benchmarks/run_dashboard.py --port 8080
```

Дашборд доступен по адресу: `http://localhost:7860` (или указанный порт)

---

## Makefile команды

Альтернативный способ запуска через Makefile.

**Файл:** `benchmarks/Makefile`

### Локальный режим

| Команда | Описание |
|---------|----------|
| `make install-local` | Установить зависимости через uv sync |
| `make load-dump-local` | Загрузить дамп БД |
| `make drop-tables-local` | Удалить таблицы БД |
| `make generate-embeddings-local` | Сгенерировать эмбеддинги |
| `make generate-dataset-local` | Сгенерировать датасет (20 вопросов) |
| `make run-benchmarks-local` | Запустить бенчмарки (synthetic, 10 вопросов) |
| `make run-dashboard-local` | Запустить дашборд локально |

### Docker режим

| Команда | Описание |
|---------|----------|
| `make COMPOSE_FILE=../docker-compose.benchmarks.yml up` | Поднять стек |
| `make COMPOSE_FILE=../docker-compose.benchmarks.yml down` | Остановить стек |
| `make COMPOSE_FILE=../docker-compose.benchmarks.yml load-dump` | Загрузить дамп |
| `make COMPOSE_FILE=../docker-compose.benchmarks.yml generate-embeddings` | Сгенерировать эмбеддинги |
| `make COMPOSE_FILE=../docker-compose.benchmarks.yml generate-dataset` | Сгенерировать датасет (500 вопросов) |
| `make COMPOSE_FILE=../docker-compose.benchmarks.yml run-benchmarks` | Запустить бенчмарки |
| `make COMPOSE_FILE=../docker-compose.benchmarks.yml run-dashboard` | Запустить дашборд |
| `make COMPOSE_FILE=../docker-compose.benchmarks.yml ps` | Статус сервисов |
| `make COMPOSE_FILE=../docker-compose.benchmarks.yml logs` | Логи benchmarks |

---

## Связанные документы

- [RUN_MODES_LOCAL_VS_DOCKER.md](RUN_MODES_LOCAL_VS_DOCKER.md) — описание режимов
- [ARCHITECTURE.md](ARCHITECTURE.md) — архитектура модуля
- [SMOKE_SCENARIO_LOCAL.md](SMOKE_SCENARIO_LOCAL.md) — инструкции для локального режима
- [SMOKE_SCENARIO_DOCKER.md](SMOKE_SCENARIO_DOCKER.md) — инструкции для Docker режима
