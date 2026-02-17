# Модуль для оценки качества RAG-системы Вопрошалыч

Enterprise-grade система бенчмарков для оценки Retrieval, Generation и End-to-End
качества RAG-пайплайна на реальных данных PostgreSQL и LLM-as-a-Judge.

## Обзор

Модуль `benchmarks` покрывает три рабочих сценария:
- **Synthetic dataset**: автоматически сгенерированный golden standard.
- **Manual dataset**: экспертная разметка для строгой валидации.
- **Real user data**: retrieval-метрики на реальных вопросах пользователей.

### Ключевые особенности

- ✅ **Single Source of Truth**: работа с реальной PostgreSQL базой
- ✅ **LLM-as-a-Judge**: оценка quality-метрик через judge-model
- ✅ **Три tier-уровня**: Retrieval, Generation, End-to-End
- ✅ **Manual + Real Users режимы**: отдельные пайплайны для академичной оценки
- ✅ **Версионированные артефакты**: JSON/Markdown отчёты + `benchmark_runs`

## Структура проекта

```
benchmarks/
├── data/
│   ├── dataset_YYYYMMDD_HHMMSS.json
│   ├── manual_dataset_YYYYMMDD_HHMMSS.json
│   └── dataset_errors_YYYYMMDD_HHMMSS.json
├── docs/
│   └── manual_annotation_guide.md
├── models/
│   ├── rag_benchmark.py
│   └── real_queries_benchmark.py
├── reports/
│   ├── rag_benchmark_*.json
│   └── rag_benchmark_*.md
├── utils/
│   ├── llm_judge.py
│   ├── evaluator.py
│   └── embedding_generator.py
├── Makefile
├── dashboard.py
├── generate_embeddings.py
├── generate_dataset.py
├── load_database_dump.py
├── run_comprehensive_benchmark.py
└── run_dashboard.py
```
 
## Smoke-сценарии тестирования

Для быстрого тестирования системы бенчмарков доступны два Smoke-сценария:

### Локальное тестирование

Полный сценарий для локального запуска без Docker:
- 📄 [SMOKE_SCENARIO_LOCAL.md](SMOKE_SCENARIO_LOCAL.md) - локальное тестирование с `uv run`
- Требуется только PostgreSQL и Python 3.12+
- Все команды выполняются локально через UV

### Контейнеризация с Docker

Полный сценарий для работы в Docker окружении:
- 📄 [SMOKE_SCENARIO_DOCKER.md](SMOKE_SCENARIO_DOCKER.md) - тестирование с `docker-compose.benchmarks.yml`
- Изолированный стек с отдельными volumes для кэша, отчётов и данных
- Все команды выполняются внутри контейнеров

### Выбор сценария

| Сценарий | Когда использовать | Окружение |
|-----------|-------------------|-------------|
| LOCAL | Разработка, отладка, быстрые проверки | Локальный Python + PostgreSQL |
| DOCKER | CI/CD, интеграционное тестирование, продакшн | Docker + Docker Compose |

## Компоненты системы

### 1) LLM-судья (`utils/llm_judge.py`)

- `generate_question_from_chunk()` — генерация вопроса и ground truth ответа
- `evaluate_faithfulness()` — оценка фактичности (1..5)
- `evaluate_answer_relevance()` — оценка релевантности (1..5)
- `evaluate_e2e_quality()` — итоговая оценка ответа (1..5)

### 2) Генерация эмбеддингов (`generate_embeddings.py`)

Генерирует эмбеддинги для `QuestionAnswer` и `Chunk`.

### 3) Генерация synthetic dataset (`generate_dataset.py`)

Новая стратегия генерации:
- попытка генерации для **каждого чанка** с эмбеддингом и непустым текстом;
- стратегия **1 чанк -> 1 вопрос** до достижения `--max-questions`;
- retry на уровне скрипта + retry в `LLMJudge`;
- отдельный отчёт `dataset_errors_*.json` с причинами ошибок;
- поддержка инкрементального дополнения через `--skip-existing-dataset`.

### 4) Комплексный раннер (`run_comprehensive_benchmark.py`)

Поддерживаемые режимы:
- `--mode synthetic` (по умолчанию)
- `--mode manual`
- `--mode real-users`

## Три уровня тестирования

### Tier 1: Retrieval Accuracy

**Процедура:**
1. Для каждого вопроса рассчитывается эмбеддинг запроса.
2. Выполняется vector search по `Chunk.embedding.cosine_distance(...)`.
3. Рассчитываются hit-rate и классические IR-метрики.

В коде: `benchmarks/models/rag_benchmark.py`, `benchmarks/utils/evaluator.py`.

**Метрики:** `HitRate@K`, `Recall@K`, `Precision@K`, `MRR`, `NDCG@K`.

$$
\mathrm{HitRate@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbf{1}[\exists d \in \mathrm{TopK}(q): d \in G_q]
$$

$$
\mathrm{Recall@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\mathrm{TopK}(q) \cap G_q|}{|G_q|}
$$

$$
\mathrm{Precision@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\mathrm{TopK}(q) \cap G_q|}{K}
$$

$$
\mathrm{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\mathrm{rank}_q}
$$

$$
\mathrm{NDCG@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{\mathrm{DCG@K}(q)}{\mathrm{IDCG@K}(q)}
$$

### Tier 2: Generation Quality

**Процедура:**
1. Для вопроса собирается релевантный контекст (по `chunk_text`,
   `relevant_chunk_ids`, `chunk_id` или `relevant_urls`).
2. Генерируется ответ через реальный pipeline `qa.main.get_answer`.
3. LLM Judge оценивает фактичность и релевантность.

**Метрики:** `avg_faithfulness`, `avg_answer_relevance`.

$$
\mathrm{avg\_faithfulness} = \frac{1}{N} \sum_{i=1}^{N} s_i,
\quad s_i \in \{1,2,3,4,5\}
$$

$$
\mathrm{avg\_answer\_relevance} = \frac{1}{N} \sum_{i=1}^{N} s_i,
\quad s_i \in \{1,2,3,4,5\}
$$

Шкала 1..5 трактуется как **ordinal scale** (упорядоченная, не строго линейная).

### Tier 3: End-to-End

**Процедура:**
1. Выполняется retrieval top-1 для вопроса.
2. Генерируется финальный ответ на retrieved контексте.
3. Judge оценивает `E2E score` относительно `ground_truth_answer`.
4. Считается косинусная близость эмбеддингов системного и эталонного ответа.

**Метрики:** `avg_e2e_score`, `avg_semantic_similarity`.

$$
\mathrm{avg\_e2e\_score} = \frac{1}{N} \sum_{i=1}^{N} s_i,
\quad s_i \in \{1,2,3,4,5\}
$$

$$
\mathrm{cos\_sim}(u, v) = \frac{u \cdot v}{\|u\|\,\|v\|}
$$

## CLI команды

### `load_database_dump.py`

```bash
uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump
uv run python benchmarks/load_database_dump.py --dump-dir benchmarks/data/dump
uv run python benchmarks/load_database_dump.py --drop-tables-only
```

**Важно:** При загрузке дампа таблицы автоматически очищаются перед загрузкой. Флаг `--drop-tables-only` используется для удаления таблиц без загрузки дампа.

### `generate_embeddings.py`

```bash
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --chunks
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --all
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --score 5
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --check-coverage
```

### `generate_dataset.py`

```bash
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py --max-questions 500
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py --max-questions 300 --output benchmarks/data/dataset_custom.json
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py --max-questions 500 --skip-existing-dataset benchmarks/data/dataset_20260216_124845.json
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py --check-only --output benchmarks/data/dataset_custom.json
```

### `run_comprehensive_benchmark.py`

```bash
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic --dataset benchmarks/data/dataset_20260216_124845.json
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py --tier all --mode manual --manual-dataset benchmarks/data/manual_dataset_20260217_101500.json
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py --mode real-users --real-score 5 --real-limit 500 --top-k 10
```

### `run_dashboard.py`

```bash
docker compose -f docker-compose.benchmarks.yml run --rm -p 7860:7860 benchmarks uv run python benchmarks/run_dashboard.py
```

## Запуск с docker-compose.benchmarks.yml

Для работы с бенчмарками используется отдельный docker-compose файл `docker-compose.benchmarks.yml`.

### Описание

Этот файл включает в себя:
- Все сервисы основного приложения (db, db-migrate, qa, chatbot, adminpanel, max)
- Дополнительный сервис `benchmarks` с портом 7860 для дашборда
- Отдельные volumes для reports, data и cache бенчмарков

### Преимущества

- **Изолированный стек**: Полная среда для тестирования бенчмарков
- **Отдельные volumes**: Данные бенчмарков хранятся отдельно от данных приложения
- **Управление зависимостями**: Сервис benchmarks использует UV для управления зависимостями
- **Единая точка запуска**: Все сервисы поднимаются одной командой

### Быстрый старт

```bash
# Поднять стек с бенчмарками
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml up -d --build

# Остановить
docker compose -f docker-compose.benchmarks.yml down
```

👉 **Подробные инструкции**: [SMOKE_SCENARIO_DOCKER.md](SMOKE_SCENARIO_DOCKER.md)

## Быстрые шорткаты (docker compose)

Из `Submodules/voproshalych` с использованием `docker-compose.benchmarks.yml`:

```bash
uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --chunks
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py --max-questions 500
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic
docker compose -f docker-compose.benchmarks.yml run --rm -p 7860:7860 benchmarks uv run python benchmarks/run_dashboard.py
```

### Через Makefile

```bash
cd Submodules/voproshalych/benchmarks
make install
make load-dump
make generate-embeddings
make generate-dataset
make run-benchmarks
make run-dashboard
make help
```

Доступные команды Makefile:
- `up` - поднять основной стек
- `down` - остановить основной стек
- `up-benchmarks` - поднять стек с бенчмарками
- `down-benchmarks` - остановить стек с бенчмарками
- `ps` - показать статус сервисов
- `logs` - показать логи сервиса benchmarks

## Архитектура

```mermaid
sequenceDiagram
    participant User
    participant Benchmarks as benchmarks CLI/dashboard
    participant QAService as qa.get_answer
    participant Postgres as PostgreSQL
    participant JudgeLLM as LLMJudge

    User->>Benchmarks: run_comprehensive_benchmark.py (tier=all, mode=synthetic/manual)
    Benchmarks->>Postgres: SELECT Chunk / BenchmarkRun / QuestionAnswer
    Benchmarks->>QAService: get_answer(question, context)
    QAService->>Postgres: vector retrieval (cosine_distance)
    QAService-->>Benchmarks: generated answer + retrieved chunks
    Benchmarks->>JudgeLLM: faithfulness/relevance/e2e evaluation
    JudgeLLM-->>Benchmarks: scores 1..5
    Benchmarks->>Postgres: INSERT INTO benchmark_runs
    User->>Benchmarks: run_dashboard.py
    Benchmarks->>Postgres: SELECT benchmark_runs
    Benchmarks-->>User: interactive metrics dashboard
    ```

## Полезные ссылки

- `benchmarks/docs/manual_annotation_guide.md`
- `benchmarks/dashboard.py`
- `benchmarks/run_comprehensive_benchmark.py`
