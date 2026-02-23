# Модуль для оценки качества RAG-системы Вопрошалыч

Enterprise-grade система бенчмарков для оценки Retrieval, Generation,
End-to-End качества, качества LLM-судьи и аналитики источников данных.

---

## Обзор

Модуль `benchmarks` покрывает ключевые сценарии оценки:

- **Synthetic dataset**: автоматически сгенерированный golden standard.
- **Manual dataset**: экспертная разметка для строгой валидации.
- **Real user data**: источник продуктовой и доменной аналитики.

### Ключевые особенности

- ✅ **PostgreSQL + pgvector** как production-поиск для benchmark сценариев.
- ✅ **LLM-as-a-Judge** с режимами `direct` и `reasoned`.
- ✅ **Tier 0-3 + Judge + Judge Pipeline + UX**.
- ✅ **Source Analytics**: Vector Space, Chunk Utilization, Topic Coverage,
  Domain Analysis.
- ✅ **Model Sweep**: сравнение метрик по нескольким judge/generation моделям.
- ✅ **Локальный запуск через UV** (без benchmark-контейнера).

---

## Режим запуска

Текущий рабочий режим:

- инфраструктура (`db`, `db-migrate`) поднимается через
  `docker-compose.benchmarks.yml`;
- benchmark-скрипты запускаются локально через `uv run`.

Контейнерный запуск benchmark-скриптов не используется.

---

## Документация

| Документ | Описание |
|----------|----------|
| [CLI_REFERENCE.md](docs/CLI_REFERENCE.md) | Справочник по CLI командам |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Архитектура и детальный пайплайн |
| [SMOKE_SCENARIO_LOCAL.md](docs/SMOKE_SCENARIO_LOCAL.md) | Полный локальный smoke-цикл |
| [FULL_RUN_MODEL_LAB_GUIDE.md](docs/FULL_RUN_MODEL_LAB_GUIDE.md) | Финальный full-run и сравнение всех типов моделей |
| [MODEL_COMPARISON_GUIDE.md](docs/MODEL_COMPARISON_GUIDE.md) | Гайд по multi-model сравнениям |
| [manual_annotation_guide.md](docs/manual_annotation_guide.md) | Ручная аннотация датасетов |
| [METRICS.md](docs/METRICS.md) | Научное описание всех метрик |

---

## Быстрый старт

```bash
cd Submodules/voproshalych/benchmarks

make up
make install-local
make load-dump-local
make generate-embeddings-local
make generate-dataset-local
make run-benchmarks-local
make run-dashboard-local
```

Дашборд: `http://localhost:7860`

---

## Уровни оценки

### Tier 0: Intrinsic Embedding Quality

Метрики геометрии векторного пространства без `cluster_id/label`.

**Примеры метрик:** `avg_nn_distance`, `density_score`,
`effective_dimensionality`, `avg_pairwise_distance`.

### Tier 1: Retrieval Accuracy

Метрики поиска по `pgvector` + дополнительные consistency сигналы.

**Примеры метрик:** `hit_rate@k`, `mrr`, `ndcg@k`,
`retrieval_consistency`.

### Tier 2: Generation Quality

Оценка генерации при фиксированном контексте.

**Примеры метрик:** `avg_faithfulness`, `avg_answer_relevance`,
`avg_answer_correctness`, `response_semantic_consistency`, ROUGE/BLEU.

### Tier 3: End-to-End

Оценка полного production-like пайплайна retrieval + generation.

**Примеры метрик:** `avg_e2e_score`, `avg_semantic_similarity`,
`avg_dot_similarity`, `avg_euclidean_distance`, consistency, ROUGE/BLEU.

### Tier Judge

Оценка стабильности benchmark judge-модели.

### Tier Judge Pipeline

Оценка production judge (Yes/No) как классификатора.

### Tier UX

Оценка UX-прокси метрик: кэш, контекст, многотуровая согласованность.

### Real-users data

Используется как аналитический источник для topic coverage, utilization
и domain insights. В качестве отдельного benchmark-датасета для Tier 1/2/3
не используется.

---

## Дополнительные аналитические скрипты

- `visualize_vector_space.py` — UMAP 2D/3D.
- `analyze_chunk_utilization.py` — доля используемых чанков.
- `analyze_topic_coverage.py` — покрытие тематических кластеров.
- `analyze_real_users_domain.py` — доменная аналитика real-user корпуса.

---

## Структура

```
benchmarks/
├── data/
├── docs/
├── models/
├── reports/
├── utils/
├── analyze_chunk_utilization.py
├── analyze_real_users_domain.py
├── analyze_topic_coverage.py
├── dashboard.py
├── generate_dataset.py
├── generate_embeddings.py
├── load_database_dump.py
├── run_comprehensive_benchmark.py
├── run_dashboard.py
├── visualize_vector_space.py
└── Makefile
```

---

## Переменные окружения

Используются значения из `Submodules/voproshalych/.env` и `.env.docker`.

Ключевые переменные:

- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`,
  `POSTGRES_DB`
- `MISTRAL_API`, `MISTRAL_MODEL`
- `JUDGE_API`, `JUDGE_MODEL`
- `BENCHMARKS_JUDGE_API_KEY`, `BENCHMARKS_JUDGE_MODEL` (опционально)
- `EMBEDDING_MODEL_PATH`
