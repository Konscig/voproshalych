# Модуль для оценки качества RAG-системы Вопрошалыч

Enterprise-grade система бенчмарков для оценки Retrieval, Generation и End-to-End качества RAG-пайплайна.

---

## Обзор

Модуль `benchmarks` покрывает три рабочих сценария:

- **Synthetic dataset**: автоматически сгенерированный golden standard
- **Manual dataset**: экспертная разметка для строгой валидации
- **Real user data**: retrieval-метрики на реальных вопросах пользователей

### Ключевые особенности

- ✅ **Single Source of Truth**: работа с реальной PostgreSQL базой
- ✅ **LLM-as-a-Judge**: оценка quality-метрик через judge-model
- ✅ **Семь tier-уровней**: Tier 0 (Embedding), Tier 1 (Retrieval), Tier 2 (Generation), Tier 3 (End-to-End), Tier Judge (Qwen), Tier Judge Pipeline (Mistral), Tier UX
- ✅ **Manual + Real Users режимы**: отдельные пайплайны для академичной оценки
- ✅ **Версионированные артефакты**: JSON/Markdown отчёты + `benchmark_runs`

---

## Документация

| Документ | Описание |
|----------|----------|
| [CLI_REFERENCE.md](docs/CLI_REFERENCE.md) | Справочник по CLI командам |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Архитектура с Mermaid диаграммами |
| [RUN_MODES_LOCAL_VS_DOCKER.md](docs/RUN_MODES_LOCAL_VS_DOCKER.md) | Сравнение режимов работы |
| [SMOKE_SCENARIO_LOCAL.md](docs/SMOKE_SCENARIO_LOCAL.md) | Инструкции для локального режима |
| [SMOKE_SCENARIO_DOCKER.md](docs/SMOKE_SCENARIO_DOCKER.md) | Инструкции для Docker режима |
| [manual_annotation_guide.md](docs/manual_annotation_guide.md) | Руководство по ручной разметке |
| [CODE_DUPLICATION_ANALYSIS.md](docs/CODE_DUPLICATION_ANALYSIS.md) | Анализ дублирования кода |

---

## Быстрый старт

### Локальный режим

```bash
cd Submodules/voproshalych/benchmarks

make install-local
make load-dump-local
make generate-embeddings-local
make generate-dataset-local
make run-benchmarks-local
make run-dashboard-local
```

### Docker режим

```bash
cd Submodules/voproshalych/benchmarks

make COMPOSE_FILE=../docker-compose.benchmarks.yml up
make COMPOSE_FILE=../docker-compose.benchmarks.yml load-dump
make COMPOSE_FILE=../docker-compose.benchmarks.yml generate-embeddings
make COMPOSE_FILE=../docker-compose.benchmarks.yml generate-dataset
make COMPOSE_FILE=../docker-compose.benchmarks.yml run-benchmarks
make COMPOSE_FILE=../docker-compose.benchmarks.yml run-dashboard
```

---

## Уровни тестирования

### Tier 0: Intrinsic Embedding Quality

Оценка внутреннего качества эмбеддингов через кластерный анализ.

**Метрики:** `avg_intra_cluster_sim`, `avg_inter_cluster_dist`, `silhouette_score`

### Tier 1: Retrieval Accuracy

Оценка качества векторного поиска через pgvector.

**Метрики:** `HitRate@K`, `Recall@K`, `Precision@K`, `MRR`, `NDCG@K`

### Tier 2: Generation Quality

Оценка качества генерации ответов при идеальном контексте.

**Метрики:** `avg_faithfulness`, `avg_answer_relevance` (LLM Judge 1-5), `avg_rouge1_f`, `avg_rougeL_f`, `avg_bleu`

### Tier 3: End-to-End

Оценка полного пайплайна RAG-системы.

**Метрики:** `avg_e2e_score`, `avg_semantic_similarity`, `avg_rouge1_f`, `avg_bleu`

### Tier Judge: LLM-as-a-Judge Quality

Оценка качества и стабильности LLM-судьи (Qwen).

**Метрики:** `consistency_score`, `error_rate`, `avg_latency_ms`

### Tier Judge Pipeline: Production Judge Quality

Оценка качества production judge (Mistral) в реальном RAG-пайплайне.

**Метрики:** `accuracy`, `precision`, `recall`, `f1_score`, `avg_latency_ms`

### Tier UX: User Experience Quality

Оценка качества пользовательского опыта и работы кэша.

**Метрики:** `cache_hit_rate`, `context_preservation`, `multi_turn_consistency`

### Real Users

Retrieval-метрики на реальных вопросах пользователей с высоким score.

---

## Структура проекта

```
benchmarks/
├── data/
│   ├── dataset_YYYYMMDD_HHMMSS.json
│   ├── manual_dataset_YYYYMMDD_HHMMSS.json
│   └── dump/
├── docs/
│   ├── CLI_REFERENCE.md
│   ├── ARCHITECTURE.md
│   ├── RUN_MODES_LOCAL_VS_DOCKER.md
│   ├── SMOKE_SCENARIO_LOCAL.md
│   ├── SMOKE_SCENARIO_DOCKER.md
│   └── manual_annotation_guide.md
├── models/
│   ├── rag_benchmark.py
│   ├── real_queries_benchmark.py
│   └── retrieval_benchmark.py
├── reports/
│   ├── rag_benchmark_*.json
│   └── rag_benchmark_*.md
├── utils/
│   ├── llm_judge.py
│   ├── evaluator.py
│   ├── text_metrics.py
│   ├── embedding_generator.py
│   ├── database_dump_loader.py
│   └── report_generator.py
├── tests/
├── Makefile
├── Dockerfile
├── generate_embeddings.py
├── generate_dataset.py
├── load_database_dump.py
├── run_comprehensive_benchmark.py
├── run_dashboard.py
├── dashboard.py
└── save_models.py
```

---

## Makefile команды

Справка по доступным командам:

```bash
cd Submodules/voproshalych/benchmarks
make help
```

---

## Переменные окружения

Обязательные переменные в `.env.docker`:

- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- `MISTRAL_API`, `MISTRAL_MODEL`
- `BENCHMARKS_JUDGE_API_KEY` (или `JUDGE_API`)
- `EMBEDDING_MODEL_PATH` (или используется HF)
