# Модуль для оценки качества RAG-системы Вопрошалыч

Enterprise-grade система бенчмарков для оценки Retrieval, Generation и End-to-End качества RAG-пайплайна.

---

## Обзор

Модуль `benchmarks` покрывает три рабочих сценария:

- **Synthetic dataset**: автоматически сгенерированный golden standard
- **Manual dataset**: экспертная разметка для строгой валидации
- **Real user data**: retrieval-метрики на реальных вопросах пользователей

### Ключевые особенности

- ✅ **Single Source of Truth**: работа с PostgreSQL базой для векторного поиска
- ✅ **LLM-as-a-Judge**: оценка quality-метрик через judge-model
- ✅ **Семь tier-уровней**: Tier 0 (Embedding), Tier 1 (Retrieval), Tier 2 (Generation), Tier 3 (End-to-End), Tier Judge (Qwen), Tier Judge Pipeline (Mistral), Tier UX
- ✅ **Manual + Real Users режимы**: отдельные пайплайны для академичной оценки
- ✅ **Локальное хранение**: Датасеты → JSON, Отчёты → Markdown + JSON

---

## Документация

| Документ | Описание |
|----------|----------|
| [CLI_REFERENCE.md](docs/CLI_REFERENCE.md) | Справочник по CLI командам |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | Архитектура с Mermaid диаграммами |
| [SMOKE_SCENARIO_LOCAL.md](docs/SMOKE_SCENARIO_LOCAL.md) | Полный цикл проверки |
| [manual_annotation_guide.md](docs/manual_annotation_guide.md) | Руководство по ручной разметке |
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

Открой дашборд: `http://localhost:7860`

---

## Уровни тестирования

### Tier 0: Intrinsic Embedding Quality

Оценка внутреннего качества эмбеддингов через статистические метрики.

**Метрики:** `avg_nn_distance`, `density_score`, `avg_spread`, `effective_dimensionality`

### Tier 1: Retrieval Accuracy

Оценка качества векторного поиска через pgvector.

**Метрики:** `HitRate@K`, `Recall@K`, `Precision@K`, `MRR`, `NDCG@K`

### Tier 2: Generation Quality

Оценка качества генерации ответов при идеальном контексте.

**Метрики:** `avg_faithfulness`, `avg_answer_relevance` (LLM Judge), `avg_rouge1_f`, `avg_rougeL_f`, `avg_bleu`

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
│   ├── SMOKE_SCENARIO_LOCAL.md
│   ├── METRICS.md
│   └── manual_annotation_guide.md
├── models/
│   ├── rag_benchmark.py
│   ├── real_queries_benchmark.py
│   └── retrieval_benchmark.py
├── reports/
│   ├── rag_benchmark_*.json
│   ├── rag_benchmark_*.md
│   ├── vector_space.html
│   ├── utilization.json
│   └── topic_coverage.json
├── utils/
│   ├── llm_judge.py
│   ├── evaluator.py
│   ├── text_metrics.py
│   ├── embedding_generator.py
│   ├── database_dump_loader.py
│   └── report_generator.py
├── visualize_vector_space.py      # UMAP визуализация
├── analyze_chunk_utilization.py # Анализ использования чанков
├── analyze_topic_coverage.py    # Анализ покрытия тем
├── generate_embeddings.py
├── generate_dataset.py
├── load_database_dump.py
├── run_comprehensive_benchmark.py
├── run_dashboard.py
├── dashboard.py
└── Makefile
```

---

## Makefile команды

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
