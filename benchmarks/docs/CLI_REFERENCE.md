# CLI Reference: Команды модуля бенчмарков

Полная документация CLI команд для локального режима работы benchmark-модуля.

---

## Содержание

1. [load_database_dump.py](#load_database_dumppy)
2. [generate_embeddings.py](#generate_embeddingspy)
3. [generate_dataset.py](#generate_datasetpy)
4. [run_comprehensive_benchmark.py](#run_comprehensive_benchmarkpy)
5. [visualize_vector_space.py](#visualize_vector_spacepy)
6. [analyze_chunk_utilization.py](#analyze_chunk_utilizationpy)
7. [analyze_topic_coverage.py](#analyze_topic_coveragepy)
8. [analyze_real_users_domain.py](#analyze_real_users_domainpy)
9. [run_dashboard.py](#run_dashboardpy)

---

## load_database_dump.py

Загрузка PostgreSQL дампа или очистка таблиц.

**Файл:** `benchmarks/load_database_dump.py`

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--dump` | str | None | Путь к `.dump` файлу |
| `--dump-dir` | str | None | Директория, где ищется последний дамп |
| `--drop-tables-only` | flag | False | Удалить таблицы без загрузки дампа |
| `--verbose` | flag | False | Подробный лог |
| `--quiet` | flag | False | Только ошибки |

Пример:

```bash
uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump
```

---

## generate_embeddings.py

Генерация эмбеддингов для чанков и вопросов.

**Файл:** `benchmarks/generate_embeddings.py`

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--all` | flag | False | Эмбеддинги для всех вопросов |
| `--score` | int | None | Эмбеддинги вопросов с оценкой 1 или 5 |
| `--chunks` | flag | False | Эмбеддинги для всех чанков |
| `--check-coverage` | flag | False | Проверка покрытия эмбеддингами |
| `--overwrite` | flag | False | Перезаписать существующие эмбеддинги |

---

## generate_dataset.py

Генерация benchmark-датасета.

**Файл:** `benchmarks/generate_dataset.py`

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--mode` | str | synthetic | `synthetic`, `export-annotation` |
| `--max-questions` | int | 500 | Лимит записей |
| `--num-samples` | int | None | Устаревший алиас `--max-questions` |
| `--output` | str | None | Выходной JSON (иначе versioned auto-name) |
| `--check-only` | flag | False | Проверить существующий датасет |
| `--skip-existing-dataset` | str | None | Не дублировать элементы из указанного датасета |
| `--generation-attempts` | int | 3 | Попытки генерации на один чанк |

---

## run_comprehensive_benchmark.py

Главный orchestrator benchmark-сценариев.

**Файл:** `benchmarks/run_comprehensive_benchmark.py`

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--tier` | str | all | `0/1/2/3/judge/judge_pipeline/ux/all` |
| `--mode` | str | synthetic | `synthetic/manual` |
| `--dataset` | str | benchmarks/data/golden_dataset_synthetic.json | Основной dataset path |
| `--manual-dataset` | str | None | Dataset path для manual mode |
| `--limit` | int | None | Лимит записей |
| `--top-k` | int | 10 | top-k retrieval |
| `--output-dir` | str | benchmarks/reports | Директория отчётов |
| `--skip-checks` | flag | False | Пропустить prerequisites |
| `--analyze-utilization` | flag | False | Запустить chunk utilization |
| `--utilization-questions-source` | str | synthetic | `synthetic` или `real` |
| `--utilization-question-limit` | int | 500 | Лимит вопросов utilization |
| `--utilization-top-k` | int | 10 | top-k utilization |
| `--analyze-topics` | flag | False | Запустить topic coverage |
| `--topics-question-limit` | int | 2000 | Лимит вопросов topic coverage |
| `--topics-count` | int | 20 | Число topic clusters |
| `--topics-top-k` | int | 5 | top-k для topics |
| `--consistency-runs` | int | 1 | Повторы запроса для consistency метрик |
| `--judge-eval-mode` | str | direct | `direct` или `reasoned` |
| `--judge-models` | str | "" | CSV judge моделей для multi-run |
| `--generation-models` | str | "" | CSV generation моделей для multi-run |
| `--production-judge-models` | str | "" | CSV production judge моделей для tier_judge_pipeline |
| `--analyze-domain` | flag | False | Запустить real-user domain analysis |
| `--domain-limit` | int | 5000 | Лимит domain analysis вопросов |

Пример:

```bash
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all \
  --mode synthetic \
  --limit 50 \
  --consistency-runs 2 \
  --judge-eval-mode reasoned \
  --analyze-utilization \
  --analyze-topics \
  --analyze-domain
```

---

## visualize_vector_space.py

UMAP-визуализация эмбеддингов чанков.

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--limit` | int | 5000 | Число чанков |
| `--output` | str | benchmarks/reports/vector_space.html | Выходной HTML |
| `--3d` | flag | False | 3D режим |
| `--color-by` | str | section | `cluster/chunk_id/section` |

---

## analyze_chunk_utilization.py

Оценка доли чанков, используемых retrieval-пайплайном.

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--questions-source` | str | synthetic | `synthetic` или `real` |
| `--question-limit` | int | 500 | Лимит вопросов |
| `--top-k` | int | 10 | top-k retrieval |
| `--output` | str | benchmarks/reports/utilization.json | Выходной JSON |

---

## analyze_topic_coverage.py

Кластеризация реальных вопросов и оценка покрытия чанками.

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--question-limit` | int | 2000 | Лимит вопросов |
| `--n-topics` | int | 20 | Число topic clusters |
| `--top-k` | int | 5 | top-k retrieval |
| `--output` | str | benchmarks/reports/topic_coverage.json | Выходной JSON |

---

## analyze_real_users_domain.py

Доменная аналитика корпуса real-user вопросов.

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--limit` | int | 5000 | Лимит вопросов |
| `--output` | str | benchmarks/reports/real_users_domain_analysis.json | Выходной JSON |

---

## run_dashboard.py

Запуск аналитического dashboard.

| Флаг | Тип | По умолчанию | Описание |
|------|-----|--------------|----------|
| `--port` | int | auto(7860+) | Порт dashboard |

Пример:

```bash
uv run python benchmarks/run_dashboard.py --port 7860
```

---

## Makefile команды

`benchmarks/Makefile` содержит только локальный режим + инфраструктурные
команды для БД/миграций.

| Команда | Описание |
|---------|----------|
| `make up` | Поднять `db` и `db-migrate` через compose |
| `make down` | Остановить сервисы |
| `make install-local` | `uv sync` |
| `make load-dump-local` | Загрузить dump |
| `make drop-tables-local` | Очистить таблицы |
| `make generate-embeddings-local` | Эмбеддинги + coverage |
| `make generate-dataset-local` | Synthetic dataset (10) |
| `make run-benchmarks-local` | Тестовый комплексный benchmark |
| `make run-dashboard-local` | Запуск dashboard |
| `make run-local SCRIPT=... ARGS=...` | Запуск произвольного скрипта |

---

## Связанные документы

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [METRICS.md](METRICS.md)
- [SMOKE_SCENARIO_LOCAL.md](SMOKE_SCENARIO_LOCAL.md)
- [MODEL_COMPARISON_GUIDE.md](MODEL_COMPARISON_GUIDE.md)
- [manual_annotation_guide.md](manual_annotation_guide.md)
