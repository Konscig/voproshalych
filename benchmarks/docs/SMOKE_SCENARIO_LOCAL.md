# Smoke-сценарий для benchmarks (Локальный режим)

Полный цикл проверки в локальном режиме:

- сервисы инфраструктуры (`db`, `db-migrate`) поднимаются через
  `docker-compose.benchmarks.yml`;
- все benchmark-скрипты выполняются локально через `uv run`.

---

## Предварительные требования

- Python 3.12+
- UV установлен
- `.env.docker` заполнен рабочими значениями

---

## 0) Переход в проект

```bash
cd Submodules/voproshalych
```

---

## 1) Поднять БД и миграции

```bash
docker compose -f docker-compose.benchmarks.yml up -d db db-migrate
```

Make-аналог:

```bash
cd Submodules/voproshalych/benchmarks
make up
```

---

## 2) Установка зависимостей

```bash
cd Submodules/voproshalych
uv sync
```

Make-аналог:

```bash
cd Submodules/voproshalych/benchmarks
make install-local
```

---

## 3) Подготовка БД

### Тестовый режим

```bash
cd Submodules/voproshalych
uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump
```

Make-аналог:

```bash
cd Submodules/voproshalych/benchmarks
make load-dump-local
```

### Полный режим

```bash
cd Submodules/voproshalych
uv run python benchmarks/load_database_dump.py --drop-tables-only
uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump
```

Make-аналоги:

```bash
cd Submodules/voproshalych/benchmarks
make drop-tables-local
make load-dump-local
```

---

## 4) Генерация эмбеддингов

### Тестовый режим

```bash
cd Submodules/voproshalych
uv run python benchmarks/generate_embeddings.py --chunks
```

Make-аналог:

```bash
cd Submodules/voproshalych/benchmarks
make generate-embeddings-local
```

### Полный режим

```bash
cd Submodules/voproshalych
uv run python benchmarks/generate_embeddings.py --chunks
uv run python benchmarks/generate_embeddings.py --check-coverage
```

---

## 5) Генерация датасетов

### Тестовый режим (до 10 вопросов)

```bash
cd Submodules/voproshalych
uv run python benchmarks/generate_dataset.py --mode synthetic --max-questions 10
uv run python benchmarks/generate_dataset.py --mode export-annotation --output benchmarks/data/dataset_for_annotation.json
```

Make-аналоги:

```bash
cd Submodules/voproshalych/benchmarks
make generate-dataset-local
make run-local SCRIPT=benchmarks/generate_dataset.py ARGS="--mode export-annotation --output benchmarks/data/dataset_for_annotation.json"
```

### Полный режим

```bash
cd Submodules/voproshalych
uv run python benchmarks/generate_dataset.py --mode synthetic --max-questions 10000
```

---

## 6) Запуск бенчмарков

### Тестовый режим (10 вопросов)

```bash
cd Submodules/voproshalych

uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic --limit 10 \
  --consistency-runs 2 \
  --judge-eval-mode reasoned \
  --analyze-utilization --utilization-questions-source real --utilization-question-limit 10 \
  --analyze-topics --topics-question-limit 10 --topics-count 5 \
  --analyze-domain --domain-limit 500

uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode manual \
  --manual-dataset benchmarks/data/dataset_for_annotation.json \
  --limit 10 --judge-eval-mode reasoned --consistency-runs 2
```

Make-аналог (произвольные сценарии):

```bash
cd Submodules/voproshalych/benchmarks
make run-local SCRIPT=benchmarks/run_comprehensive_benchmark.py ARGS="--tier all --mode synthetic --limit 10 --consistency-runs 2 --judge-eval-mode reasoned --analyze-utilization --utilization-questions-source real --utilization-question-limit 10 --analyze-topics --topics-question-limit 10 --topics-count 5 --analyze-domain --domain-limit 500"
```

### Полный режим

```bash
cd Submodules/voproshalych
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic --consistency-runs 2 \
  --analyze-utilization --utilization-questions-source real --utilization-question-limit 500 \
  --analyze-topics --topics-question-limit 2000 --topics-count 20 \
  --analyze-domain --domain-limit 5000
```

---

## 7) Multi-model sweep (опционально)

### Тестовый режим

```bash
cd Submodules/voproshalych
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier 2 --mode synthetic --limit 10 \
  --judge-models "openai/gpt-4o-mini,openai/gpt-4.1-mini" \
  --generation-models "mistral-small-latest,mistral-medium-latest" \
  --judge-eval-mode reasoned
```

---

## 8) Дополнительная аналитика

### Тестовый режим

```bash
cd Submodules/voproshalych
uv run python benchmarks/visualize_vector_space.py --limit 500 --output benchmarks/reports/vector_space_test.html
uv run python benchmarks/analyze_chunk_utilization.py --questions-source real --question-limit 10 --top-k 10 --output benchmarks/reports/utilization_test.json
uv run python benchmarks/analyze_topic_coverage.py --question-limit 10 --n-topics 5 --top-k 5 --output benchmarks/reports/topic_coverage_test.json
uv run python benchmarks/analyze_real_users_domain.py --limit 500 --output benchmarks/reports/real_users_domain_analysis_test.json
```

### Полный режим

```bash
cd Submodules/voproshalych
uv run python benchmarks/visualize_vector_space.py --limit 5000 --output benchmarks/reports/vector_space.html
uv run python benchmarks/analyze_chunk_utilization.py --questions-source real --question-limit 500 --top-k 10 --output benchmarks/reports/utilization.json
uv run python benchmarks/analyze_topic_coverage.py --question-limit 2000 --n-topics 20 --top-k 5 --output benchmarks/reports/topic_coverage.json
uv run python benchmarks/analyze_real_users_domain.py --limit 5000 --output benchmarks/reports/real_users_domain_analysis.json
```

---

## 9) Дашборд

```bash
cd Submodules/voproshalych
uv run python benchmarks/run_dashboard.py --port 7860
```

Make-аналог:

```bash
cd Submodules/voproshalych/benchmarks
make run-dashboard-local
```

Открой: `http://localhost:7860`

---

## 10) Остановка сервисов

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml down
```

Make-аналог:

```bash
cd Submodules/voproshalych/benchmarks
make down
```
