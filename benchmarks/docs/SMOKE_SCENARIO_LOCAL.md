# Smoke-сценарий для benchmarks (Локальный режим)

Полный цикл проверки: от подготовки БД до проверки дашборда.

## Предварительные требования

- Python 3.12+
- UV установлен (`curl -LsSf https://astral.sh/uv | sh`)
- PostgreSQL доступен локально или в Docker
- `.env.docker` заполнен рабочими значениями

## 0) Переход в проект

```bash
cd Submodules/voproshalych
```

## 1) Поднять БД и миграции

```bash
docker compose -f docker-compose.benchmarks.yml up -d db db-migrate
```

Или через Makefile:

```bash
cd Submodules/voproshalych/benchmarks
make up
```

## 2) Установка зависимостей

```bash
cd Submodules/voproshalych
uv sync
```

Или через Makefile:

```bash
cd Submodules/voproshalych/benchmarks
make install-local
```

## 3) Подготовка БД

### Тестовый режим

```bash
cd Submodules/voproshalych
uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump
```

Или через Makefile:

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

Или через Makefile:

```bash
cd Submodules/voproshalych/benchmarks
make drop-tables-local
make load-dump-local
```

## 4) Генерация эмбеддингов

### Тестовый режим

```bash
cd Submodules/voproshalych
uv run python benchmarks/generate_embeddings.py --chunks
```

Или через Makefile:

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

## 5) Генерация датасетов

### Тестовый режим (не более 10 вопросов)

```bash
cd Submodules/voproshalych

# Synthetic
uv run python benchmarks/generate_dataset.py --mode synthetic --max-questions 10

# Real questions
uv run python benchmarks/generate_dataset.py --mode from-real-questions --max-questions 10

# Score=5
uv run python benchmarks/generate_dataset.py --mode from-real-questions-score-5 --max-questions 10

# Экспорт для ручной аннотации
uv run python benchmarks/generate_dataset.py --mode export-annotation --output benchmarks/data/dataset_for_annotation.json
```

### Полный режим

```bash
cd Submodules/voproshalych

uv run python benchmarks/generate_dataset.py --mode synthetic --max-questions 10000
uv run python benchmarks/generate_dataset.py --mode from-real-questions --max-questions 10000
uv run python benchmarks/generate_dataset.py --mode from-real-questions-score-5 --max-questions 10000
```

## 6) Запуск бенчмарков

### Тестовый режим (10 вопросов)

```bash
cd Submodules/voproshalych

# Synthetic + новые анализы
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic --limit 10 \
  --analyze-utilization \
  --utilization-questions-source real \
  --utilization-question-limit 10 \
  --utilization-top-k 10 \
  --analyze-topics \
  --topics-question-limit 10 \
  --topics-count 5 \
  --topics-top-k 5

# Manual
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode manual \
  --manual-dataset benchmarks/data/dataset_for_annotation.json \
  --limit 10

# Real users
uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 10 --top-k 10 \
  --analyze-utilization --utilization-questions-source real \
  --utilization-question-limit 10 --analyze-topics \
  --topics-question-limit 10 --topics-count 5
```

### Полный режим

```bash
cd Submodules/voproshalych

uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic \
  --analyze-utilization --utilization-questions-source real \
  --utilization-question-limit 500 --utilization-top-k 10 \
  --analyze-topics --topics-question-limit 2000 --topics-count 20 --topics-top-k 5

uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 10000 --top-k 10 \
  --analyze-utilization --utilization-questions-source real \
  --utilization-question-limit 500 --analyze-topics \
  --topics-question-limit 2000 --topics-count 20
```

## 7) Дополнительная аналитика

### Тестовый режим

```bash
cd Submodules/voproshalych

# Визуализация UMAP (2D)
uv run python benchmarks/visualize_vector_space.py --limit 500 --output benchmarks/reports/vector_space_test.html

# Визуализация UMAP (3D)
uv run python benchmarks/visualize_vector_space.py --limit 300 --3d --output benchmarks/reports/vector_space_test_3d.html

# Анализ использования чанков
uv run python benchmarks/analyze_chunk_utilization.py --questions-source real --question-limit 10 --top-k 10 --output benchmarks/reports/utilization_test.json

# Анализ покрытия тем
uv run python benchmarks/analyze_topic_coverage.py --question-limit 10 --n-topics 5 --top-k 5 --output benchmarks/reports/topic_coverage_test.json
```

### Полный режим

```bash
cd Submodules/voproshalych

# Визуализация UMAP (2D)
uv run python benchmarks/visualize_vector_space.py --limit 5000 --output benchmarks/reports/vector_space.html

# Визуализация UMAP (3D)
uv run python benchmarks/visualize_vector_space.py --limit 3000 --3d --output benchmarks/reports/vector_space_3d.html

# Анализ использования чанков
uv run python benchmarks/analyze_chunk_utilization.py --questions-source real --question-limit 500 --top-k 10 --output benchmarks/reports/utilization.json

# Анализ покрытия тем
uv run python benchmarks/analyze_topic_coverage.py --question-limit 2000 --n-topics 20 --top-k 5 --output benchmarks/reports/topic_coverage.json
```

## 8) Дашборд

```bash
cd Submodules/voproshalych
uv run python benchmarks/run_dashboard.py
```

Открой: `http://localhost:7860`

## 9) Остановка и очистка

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml down
```
