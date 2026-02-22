# Smoke-сценарий для benchmarks (Локальный режим)

Полный цикл проверки: synthetic, manual и real-users режимы без Docker.

## Предварительные требования

- Python 3.12+
- Установленный UV: `curl -LsSf https://astral.sh/uv | sh`
- PostgreSQL доступен локально или по сети
- Переменные окружения в `.env` или `.env.docker`

## 0) Установка зависимостей

```bash
cd Submodules/voproshalych
uv sync
```

## 1) Подготовка окружения

Создайте или обновите `.env.docker` файл:

```bash
cd Submodules/voproshalych
cp .env.docker.example .env.docker
```

Обязательные переменные:
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- `MISTRAL_API`, `MISTRAL_MODEL`
- `BENCHMARKS_JUDGE_API_KEY` (или `JUDGE_API`)
- `EMBEDDING_MODEL_PATH` (или используется HF)

## 2) Подготовка БД

### Вариант A: Загрузка дампа

```bash
cd Submodules/voproshalych
uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump
```

**Важно:** При загрузке дампа таблицы автоматически очищаются перед загрузкой.

### Вариант B: Только удаление таблиц

```bash
cd Submodules/voproshalych
uv run python benchmarks/load_database_dump.py --drop-tables-only
```

## 3) Генерация эмбеддингов

### Варианты генерации

```bash
cd Submodules/voproshalych

# Вариант A: Эмбеддинги для чанков (документов) — используется для RAG
uv run python benchmarks/generate_embeddings.py --chunks

# Вариант B: Эмбеддинги для всех вопросов
uv run python benchmarks/generate_embeddings.py --all

# Вариант C: Эмбеддинги для вопросов с оценкой 5 (только полезные)
uv run python benchmarks/generate_embeddings.py --score 5

# Вариант D: Эмбеддинги для вопросов с оценкой 1
uv run python benchmarks/generate_embeddings.py --score 1

# Вариант E: Перезаписать существующие эмбеддинги (любой из выше)
uv run python benchmarks/generate_embeddings.py --all --overwrite

# Вариант F: Проверить покрытие эмбеддингами
uv run python benchmarks/generate_embeddings.py --check-coverage
```

## 4) Synthetic dataset + benchmark

### Генерация датасета

```bash
cd Submodules/voproshalych

# ===========================================
# Режим 1: Synthetic — вопросы из чанков через LLM
# ===========================================

# Вариант A: Тестовый режим (5 вопросов) — проверить что работает
uv run python benchmarks/generate_dataset.py \
  --mode synthetic --max-questions 5

# Вариант B: Ограниченный режим (20 вопросов) — текущая команда
uv run python benchmarks/generate_dataset.py \
  --mode synthetic --max-questions 20

# Вариант C: Полный режим — ВСЕ чанки с эмбеддингами
uv run python benchmarks/generate_dataset.py \
  --mode synthetic --max-questions 10000

# ===========================================
# Режим 2: Из реальных вопросов пользователей
# ===========================================

# Вариант A: Тестовый режим (5 вопросов)
uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions --max-questions 5

# Вариант B: Ограниченный режим (20 вопросов)
uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions --max-questions 20

# Вариант C: Полный режим — ВСЕ реальные вопросы пользователей
uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions --max-questions 10000

# ===========================================
# Режим 3: Только вопросы с оценкой 5 (полезные)
# ===========================================

# Вариант A: Тестовый режим (5 вопросов)
uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions-score-5 --max-questions 5

# Вариант B: Ограниченный режим (20 вопросов)
uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions-score-5 --max-questions 20

# Вариант C: Полный режим — ВСЕ вопросы с оценкой 5
uv run python benchmarks/generate_dataset.py \
  --mode from-real-questions-score-5 --max-questions 10000

# ===========================================
# Режим 4: Экспорт для ручной аннотации
# ===========================================

uv run python benchmarks/generate_dataset.py \
  --mode export-annotation --output benchmarks/data/dataset_for_annotation.json
```

### Запуск benchmark

```bash
cd Submodules/voproshalych

# ===========================================
# Тестовый режим (быстрая проверка)
# ===========================================

# Synthetic, 5 вопросов, все tier
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic --limit 5

# Synthetic, 10 вопросов — текущая команда
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic --limit 10

# ===========================================
# Полный режим (все вопросы из датасета)
# ===========================================

# Synthetic, все вопросы, все tier (финальный замер)
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic

# Synthetic, все вопросы, только tier 0-1 (без judge, быстрее)
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier 1 --mode synthetic
```

## 5) Manual dataset + benchmark (ручная аннотация)

Рекомендуемый workflow для ручной аннотации:

```bash
cd Submodules/voproshalych

# 1. Экспорт существующего датасета для аннотации
uv run python benchmarks/generate_dataset.py \
  --mode export-annotation --output benchmarks/data/dataset_for_annotation.json

# 2. Отредактируйте файл вручную - добавьте метки качества
# Поля для заполнения: is_question_ok, is_answer_ok, is_chunk_ok, notes

# 3. Запуск бенчмарка на аннотированном датасете

# Вариант A: Тестовый режим (5 вопросов)
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode manual \
  --manual-dataset benchmarks/data/dataset_for_annotation.json \
  --limit 5

# Вариант B: Полный режим — все вопросы из файла аннотации
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode manual \
  --manual-dataset benchmarks/data/dataset_for_annotation.json
```

Также можно использовать `--mode manual` с любым JSON файлом в новом формате.

## 6) Real users benchmark

```bash
cd Submodules/voproshalych

# ===========================================
# Тестовый режим
# ===========================================

# Вариант A: 10 вопросов, score 5
uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 10 --top-k 10

# Вариант B: 50 вопросов, score 5
uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 50 --top-k 10

# ===========================================
# Полный режим
# ===========================================

# Вариант A: ВСЕ вопросы с оценкой 5
uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 10000 --top-k 10

# Вариант B: ВСЕ вопросы с оценкой 1 (негативные)
uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 1 --real-limit 10000 --top-k 10

# Вариант C: Последние вопросы (по дате создания)
uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-latest --real-limit 100 --top-k 10
```

## 7) Проверка результатов через дашборд

```bash
cd Submodules/voproshalych
uv run python benchmarks/run_dashboard.py
```

Дашборд доступен по адресу: `http://localhost:7860`

Ожидаемо:
- в `benchmarks/reports/` появляются свежие `rag_benchmark_*.json/.md`;
- в `benchmarks/reports/benchmark_runs.json` появляются новые записи;
- в дашборде видны метрики и графики по новым запускам.

## Устранение проблем

### Проблема: "Не удалось подключиться к БД"

Проверьте `.env.docker` файл:
```bash
cd Submodules/voproshalych
cat .env.docker | grep POSTGRES
```

Убедитесь что PostgreSQL запущен:
```bash
psql -h localhost -U postgres -d postgres -c "SELECT 1"
```

### Проблема: "Ошибка импорта qa.config"

Убедитесь что вы находитесь в директории `Submodules/voproshalych` при запуске команд:
```bash
cd Submodules/voproshalych
uv run python benchmarks/generate_embeddings.py --chunks
```

### Проблема: "Нет эмбеддингов в БД"

Запустите генерацию эмбеддингов:
```bash
cd Submodules/voproshalych
uv run python benchmarks/generate_embeddings.py --chunks
```

## Очистка

Для очистки после тестирования:

```bash
cd Submodules/voproshalych

# Очистить отчёты
rm -rf benchmarks/reports/*.json benchmarks/reports/*.md

# Очистить историю запусков
rm -f benchmarks/reports/benchmark_runs.json

# Очистить датасеты
rm -rf benchmarks/data/dataset_*.json

# Очистить кэш UV
rm -rf .venv/
```
