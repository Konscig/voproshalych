# Модуль для оценки качества RAG-системы Вопрошалыч

Enterprise-grade система бенчмарков для комплексной оценки качества RAG (Retrieval-Augmented Generation) архитектуры чат-бота Вопрошалыч.

## Обзор

Система бенчмарков предназначена для объективной оценки качества работы RAG-системы на основе реальной PostgreSQL базы данных и LLM-оценки.

### Ключевые особенности

- ✅ **Single Source of Truth**: Работает только с реальной PostgreSQL (Docker)
- ✅ **LLM-as-a-Judge**: Использует DeepSeek/Qwen для семантической оценки
- ✅ **Три уровня тестирования**: Retrieval, Generation, End-to-End
- ✅ **Синтетический Golden Standard**: Автоматическая генерация тестовых данных
- ✅ **Enterprise-grade**: Отказоустойчивость, retry логика, подробные отчёты
- ✅ **Configuration Management**: Единый источник конфигурации для всех компонентов

## Структура проекта

```
benchmarks/
├── data/
│   └── dataset_YYYYMMDD_HHMMSS.json   # Версионированный золотой стандарт
│
├── models/
│   └── rag_benchmark.py              # Основной класс бенчмарка
│
├── utils/
│   ├── llm_judge.py                  # LLM-судья (DeepSeek/Qwen)
│   └── embedding_generator.py        # Генератор эмбеддингов
│
├── reports/                          # Отчёты бенчмарков
│   ├── rag_benchmark_*.json          # Метрики в JSON
│   └── rag_benchmark_*.md            # Отчёты в Markdown
│
├── dashboard.py                      # Интерактивный дашборд (Gradio)
├── generate_embeddings.py            # Генерация эмбеддингов
├── generate_dataset.py               # Генерация синтетического датасета
├── run_comprehensive_benchmark.py    # Запуск всех уровней
├── pyproject.toml                    # Управление зависимостями
└── README.md                         # Этот файл
```

## Компоненты системы

### 1. LLM-судья (`utils/llm_judge.py`)

Клиент для взаимодействия с DeepSeek API с retry логикой и проверкой соединения.

**Ключевые методы:**
- `generate_question_from_chunk()` - Генерация вопроса и идеального ответа
- `evaluate_faithfulness()` - Оценка точности ответа (1-5)
- `evaluate_answer_relevance()` - Оценка релевантности (1-5)
- `evaluate_e2e_quality()` - Оценка E2E качества (1-5)

**Конфигурация (.env.docker):**
```bash
BENCHMARKS_JUDGE_API_KEY=your_api_key
BENCHMARKS_JUDGE_BASE_URL=https://api.deepseek.com
BENCHMARKS_JUDGE_MODEL=deepseek-chat
```

### 2. Генератор эмбеддингов (`generate_embeddings.py`)

Генерирует векторные представления для вопросов и чанков.

**Запуск:**
```bash
# Генерация для всех вопросов
python benchmarks/generate_embeddings.py --all

# Генерация для чанков (NEW!)
python benchmarks/generate_embeddings.py --chunks

# Проверка покрытия
python benchmarks/generate_embeddings.py --check-coverage
```

### 3. Генерация датасета (`generate_dataset.py`)

Автоматическая генерация синтетического золотого стандарта с помощью LLM-судьи.

**Запуск:**
```bash
# Генерация 100 вопросов
python benchmarks/generate_dataset.py --num-samples 100

# Проверка существующего датасета
python benchmarks/generate_dataset.py --check-only --output benchmarks/data/dataset_custom.json
```

**Формат датасета:**
```json
{
  "chunk_id": 123,
  "chunk_text": "...",
  "question": "Как...",
  "ground_truth_answer": "...",
  "confluence_url": "https://..."
}
```

### 4. RAG бенчмарк (`models/rag_benchmark.py`)

Основной класс для выполнения трёх уровней тестирования.

## Управление зависимостями

### Файл конфигурации: `benchmarks/pyproject.toml`

Для изолированного управления зависимостями модуля бенчмарков используется отдельный `pyproject.toml`.

**Структура зависимостей:**
```toml
[project]
dependencies = [
    "python-dotenv",
    "sqlalchemy",
    "psycopg2-binary",
    "pgvector",
    "sentence-transformers",
    "numpy",
    "openai",
    "tenacity",
]

[project.optional-dependencies]
dashboard = [
    "gradio>=6.5.1",
]

all = [
    "voproshalych-benchmarks[dashboard]",
]
```

**Преимущества:**
- Изолированные зависимости для бенчарков
- Опциональные группы для разных сценариев использования
- Чистое разделение ответственности

**Установка зависимостей:**
```bash
# Установить все зависимости
cd benchmarks
uv sync

# Установить включая дашборд
cd benchmarks
uv sync --extra dashboard
```

## Конфигурация моделей

### Единый источник конфигурации

Все настройки моделей управляются через `qa/Config` класс, что исключает хардкод и обеспечивает согласованность.

### Переменные окружения

В `.env.docker` добавлены следующие переменные:

```bash
# настройки API Mistral
MISTRAL_API=
MISTRAL_MODEL=

# настройки модели эмбеддингов
EMBEDDING_MODEL_PATH=saved_models/multilingual-e5-large-wikiutmn

# настройки модели судьи
JUDGE_API=
JUDGE_MODEL=

# настройки LLM-судьи для бенчмарков (DeepSeek/Qwen)
BENCHMARKS_JUDGE_API_KEY=
BENCHMARKS_JUDGE_BASE_URL=https://api.deepseek.com
BENCHMARKS_JUDGE_MODEL=deepseek-chat
```

### Использование конфигурации в коде

**В скриптах бенчарков:**
```python
from qa.config import Config

# Генерация эмбеддингов
model_path = Config.EMBEDDING_MODEL_PATH  # НЕ "nizamovtimur/..."
encoder = SentenceTransformer(model_path, device="cpu")
```

**Преимущества:**
- Отсутствие хардкода путей к моделям
- Лёгкая смена модели без изменения кода
- Единый источник правды для конфигурации

## Три уровня тестирования

### Tier 1: Retrieval Accuracy (Поиск)

**Цель:** Находит ли векторный поиск через pgvector правильный чанк?

**Техническая реализация:**
- Используются **реальные SQL запросы** через SQLAlchemy
- Оператор `cosine_distance` для векторного поиска в pgvector
- Тестирование реальной производительности PostgreSQL

**Метрики:**
- `HitRate@1/5/10` - Доля релевантных результатов в топ-K
- `MRR` (Mean Reciprocal Rank) - Средний обратный ранг

**Пример запуска:**
```bash
python benchmarks/run_comprehensive_benchmark.py --tier 1
```

### Tier 2: Generation Quality (Генерация)

**Цель:** Может ли Mistral LLM ответить на вопрос с идеальным контекстом?

**Техническая реализация:**
- Используется **реальная функция генерации** из `qa.main.get_answer()`
- Вызовы к Mistral API через проектный промпт
- Оценка ответов LLM-судьёй

**Метрики:**
- `avg_faithfulness` - Точность ответа (отсутствие галлюцинаций)
- `avg_answer_relevance` - Релевантность ответа

**Пример запуска:**
```bash
python benchmarks/run_comprehensive_benchmark.py --tier 2
```

### Tier 3: End-to-End (Полный пайплайн)

**Цель:** Как работает система целиком (Поиск + Генерация)?

**Техническая реализация:**
- Реальный pgvector поиск (как в Tier 1)
- Реальная Mistral генерация (как в Tier 2)
- Полный пайплайн RAG-системы

**Метрики:**
- `avg_e2e_score` - Общая оценка качества (1-5)
- `avg_semantic_similarity` - Косинусное сходство ответов

**Пример запуска:**
```bash
python benchmarks/run_comprehensive_benchmark.py --tier 3
```

## Полный рабочий цикл

### Шаг 1: Настройка окружения

```bash
# Из корня проекта
cd Submodules/voproshalych

# Python-зависимости (локальный запуск)
uv sync

# Настройка переменных окружения
cp .env.docker.example .env.docker
# Заполните API ключи в .env.docker
# - BENCHMARKS_JUDGE_API_KEY (или JUDGE_API)
# - MISTRAL_API
```

### Шаг 2: Запуск инфраструктуры

```bash
# Запустите сервисы и пересоберите образы
docker compose up -d --build
```

### Шаг 3: Генерация эмбеддингов

Рекомендуемый запуск бенчмарков - внутри контейнера `qa` с монтированием
текущей директории, чтобы отчеты и датасет сохранялись на хосте.

```bash
cd Submodules/voproshalych

# Генерация эмбеддингов для чанков
docker run --rm \
  --network voproshalych_chatbot-conn \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  virtassist/qa:latest \
  python benchmarks/generate_embeddings.py --chunks

# Проверка покрытия
docker run --rm \
  --network voproshalych_chatbot-conn \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  virtassist/qa:latest \
  python benchmarks/generate_embeddings.py --check-coverage
```

### Шаг 4: Генерация датасета

```bash
cd Submodules/voproshalych

# Сгенерируйте золотой стандарт (100 вопросов)
docker run --rm \
  --network voproshalych_chatbot-conn \
  -v "$PWD:/workspace" \
  -w /workspace \
  virtassist/qa:latest \
  python benchmarks/generate_dataset.py --num-samples 100

# Результат будет сохранен как versioned файл:
# benchmarks/data/dataset_YYYYMMDD_HHMMSS.json

# При необходимости укажите имя файла явно
docker run --rm \
  --network voproshalych_chatbot-conn \
  -v "$PWD:/workspace" \
  -w /workspace \
  virtassist/qa:latest \
  python benchmarks/generate_dataset.py --num-samples 100 --output benchmarks/data/dataset_custom.json
```

### Шаг 5: Запуск бенчмарков

```bash
cd Submodules/voproshalych

# Запустите все уровни
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

# Запуск на конкретном датасете
docker run --rm \
  --network voproshalych_chatbot-conn \
  -e BENCHMARK_GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)" \
  -e BENCHMARK_GIT_COMMIT_HASH="$(git rev-parse --short HEAD)" \
  -e BENCHMARK_RUN_AUTHOR="$(git config user.name)" \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  virtassist/qa:latest \
  python benchmarks/run_comprehensive_benchmark.py --tier all --dataset benchmarks/data/dataset_YYYYMMDD_HHMMSS.json --limit 50

# Или отдельные уровни
docker run --rm --network voproshalych_chatbot-conn -e BENCHMARK_GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)" -e BENCHMARK_GIT_COMMIT_HASH="$(git rev-parse --short HEAD)" -e BENCHMARK_RUN_AUTHOR="$(git config user.name)" -v "$PWD:/workspace" -w /workspace virtassist/qa:latest python benchmarks/run_comprehensive_benchmark.py --tier 1
docker run --rm --network voproshalych_chatbot-conn -e BENCHMARK_GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)" -e BENCHMARK_GIT_COMMIT_HASH="$(git rev-parse --short HEAD)" -e BENCHMARK_RUN_AUTHOR="$(git config user.name)" -v "$PWD:/workspace" -w /workspace virtassist/qa:latest python benchmarks/run_comprehensive_benchmark.py --tier 2
docker run --rm --network voproshalych_chatbot-conn -e BENCHMARK_GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)" -e BENCHMARK_GIT_COMMIT_HASH="$(git rev-parse --short HEAD)" -e BENCHMARK_RUN_AUTHOR="$(git config user.name)" -v "$PWD:/workspace" -w /workspace virtassist/qa:latest python benchmarks/run_comprehensive_benchmark.py --tier 3
```

### Шаг 6: Просмотр результатов

```bash
cd Submodules/voproshalych

# Посмотрите отчёты в Markdown
cat benchmarks/reports/rag_benchmark_*.md

# Или запустите дашборд
cd benchmarks
uv sync --extra dashboard
uv run python run_dashboard.py
# Дашборд будет доступен по адресу: http://localhost:7860
```

## Структура отчётов

Отчёты бенчмарков хранятся в `benchmarks/reports/`:

```
benchmarks/reports/
├── rag_benchmark_20260215_120000.json      # Метрики JSON
├── rag_benchmark_20260215_120000.md        # Отчёт Markdown
├── rag_benchmark_20260215_130000.json
├── rag_benchmark_20260215_130000.md
└── ...
```

**Формат отчёта JSON:**
```json
{
  "tier_1": {
    "tier": 1,
    "total_queries": 50,
    "hit_rate@1": 0.82,
    "hit_rate@5": 0.91,
    "hit_rate@10": 0.96,
    "mrr": 0.85
  },
  "tier_2": {
    "tier": 2,
    "total_queries": 50,
    "avg_faithfulness": 4.2,
    "avg_answer_relevance": 4.3
  },
  "tier_3": {
    "tier": 3,
    "total_queries": 50,
    "avg_e2e_score": 3.8,
    "avg_semantic_similarity": 0.75
  }
}
```

В JSON-артефакт добавляются поля `run_metadata`, `overall_status` и
`dataset_file`, а в БД (`benchmark_runs`) сохраняется связь запуска с
использованным датасетом.

## Интерактивный дашборд

### Запуск

```bash
# Из директории Submodules/voproshalych/benchmarks
cd Submodules/voproshalych/benchmarks

# Установите зависимости дашборда
uv sync --extra dashboard

# Запустите дашборд
uv run python run_dashboard.py
```

Дашборд будет доступен по адресу: http://localhost:7860

### Функционал

Дашборд предоставляет шесть основных вкладок:

#### Tab 1: Последний запуск

Сводная таблица `Latest Metrics Summary`:
- Tier 1: HitRate@5, MRR
- Tier 2: Faithfulness, Relevance
- Tier 3: E2E Score

Дополнительно в шапке доступен `System Info`:
- Judge model
- Generation model
- Embedding model

#### Tab 2: История

Графики изменения метрик во времени:
- Выбор уровня бенчмарка (Tier 1/2/3)
- Выбор метрики
- Визуализация трендов

#### Tab 3: Сравнение уровней

Сравнение метрик между уровнями на одном графике:
- Выбор метрики
- Наложение линий Tier 1/2/3

#### Tab 4: Все запуски

Таблица со всеми запусками бенчмарков:
- Timestamp
- Branch / Commit
- T1: HitRate@5, T1: MRR
- T2: Faithfulness, T2: Relevance
- T3: E2E Score

#### Tab 5: Run Dataset

Просмотр датасета, который был привязан к конкретному запуску:
- выбор запуска
- имя dataset-файла
- preview первых строк датасета

#### Tab 6: Справка

Академическая справка по метрикам и методологии LLM-as-a-Judge.

### Технические требования

- Python 3.12+
- gradio (для интерфейса)
- pandas (для обработки данных)

### Установка зависимостей

```bash
# Запускайте из benchmarks и установите extra-зависимости
cd Submodules/voproshalych/benchmarks
uv sync --extra dashboard
```

## Метрики качества

### Целевые значения

| Метрика | Tier 1 | Tier 2 | Tier 3 | Приоритет |
|---------|--------|--------|--------|-----------|
| HitRate@1 | ≥ 0.80 | - | - | 🔴 Высокий |
| HitRate@5 | ≥ 0.90 | - | - | 🔴 Высокий |
| HitRate@10 | ≥ 0.95 | - | - | 🟡 Средний |
| MRR | ≥ 0.80 | - | - | 🔴 Высокий |
| Faithfulness | - | ≥ 4.5 | - | 🔴 Высокий |
| Answer Relevance | - | ≥ 4.2 | - | 🔴 Высокий |
| E2E Score | - | - | ≥ 4.2 | 🔴 Высокий |
| Semantic Similarity | - | - | ≥ 0.85 | 🟡 Средний |

### Интерпретация метрик

| Метрика | Отлично | Хорошо | Средне | Плохо |
|---------|--------|--------|--------|--------|
| **Tier 1** | | | | |
| HitRate@1 | ≥80% | ≥60% | ≥40% | <40% |
| HitRate@5 | ≥90% | ≥70% | ≥50% | <50% |
| HitRate@10 | ≥95% | ≥80% | ≥60% | <60% |
| MRR | ≥0.8 | ≥0.6 | ≥0.4 | <0.4 |
| **Tier 2** | | | | |
| Faithfulness | ≥5.0 | ≥4.0 | ≥3.0 | <3.0 |
| Answer Relevance | ≥5.0 | ≥4.0 | ≥3.0 | <3.0 |
| **Tier 3** | | | | |
| E2E Score | ≥4.0 | ≥3.0 | ≥2.0 | <2.0 |
| Semantic Similarity | ≥0.8 | ≥0.6 | ≥0.4 | <0.4 | |

## CLI команды

### generate_embeddings.py

```bash
# Генерация для вопросов
python benchmarks/generate_embeddings.py --all
python benchmarks/generate_embeddings.py --score 5
python benchmarks/generate_embeddings.py --score 1

# Генерация для чанков
python benchmarks/generate_embeddings.py --chunks

# Проверка покрытия
python benchmarks/generate_embeddings.py --check-coverage

# Перезапись существующих
python benchmarks/generate_embeddings.py --all --overwrite
python benchmarks/generate_embeddings.py --chunks --overwrite
```

### generate_dataset.py

```bash
# Генерация датасета
python benchmarks/generate_dataset.py --num-samples 100

# Проверка существующего
python benchmarks/generate_dataset.py --check-only --output benchmarks/data/my_dataset.json

# Кастомный путь
python benchmarks/generate_dataset.py --num-samples 100 \
    --output benchmarks/data/my_dataset.json
```

### run_comprehensive_benchmark.py

```bash
# Все уровни
python benchmarks/run_comprehensive_benchmark.py --tier all

# Отдельные уровни
python benchmarks/run_comprehensive_benchmark.py --tier 1
python benchmarks/run_comprehensive_benchmark.py --tier 2
python benchmarks/run_comprehensive_benchmark.py --tier 3

# С ограничением количества записей
python benchmarks/run_comprehensive_benchmark.py --tier all --limit 10

# Кастомный Top-K для Tier 1
python benchmarks/run_comprehensive_benchmark.py --tier 1 --top-k 20

# Кастомный путь к датасету
python benchmarks/run_comprehensive_benchmark.py --tier all \
    --dataset benchmarks/data/my_dataset.json

# Кастомная директория для отчётов
python benchmarks/run_comprehensive_benchmark.py --tier all \
    --output-dir benchmarks/my_reports

# Пропуск проверок
python benchmarks/run_comprehensive_benchmark.py --tier all --skip-checks
```

## Архитектура

### Поток данных

```
┌─────────────────────┐
│  PostgreSQL (Docker)│
│  - QuestionAnswer   │
│  - Chunk           │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Config (qa/config) │
│  - EMBEDDING_MODEL_PATH│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ generate_embeddings│
│   (QA + Chunks)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ generate_dataset    │
│  (LLM Judge)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ rag_benchmark.py    │
│  - Tier 1: Retrieval│
│  - Tier 2: Generation│
│  - Tier 3: E2E     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Reports (JSON/MD)   │
│ + benchmark_runs DB │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Dashboard.py      │
│ (чтение из БД)      │
└─────────────────────┘
```

### Стратегия модульности

**Стратегия:** модуль `benchmarks` остаётся внутри `voproshalych`.

**Почему так:**
- Бенчмарки глубоко связаны с `qa.config` и SQLAlchemy-моделями приложения
  (`Chunk`, `QuestionAnswer`, `BenchmarkRun`).
- Вынос в отдельный репозиторий приведёт к дублированию моделей и сложной
  синхронизации зависимостей.
- Текущая интеграция позволяет быстрее развивать продуктовую аналитику качества.

**Future-proof план:**
- При реальной потребности переиспользования вынесем `benchmarks` в отдельный
  Python-пакет.
- До этого момента вынос репозитория избыточен (YAGNI).

### Операторы поиска

**Консистентное использование:**
- `qa/confluence_retrieving.py` → `cosine_distance()`
- `benchmarks/models/rag_benchmark.py` → `cosine_distance()`
- Все компоненты используют один и тот же оператор

**Проверено:** ✅ Операторы синхронизированы

### Зависимости

**Основные (pyproject.toml):**
- `sentence-transformers` - Эмбеддинги
- `sqlalchemy` - База данных
- `openai` - LLM API клиент (опционально)
- `tenacity` - Retry логика (опционально)
- `gradio` - Дашборд (опционально)

**Из основного проекта:**
- Все остальные зависимости через `pyproject.toml` корневого проекта

## Troubleshooting

### Проблема: запуск из локального Python не видит хост `db`

**Симптом:**
`OperationalError: could not translate host name "db" to address`

**Решение:**
- Для локального запуска создайте `.env` (рядом с `.env.docker`) и задайте
  `POSTGRES_HOST=localhost` (и при необходимости `POSTGRES_PORT=5432`)
- Для запуска внутри Docker оставьте `POSTGRES_HOST=db`

`qa/config.py` загружает `.env` c приоритетом над `.env.docker`, поэтому
локальные переопределения применяются автоматически.

### Проблема: локально не грузится embedding-модель из `saved_models`

**Симптом:**
`OSError: ... no file named model.safetensors ...`

**Решение:**
- Используйте запуск внутри контейнера с кэшем HF (см. команды выше)
- Или задайте в `.env`:

```bash
EMBEDDING_MODEL_PATH=nizamovtimur/multilingual-e5-large-wikiutmn
```

### Проблема: не хватает зависимостей LLM Judge

**Симптом:**
`ModuleNotFoundError: No module named 'openai'`

**Решение:**
- Убедитесь, что установлены `openai` и `tenacity`
- В проекте они добавлены в `benchmarks/pyproject.toml` и используются для
  `benchmarks/utils/llm_judge.py`

### Проблема: BENCHMARKS_JUDGE_API_KEY не установлен

**Решение:**
```bash
# Установите переменную окружения
export BENCHMARKS_JUDGE_API_KEY=your_api_key

# Или добавьте в .env.docker
echo "BENCHMARKS_JUDGE_API_KEY=your_api_key" >> .env.docker
```

### Проблема: Нет чанков с эмбеддингами

**Решение:**
```bash
# Сгенерируйте эмбеддинги для чанков
python benchmarks/generate_embeddings.py --chunks

# Проверьте покрытие
python benchmarks/generate_embeddings.py --check-coverage
```

### Проблема: Датасет не найден

**Решение:**
```bash
# Сгенерируйте датасет
python benchmarks/generate_dataset.py --num-samples 100
```

### Проблема: Ошибка подключения к DeepSeek

**Решение:**
```python
# Проверьте API ключ и URL
import os
print(f"API Key: {os.getenv('BENCHMARKS_JUDGE_API_KEY')}")
print(
    f"Base URL: {os.getenv('BENCHMARKS_JUDGE_BASE_URL', 'https://api.deepseek.com')}"
)
```

### Проблема: Дашборд не запускается

**Решение:**
```bash
# Запускайте из benchmarks и установите extra-зависимости
cd Submodules/voproshalych/benchmarks
uv sync --extra dashboard

# Запуск
uv run python run_dashboard.py

# Или используйте статические отчёты
cat benchmarks/reports/rag_benchmark_*.md
```

## Планируемые улучшения

### В ближайшее время

- ⏳ Добавить A/B тестирование разных стратегий поиска
- ⏳ Улучшить промпты для LLM-судьи
- ⏳ Добавить CI/CD интеграцию
- ⏳ Увеличить покрытие тестами до 90%+

### В долгосрочной перспективе

- ⏳ Мониторинг и алерты
- ⏳ Экспорт отчётов в PDF
- ⏳ Многоязычная поддержка оценок
- ⏳ Интеграция с системами логирования

## Полезные ссылки

- **Документация DeepSeek API:** https://platform.deepseek.com/docs
- **Sentence Transformers:** https://www.sbert.net/
- **Gradio:** https://gradio.app/
- **Testing Guide:** `../docs/main/TESTING_GUIDE.md`

---

**Версия:** 4.1 (Production Ready)
**Последнее обновление:** 2026-02-16
**Статус:** ✅ Работоспособно
