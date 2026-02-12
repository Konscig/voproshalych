# Система бенчмарков для Вопрошалыча

Модуль для оценки качества RAG-системы Вопрошалыч.

## Структура

```
benchmarks/
├── data/                 # Данные для бенчмарков
│   ├── raw/             # Сырые данные
│   ├── cleaned/         # Очищенные данные
│   ├── annotated/       # Аннотированные данные
│   └── final/           # Финальные датасеты
├── models/              # Классы бенчмарков
│   └── retrieval_benchmark.py
├── utils/               # Утилиты
│   ├── data_loader.py
│   ├── embedding_generator.py
│   ├── evaluator.py
│   └── report_generator.py
├── tests/               # Тесты
└── reports/             # Отчёты
```

## Компоненты

### 1. Генератор эмбеддингов

Генерирует эмбеддинги для всех вопросов в базе данных.

**Запуск:**
```bash
python benchmarks/generate_embeddings.py --all
python benchmarks/generate_embeddings.py --score 5
python benchmarks/generate_embeddings.py --check-coverage
```

**Опции:**
- `--all`: генерировать для всех вопросов
- `--score N`: генерировать для вопросов с оценкой N (1 или 5)
- `--check-coverage`: проверить покрытие эмбеддингами
- `--overwrite`: перезаписывать существующие эмбеддинги

### 2. Бенчмарки поиска

Оценивает качество векторного поиска.

**Запуск:**
```bash
python benchmarks/run_benchmarks.py --tier 1 --dataset golden_set
python benchmarks/run_benchmarks.py --tier 2 --dataset questions_with_url
```

**Опции:**
- `--tier N`: Tier бенчмарка (1 - поиск похожих вопросов, 2 - поиск чанков)
- `--dataset DATASET`: датасет (golden_set, low_quality, questions_with_url, recent)
- `--top-k N`: количество результатов (default: 10)
- `--output-dir DIR`: директория для отчётов (default: benchmarks/reports)

### 3. Загрузчик данных

Загружает тестовые данные из базы данных.

**Методы:**
- `load_golden_set()`: загрузить вопросы с score=5
- `load_low_quality_set(limit)`: загрузить вопросы с score=1
- `load_questions_with_url()`: загрузить вопросы с URL
- `load_recent_questions(months, limit)`: загрузить последние вопросы
- `prepare_ground_truth_urls(questions)`: подготовить ground truth для оценки

### 4. Оценщик метрик

Вычисляет метрики качества поиска.

**Метрики:**
- `Recall@K`: доля релевантных документов в top-K
- `Precision@K`: точность в top-K
- `MRR`: Mean Reciprocal Rank
- `NDCG@K`: Normalized Discounted Cumulative Gain

### 5. Генератор отчётов

Генерирует отчёты в формате Markdown.

**Функции:**
- `generate_retrieval_report()`: генерация отчёта по бенчмаркам поиска
- `save_report()`: сохранение отчёта в файл
- `save_metrics_json()`: сохранение метрик в JSON

## Метрики

### Поиск (Retrieval)

| Метрика | Описание | Целевое значение |
|---------|----------|------------------|
| Recall@1 | Доля релевантных документов на первой позиции | ≥ 0.8 |
| Recall@5 | Доля релевантных документов в топ-5 | ≥ 0.9 |
| Recall@10 | Доля релевантных документов в топ-10 | ≥ 0.95 |
| Precision@5 | Точность в топ-5 | ≥ 0.8 |
| MRR | Обратный ранг первого релевантного документа | ≥ 0.8 |
| NDCG@10 | Нормализованный DCG | ≥ 0.8 |

## Использование

### Полный пайплайн

1. **Генерация эмбеддингов**
   ```bash
   python benchmarks/generate_embeddings.py --all
   ```

2. **Проверка покрытия**
   ```bash
   python benchmarks/generate_embeddings.py --check-coverage
   ```

3. **Запуск бенчмарков**
   ```bash
   python benchmarks/run_benchmarks.py --tier 1 --dataset golden_set
   python benchmarks/run_benchmarks.py --tier 2 --dataset questions_with_url
   ```

4. **Просмотр отчётов**
    ```bash
    cat benchmarks/reports/retrieval_tier1_golden_set_*.md
    ```

## Запуск системы бенчмарков

### Требования к окружению

**Обязательно:**
- Python 3.11+
- Установленные зависимости: `uv sync`
- Модель эмбеддингов: `saved_models/multilingual-e5-large-wikiutmn/`

**Опционально:**
- Запущенные контейнеры (qa, postgres)
- Сгенерированные эмбеддинги в БД

### Режимы работы

Система бенчмарков работает в двух режимах:

#### Режим 1: Без запуска проекта (CLI с готовыми датасетами)

**Когда использовать:**
- Если датасеты уже экспортированы (`benchmarks/data/raw/`)
- Если не требуется генерация эмбеддингов
- Если не нужно обновлять данные

**Пайплайн:**
```bash
cd Submodules/voproshalych

# 1. Генерация эмбеддингов (если нужно)
python benchmarks/generate_embeddings.py --all

# 2. Запуск бенчмарков с готовыми датасетами
python benchmarks/run_benchmarks.py --tier 1 --dataset golden_set
python benchmarks/run_benchmarks.py --tier 2 --dataset golden_set

# 3. Просмотр результатов
ls benchmarks/reports/*.md
```

**Преимущества:**
- ✅ Не нужно запускать Docker
- ✅ Работает с пустой локальной БД
- ✅ Использует готовые датасеты

**Ограничения:**
- ❌ Датасеты не обновляются автоматически
- ❌ Требуется ручная генерация эмбеддингов

#### Режим 2: С запуском проекта (Docker + автоматическая загрузка дампа)

**Когда использовать:**
- Если нужны свежие данные из продакшн БД
- Если нужно протестировать с актуальной базой знаний
- Если требуется интеграция с QA-сервисом

**Пайплайн:**
```bash
cd Submodules/voproshalych

# 1. Подготовка дампа БД (опционально)
# Если есть дампа из серверной БД, положите в benchmarks/data/dump/
# Скрипт автоматически загрузит его при старте QA-сервиса

# 2. Запуск проекта
docker compose up -d --build

# QA-сервис автоматически:
# - Создаст таблицы в PostgreSQL
# - Загрузит дампа БД из benchmarks/data/dump/ (если есть)
# - Генерирует эмбеддинги по запросу
# - Будет доступен для бенчмарков

# 3. Генерация эмбеддингов (если в дампе их нет)
python benchmarks/generate_embeddings.py --all

# 4. Запуск бенчмарков
python benchmarks/run_benchmarks.py --tier 1 --dataset golden_set
python benchmarks/run_benchmarks.py --tier 2 --dataset golden_set
```

**Преимущества:**
- ✅ Свежие данные из продакшн БД
- ✅ Полная интеграция с QA-сервисом
- ✅ Автоматическая загрузка дампа

**Ограничения:**
- ⚠️ Требуется запуск Docker (ресурсы)
- ⚠️ Время на развёртывание контейнеров
- ⚠️ Необходим доступ к серверной БД для дампа

### Инструкция по запуску в Режиме 1 (без Docker)

Этот режим рекомендуется для первого запуска бенчмарков.

```bash
# 1. Установить зависимости
cd Submodules/voproshalych
uv sync

# 2. Проверить наличие модели эмбеддингов
ls -la saved_models/multilingual-e5-large-wikiutmn/

# 3. Проверить наличие готовых датасетов
ls benchmarks/data/raw/
# Должно быть 10 файлов с расширением .json.txt

# 4. Проверить покрытие эмбеддингов
python benchmarks/generate_embeddings.py --check-coverage

# 5. Если покрытие < 80%, генерировать эмбеддинги
# Это можно сделать сразу или поэтапно:
python benchmarks/generate_embeddings.py --all

# 6. Запустить бенчмарки
# Tier 1 (поиск похожих вопросов)
python benchmarks/run_benchmarks.py --tier 1 --dataset golden_set

# Tier 2 (поиск чанков)
python benchmarks/run_benchmarks.py --tier 2 --dataset golden_set

# С разными датасетами
python benchmarks/run_benchmarks.py --tier 1 --dataset questions_with_url
python benchmarks/run_benchmarks.py --tier 2 --dataset recent
```

### Инструкция по запуску в Режиме 2 (с Docker)

Этот режим рекомендуется для работы с актуальными данными.

```bash
# 1. Подготовка дампа БД (опционально)
# Экспортировать дамп с серверной БД
# Сохранить в Submodules/voproshalych/benchmarks/data/dump/virtassist-main-YYYYMMDD-HHMMSS.dump

# 2. Создать файл .env с переменными окружения
cp .env.example .env
# Отредактировать POSTGRES_PASSWORD и другие параметры

# 3. Запустить проект
cd Submodules/voproshalych
docker compose up -d --build

# Проверить статус контейнеров
docker compose ps
# Должны быть running: qa, postgres

# QA-сервис автоматически загрузит дампа из benchmarks/data/dump/

# 4. Генерация эмбеддингов (если в дампе их нет)
# Через контейнер QA-сервиса
docker compose exec -it qa python benchmarks/generate_embeddings.py --all

# Или напрямую через CLI (если контейнер имеет доступ к БД)
python benchmarks/generate_embeddings.py --all

# 5. Запуск бенчмарков
python benchmarks/run_benchmarks.py --tier 1 --dataset golden_set
python benchmarks/run_benchmarks.py --tier 2 --dataset golden_set
```

### Разница между режимами

| Параметр | Режим 1 (без Docker) | Режим 2 (с Docker) |
|-----------|------------------------|----------------------|
| Запуск Docker | ❌ Не требуется | ✅ Обязательно |
| Автозагрузка дампа | ❌ Нет | ✅ Да |
| Актуальность данных | ⚠️ Может быть устаревшей | ✅ Свежие |
| Использование QA-API | ❌ Нет | ✅ Да |
| Ресурсы | ✅ Минимальные | ⚠️ Требует больше |
| Время запуска | ✅ Быстрый | ⚠️ Долгий |

### Рекомендации по выбору режима

**Используйте Режим 1, если:**
- Впервые запускаете бенчмарки
- Хотите проверить работоспособность кода
- Не нужен доступ к серверной БД
- Хотите сэкономить ресурсы

**Используйте Режим 2, если:**
- Нужны свежие данные из продакшн БД
- Хотите протестировать актуальную базу знаний
- Есть доступ к серверной БД для дампа
- Подготовливаете результаты для публикации

---

## Планируемые улучшения

### В разработке

- ⏳ Автоматическая загрузка дампа БД при запуске QA-сервиса
- ⏳ Интерактивные дашборды для просмотра метрик (Gradio, Streamlit)
- ⏳ Интеграция с CI/CD

### Будущие улучшения

- ⏳ Расширение метрик (BERTScore, ROUGE, BLEU)
- ⏳ Многоуровневые бенчмарки
- ⏳ A/B тестирование разных стратегий
- ⏳ Мониторинг и алерты

---

## Архитектура

Подробная архитектура системы бенчмарков описана в отдельном документе:

**Ссылка:** `ARCHITECTURE_BENCHMARKS.md` в `2026_tasks/LLM/task_benchmarks/`

**Содержание:**
- Общая архитектура RAG-системы
- Архитектура системы бенчмарков
- Поток данных при выполнении бенчмарков
- Диаграммы компонентов и их взаимодействия
- Критерии качества и метрики

---

### Примеры

**Бенчмарк Tier 1 с золотым набором:**
```bash
python benchmarks/run_benchmarks.py --tier 1 --dataset golden_set
```

**Бенчмарк Tier 2 с вопросами по URL:**
```bash
python benchmarks/run_benchmarks.py --tier 2 --dataset questions_with_url --top-k 10
```

**Бенчмарк с последними вопросами:**
```bash
python benchmarks/run_benchmarks.py --tier 2 --dataset recent --top-k 20
```

## Зависимости

- `sentence-transformers`: модель эмбеддингов
- `sqlalchemy`: ORM для базы данных
- `numpy`: вычисления

## Конфигурация

Конфигурация задаётся через переменные окружения:

```bash
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_HOST=localhost
POSTGRES_DB=voproshalych
```

## Логирование

Скрипты используют стандартный модуль logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
```

## Отчёты

Отчёты сохраняются в формате Markdown и JSON:

- `retrieval_tier{N}_{DATASET}_{TIMESTAMP}.md`: отчёт в формате Markdown
- `retrieval_tier{N}_{DATASET}_{TIMESTAMP}.json`: метрики в формате JSON
