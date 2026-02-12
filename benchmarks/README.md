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
