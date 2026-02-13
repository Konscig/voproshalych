# Бенчарки Вопрошалыча

## Задача

Система бенчарков для оценки качества RAG-системы Вопрошалыча.

## Зачем

- Оценивать качество поиска по базе знаний
- Отслеживать деградацию метрик
- Сравнивать разные конфигурации системы
- Обеспечивать объективные метрики качества

## Принципы работы

1. **Автоматизация**
   - CLI-интерфейс для всех операций
   - Генерация отчётов в Markdown и JSON
   - Логирование всех операций

2. **Независимость от данных**
   - Работает с готовой БД
   - Поддерживает статичные датасеты
   - Не требует запуска QA-сервиса

3. **Метрики**
   - Recall@K: доля релевантных в топ-K
   - Precision@K: точность в топ-K
   - MRR: обратный ранг первого релевантного
   - NDCG@K: нормализованный DCG

## Структура

```
benchmarks/
├── data/                       # Данные для бенчарков
│   └── static/               # Статичные неизменяемые датасеты
├── models/                     # Модели бенчарков
│   └── retrieval_benchmark.py
├── utils/                      # Утилиты
│   ├── evaluator.py
│   ├── static_data_loader.py
│   ├── embedding_generator.py
│   ├── database_dump_loader.py
│   └── report_generator.py
├── reports/                     # Отчёты бенчарков
│   └── summaries/                    # Саммари запусков
│       ├── README.md              # Шаблон для саммари
│       └── YYYY-MM-DD_*.md            # Саммари по датам
└── SUMMARY.md                  # Шаблон для анализа саммари
└── architecture_snapshots/        # Версии архитектуры
│       └── README.md              # Правила версионирования
│       └── 2026-02-13_Architecture_Audit.md
└── generate_embeddings.py         # CLI: генерация эмбеддингов
└── run_benchmarks.py               # CLI: запуск бенчарков
└── load_database_dump.py         # CLI: загрузка дампа
└── run_dashboard.py                # Скрипт для дашборда
└── DASHBORAD.md                  # Документация по дашборду
└── README.md                       # Документация бенчарков
└── 01_Изоляция_проектных_файлов_в_директорию_бенчарков.md # Анализ изоляции
└── architecture_snapshots/       # Версии архитектуры
```

## Использование

### Генерация эмбеддингов

```bash
# Проверить покрытие
uv run python benchmarks/generate_embeddings.py --check-coverage

# Сгенерировать для всех
uv run python benchmarks/generate_embeddings.py --all

# Сгенерировать для score=5
uv run python benchmarks/generate_embeddings.py --score 5
```

### Запуск бенчарков

```bash
# Tier 1 (поиск похожих вопросов)
uv run python benchmarks/run_benchmarks.py --tier 1 --dataset golden_set
uv run python benchmarks/run_benchmarks.py --tier 1 --dataset low_quality

# Tier 2 (поиск чанков)
uv run python benchmarks/run_benchmarks.py --tier 2 --dataset golden_set
uv run python benchmarks/run_benchmarks.py --tier 2 --dataset low_quality
```

### Интерактивный дашборд

```bash
# Запуск дашборда
cd Submodules/voproshalych
uv run python benchmarks/run_dashboard.py
```

**Возможности дашборда:**
- Просмотр текущих результатов бенчарков
- Сравнение разных запусков
- Графики метрик по времени
- Фильтрация по датасетам и типам бенчарков

### Загрузка дампа БД

```bash
# Загрузить дамп с удалением таблиц
uv run python benchmarks/load_database_dump.py --drop-tables --dump benchmarks/data/dump/virtassist_backup.dump
```

## Метрики

### Tier 1: Поиск похожих вопросов

| Метрика | Отлично | Хорошо | Средне | Плохо |
|---------|---------|---------|---------|--------|
| MRR | ≥0.8 | ≥0.6 | ≥0.4 | <0.4 |
| Recall@1 | ≥80% | ≥60% | ≥40% | <40% |
| Precision@5 | ≥80% | ≥60% | ≥40% | <40% |

### Tier 2: Поиск чанков

| Метрика | Отлично | Хорошо | Средне | Плохо |
|---------|---------|---------|---------|--------|
| MRR | ≥0.5 | ≥0.4 | ≥0.3 | <0.3 |
| Recall@10 | ≥70% | ≥50% | ≥30% | <30% |
| Precision@5 | ≥50% | ≥40% | ≥30% | <30% |

## Статус реализации

### Полностью реализовано
- ✅ Генерация эмбеддингов
- ✅ Бенчмарк Tier 1 (поиск вопросов)
- ✅ Бенчмарк Tier 2 (поиск чанков)
- ✅ Вычисление метрик (Recall, Precision, MRR, NDCG)
- ✅ Генерация отчётов (Markdown, JSON)
- ✅ Статичные датасеты
- ✅ Загрузка дампа БД

### В разработке
- ⏳ Увеличение покрытия эмбеддингами (текущее: 4.78%, цель: 80%)
- ⏳ Улучшение чанковедения
- ⏳ Интеграция дашборда с CLI

### Планируется
- ⏳ Векторная база данных (Qdrant, Milvus)
- ⏳ A/B тестирование
- ⏳ Больше метрик (MAP, F1-Score)
- ⏳ CI/CD интеграция

## Зависимости

- Python 3.11+
- sentence-transformers
- sqlalchemy
- psycopg2-binary
- pgvector
- numpy
- streamlit

## Документация

- **Архитектура:** `architecture_snapshots/2026-02-13_Architecture_Audit.md`
- **Реализация:** `README.md`
- **Анализ структуры:** `01_Изоляция_проектных_файлов_в_директорию_бенчарков.md`

## Структура хранения отчётов

```
2026_benchmarks/reports/
├── summaries/                    # Саммари запусков
│   ├── README.md              # Шаблон для саммари
│   └── YYYY-MM-DD_*.md            # Саммари по датам
└── SUMMARY.md                  # Шаблон для анализа саммари
```
