# Руководство по ручной аннотации датасета для RAG-бенчмарков

## Обзор

Ручная аннотация — это процесс **просмотра и уточнения уже сгенерированных датасетов**, а не создание датасета с нуля.

### Основные сценарии

1. **Synthetic dataset** — сгенерирован из чанков через LLM
2. **Manual dataset** — вручную аннотированный synthetic датасет

---

## Генерация датасета

### Режимы генерации

```bash
# Synthetic — вопросы генерируются из чанков
uv run python benchmarks/generate_dataset.py \
    --mode synthetic \
    --max-questions 500
```

### Структура датасета

Каждая запись содержит расширенные метаданные:

```json
{
  "id": "syn_a1b2c3d4e5f6",
  "source": "synthetic",
  "question_source": "synthetic|manual",
  "question": "Вопрос пользователя",
  "ground_truth_answer": "Эталонный ответ",
  "chunk_id": 12345,
  "chunk_text": "Текст чанка...",
  "confluence_url": "https://...",
  "relevant_chunk_ids": [123, 456],
  "user_score": 5,
  "question_answer_id": 789,
  "notes": "Комментарий аннотатора"
}
```

Поля:
- `id` — уникальный идентификатор записи
- `source` — источник данных (synthetic)
- `question_source` — происхождение вопроса (synthetic/manual)

---

## Ручная аннотация

### Экспорт для аннотации

```bash
# Экспорт датасета в файл для ручного редактирования
uv run python benchmarks/generate_dataset.py \
    --mode export-annotation \
    --output benchmarks/data/dataset_20260220_123456.json
```

### Поля для аннотации

При аннотации заполняются поля:

| Поле | Тип | Описание |
|------|-----|----------|
| `is_question_ok` | 0/1 | Корректность вопроса |
| `is_answer_ok` | 0/1 | Корректность ответа (ground_truth_answer) |
| `is_chunk_ok` | 0/1 | Корректность источника (chunk) |
| `notes` | строка | Комментарии аннотатора |

### Использование в бенчмарках

```bash
# Запуск на аннотированном датасете
uv run python benchmarks/run_comprehensive_benchmark.py \
    --mode manual \
    --manual-dataset benchmarks/data/dataset_annotated.json \
    --tier all \
    --judge-eval-mode reasoned
```

### Рекомендация по LLM Judge для аннотированного synthetic датасета

Для вручную проверенного synthetic-датасета рекомендуемый режим:

- `--mode manual`
- `--judge-eval-mode reasoned`
- `--consistency-runs 2` (или выше для контроля воспроизводимости)

Пример:

```bash
uv run python benchmarks/run_comprehensive_benchmark.py \
    --tier all \
    --mode manual \
    --manual-dataset benchmarks/data/dataset_annotated.json \
    --judge-eval-mode reasoned \
    --consistency-runs 2
```

Такой запуск повышает строгость оценки при работе с датасетом,
проверенным доменным экспертом.

---

## Цикл работы

```
1. Генерация датасета
   │
   └── synthetic (из чанков)
   │
   ▼
2. Экспорт для аннотации
   │
   ▼
3. Ручная правка (аннотатор)
   │
   ├── Проверка вопросов
   ├── Проверка ответов
   └── Добавление комментариев
   │
   ▼
4. Запуск бенчмарков
   │
   ├── Tier 1 (Retrieval)
   ├── Tier 2 (Generation)
   ├── Tier Judge (сравнение с человеком)
   └── Tier UX (user experience)
```

---

## Принципы аннотации

1. **Фактичность**: ответ должен опираться только на подтверждённые источники
2. **Полнота**: указывайте все релевантные `chunk_id` и `relevant_urls`
3. **Консистентность**: одинаковые типы вопросов размечайте одинаково
4. **Трассируемость**: фиксируйте неоднозначные решения в `notes`

---

## Хранение аннотаций

Аннотации хранятся в таблице БД `benchmark_annotations`:

```sql
SELECT * FROM benchmark_annotations 
WHERE dataset_file = 'dataset_20260220_123456.json';
```

Поля таблицы:
- `dataset_file` — имя файла датасета
- `item_id` — id записи из датасета
- `is_question_ok`, `is_answer_ok`, `is_chunk_ok` — бинарные метки
- `notes` — текстовые заметки
- `annotator` — имя аннотатора
- `created_at`, `updated_at` — временные метки
