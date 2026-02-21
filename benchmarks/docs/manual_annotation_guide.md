# Руководство по ручной аннотации датасета для RAG-бенчмарков

## Обзор

Ручная аннотация — это процесс **просмотра и уточнения уже сгенерированных датасетов**, а не создание датасета с нуля.

### Основные сценарии

1. **Synthetic dataset** — сгенерирован из чанков через LLM
2. **Real-users dataset** — на основе реальных вопросов пользователей
3. **Score-5 dataset** — только вопросы с оценкой 5 (положительный фидбек)

---

## Генерация датасета

### Режимы генерации

```bash
# 1. Synthetic — вопросы генерируются из чанков (как раньше)
uv run python benchmarks/generate_dataset.py \
    --mode synthetic \
    --max-questions 500

# 2. Из реальных вопросов пользователей
uv run python benchmarks/generate_dataset.py \
    --mode from-real-questions \
    --max-questions 500

# 3. Только вопросы с оценкой score=5
uv run python benchmarks/generate_dataset.py \
    --mode from-real-questions-score-5 \
    --max-questions 500
```

### Структура датасета

Каждая запись содержит расширенные метаданные:

```json
{
  "id": "syn_a1b2c3d4e5f6",
  "source": "synthetic|real-users",
  "question_source": "synthetic|real-user|manual",
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
- `source` — источник данных (synthetic, real-users)
- `question_source` — происхождение вопроса (synthetic, real-user, manual)

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
    --tier all
```

---

## Цикл работы

```
1. Генерация датасета
   │
   ├── synthetic (из чанков)
   ├── from-real-questions (все вопросы)
   └── from-real-questions-score-5 (только score=5)
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
