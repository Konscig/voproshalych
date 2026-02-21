# Анализ дублирования кода RAG

Документ описывает места дублирования кода между модулями `benchmarks/` и `qa/` и рекомендации по рефакторингу.

---

## 1. Векторный поиск

### Дублирование

**qa/confluence_retrieving.py:117-136**
```python
def get_chunk(engine, encoder_model, question) -> Chunk | None:
    with Session(engine) as session:
        return session.scalars(
            select(Chunk)
            .order_by(Chunk.embedding.cosine_distance(encoder_model.encode(question)))
            .limit(1)
        ).first()
```

**benchmarks/models/rag_benchmark.py:180-186** (Tier 1) — свой код с top-k
```python
top_chunks = session.scalars(
    select(Chunk)
    .where(Chunk.embedding.isnot(None))
    .order_by(Chunk.embedding.cosine_distance(question_embedding))
    .limit(top_k)
).all()
```

**benchmarks/models/rag_benchmark.py** (Tier 3) — ✅ ИМПОРТИРОВАН
```python
# Импортировано из qa.confluence_retrieving:
from qa.confluence_retrieving import get_chunk

# Использование:
retrieved_chunk = get_chunk(self.engine, self.encoder, question)
```

**benchmarks/models/real_queries_benchmark.py:75-80** — свой код с top-k
```python
chunks = session.scalars(
    select(Chunk)
    .where(Chunk.embedding.isnot(None))
    .order_by(Chunk.embedding.cosine_distance(embedding))
    .limit(top_k)
).all()
```

### Рекомендация

**Текущий статус (после рефакторинга):**
- ✅ Tier 3 — импортирует `get_chunk` из `qa/confluence_retrieving.py`
- ⚠️ Tier 1 — свой код с top-k
- ⚠️ Real Users — свой код с top-k
- ⚠️ Production — `get_chunk` (1 чанк)

**План на будущее (когда production будет поддерживать top-k):**

Создать общий модуль `qa/retrieval.py`:

```python
def retrieve_top_k_chunks(
    engine: Engine,
    encoder: SentenceTransformer,
    query: str,
    top_k: int = 10,
) -> List[Chunk]:
    """Получить top-k чанков по косинусному расстоянию."""
    embedding = encoder.encode(query)
    with Session(engine) as session:
        return session.scalars(
            select(Chunk)
            .where(Chunk.embedding.isnot(None))
            .order_by(Chunk.embedding.cosine_distance(embedding))
            .limit(top_k)
        ).all()
```

Использовать в:
- `qa/confluence_retrieving.py:get_chunk()` → `retrieve_top_k_chunks(..., top_k=1)[0]`
- `benchmarks/models/rag_benchmark.py:run_tier_1()` → `retrieve_top_k_chunks()`
- `benchmarks/models/rag_benchmark.py:run_tier_3()` → `retrieve_top_k_chunks(..., top_k=1)`
- `benchmarks/models/real_queries_benchmark.py:run()` → `retrieve_top_k_chunks()`

**Примечание:** Пока production использует только 1 чанк (`get_chunk` с `limit(1)`), бенчмарки:
- Tier 3 точно воспроизводит production через импорт
- Tier 1 и Real Users позволяют тестировать top-k метрики отдельно

---

## 2. Генерация ответов

### Текущее состояние

**qa/main.py:47-84** — функция `get_answer(dialog_history, knowledge_base, question)`

**benchmarks/models/rag_benchmark.py:54-72** — обёртка над `qa.main.get_answer`

```python
if get_answer_func is None:
    from qa.main import get_answer
    def _get_answer_wrapper(question: str, context: str) -> str:
        return get_answer(dialog_history=[], knowledge_base=context, question=question)
    self.get_answer_func = _get_answer_wrapper
```

### Статус: ✅ Нет дублирования

Код уже использует импорт из `qa.main`. Это правильный подход.

---

## 3. LLM Judge

### Дублирование

**qa/main.py:87-134** — функция `assess_answer(dialog_history, question, answer, content, generation)`

**benchmarks/utils/llm_judge.py** — класс `LLMJudge` с методами:
- `evaluate_faithfulness()`
- `evaluate_answer_relevance()`
- `evaluate_e2e_quality()`

### Различия

| Аспект | qa/main.py:assess_answer | benchmarks/utils/llm_judge.py |
|--------|--------------------------|-------------------------------|
| Назначение | Оценка "показывать/не показывать" ответ пользователю | Оценка качества ответа 1-5 |
| Модель | Mistral (JUDGE_API) | DeepSeek/Qwen (BENCHMARKS_JUDGE_API_KEY) |
| Промпт | Yes/No бинарный | Шкала 1-5 |
| Результат | bool | float |
| Использование | Production RAG pipeline | Tier Judge benchmark |

### Рекомендация

Не объединять. Это разные сценарии:
- `assess_answer` — production judge (Mistral) для фильтрации ответов в реальном пайплайне
- `LLMJudge` — benchmark judge (Qwen) для метрик качества в Tier Judge
- `run_tier_judge_pipeline` — тестирование production judge через сравнение с ground truth

### Два типа judge в бенчмарках

| Tier | Переменные окружения | Модель | Назначение |
|------|---------------------|--------|------------|
| Tier Judge | BENCHMARKS_JUDGE_API_KEY, BENCHMARKS_JUDGE_MODEL | Qwen/DeepSeek | Тестирование согласованности judge |
| Tier Judge Pipeline | JUDGE_API, JUDGE_MODEL | Mistral | Тестирование production pipeline judge |

---

## 4. Эмбеддинги

### Текущее состояние

- `qa/main.py:44` — загрузка модели `encoder_model`
- `benchmarks/run_comprehensive_benchmark.py:418` — загрузка модели `encoder`

### Дублирование: минимальное

Оба модуля используют одну и ту же модель из `Config.EMBEDDING_MODEL_PATH`. Это нормально.

---

## 5. Конфигурация

### Текущее состояние

Оба модуля используют `qa/config.py:Config` для получения настроек.

### Статус: ✅ Нет дублирования

---

## Итог

| Компонент | Дублирование | Приоритет рефакторинга |
|-----------|--------------|------------------------|
| Векторный поиск | ⚠️ Высокое | Высокий |
| Генерация ответов | ✅ Нет | — |
| LLM Judge | ✅ Разные сценарии | — |
| Эмбеддинги | ✅ Минимальное | — |
| Конфигурация | ✅ Нет | — |

---

## План рефакторинга

### Шаг 1: Создать qa/retrieval.py

Вынести функции векторного поиска в отдельный модуль.

### Шаг 2: Обновить qa/confluence_retrieving.py

Использовать функции из `qa/retrieval.py`.

### Шаг 3: Обновить benchmarks/models/rag_benchmark.py

Импортировать `retrieve_top_k_chunks` из `qa.retrieval`.

### Шаг 4: Обновить benchmarks/models/real_queries_benchmark.py

Импортировать `retrieve_top_k_chunks` из `qa.retrieval`.

### Шаг 5: Добавить тесты

Покрыть `qa/retrieval.py` unit-тестами.

---

## Связанные документы

- [ARCHITECTURE.md](ARCHITECTURE.md) — архитектура модуля
- [CLI_REFERENCE.md](CLI_REFERENCE.md) — CLI команды
