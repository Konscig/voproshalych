# Архитектура модуля бенчмарков RAG-системы

Подробное описание архитектуры модуля бенчмарков Вопрошалыч.

---

## Диаграмма пайплайна

```mermaid
sequenceDiagram
    autonumber
    participant User as Пользователь
    participant CLI as CLI скрипты
    participant Generator as EmbeddingGenerator
    participant Judge as LLMJudge
    participant RAGBenchmark as RAGBenchmark
    participant RealBenchmark as RealQueriesBenchmark
    participant QA as qa.main
    participant DB as PostgreSQL
    participant Dashboard as Dashboard

    Note over User,Dashboard: Фаза 1: Генерация эмбеддингов

    User->>CLI: generate_embeddings.py --chunks
    CLI->>Generator: generate_for_chunks
    Generator->>DB: SELECT Chunk WHERE embedding IS NULL
    DB-->>Generator: chunks without embeddings
    loop Для каждого чанка
        Generator->>Generator: encoder.encode(chunk.text)
        Generator->>DB: UPDATE Chunk SET embedding
    end
    Generator-->>CLI: stats
    CLI-->>User: Статистика генерации

    Note over User,Dashboard: Фаза 2: Генерация датасета (synthetic)

    User->>CLI: generate_dataset.py --max-questions N
    CLI->>DB: SELECT Chunk WHERE embedding IS NOT NULL
    DB-->>CLI: chunks with embeddings
    loop Для каждого чанка до N вопросов
        CLI->>Judge: generate_question_from_chunk(chunk.text)
        Judge-->>CLI: question, ground_truth_answer
        CLI->>CLI: dataset.append(item)
    end
    CLI->>CLI: Сохранить dataset_*.json
    CLI-->>User: Путь к датасету

    Note over User,Dashboard: Фаза 3: Запуск бенчмарков (Tier 1)

    User->>CLI: run_comprehensive_benchmark.py --tier 1
    CLI->>CLI: load_dataset(dataset_path)
    CLI->>RAGBenchmark: run_tier_1(dataset, top_k)
    loop Для каждого вопроса в датасете
        RAGBenchmark->>RAGBenchmark: encoder.encode(question)
        RAGBenchmark->>DB: SELECT Chunk ORDER BY cosine_distance LIMIT top_k
        DB-->>RAGBenchmark: top_k chunks
        RAGBenchmark->>RAGBenchmark: compute hit_rate, mrr, ndcg
    end
    RAGBenchmark-->>CLI: tier_1_metrics
    CLI->>DB: INSERT INTO benchmark_runs

    Note over User,Dashboard: Фаза 4: Запуск бенчмарков (Tier 2)

    User->>CLI: run_comprehensive_benchmark.py --tier 2
    CLI->>RAGBenchmark: run_tier_2(dataset)
    loop Для каждого вопроса в датасете
        RAGBenchmark->>RAGBenchmark: resolve_context(item)
        RAGBenchmark->>QA: get_answer(question, context)
        QA-->>RAGBenchmark: system_answer
        RAGBenchmark->>Judge: evaluate_faithfulness(question, context, answer)
        Judge-->>RAGBenchmark: faithfulness_score
        RAGBenchmark->>Judge: evaluate_answer_relevance(question, answer)
        Judge-->>RAGBenchmark: relevance_score
    end
    RAGBenchmark-->>CLI: tier_2_metrics
    CLI->>DB: INSERT INTO benchmark_runs

    Note over User,Dashboard: Фаза 5: Запуск бенчмарков (Tier 3)

    User->>CLI: run_comprehensive_benchmark.py --tier 3
    CLI->>RAGBenchmark: run_tier_3(dataset)
    loop Для каждого вопроса в датасете
        RAGBenchmark->>RAGBenchmark: encoder.encode(question)
        RAGBenchmark->>DB: SELECT Chunk ORDER BY cosine_distance LIMIT 1
        DB-->>RAGBenchmark: retrieved_chunk
        RAGBenchmark->>QA: get_answer(question, chunk.text)
        QA-->>RAGBenchmark: system_answer
        RAGBenchmark->>Judge: evaluate_e2e_quality(question, system_answer, ground_truth)
        Judge-->>RAGBenchmark: e2e_score
        RAGBenchmark->>RAGBenchmark: cosine_similarity(system_answer, ground_truth)
    end
    RAGBenchmark-->>CLI: tier_3_metrics
    CLI->>DB: INSERT INTO benchmark_runs

    Note over User,Dashboard: Фаза 6: Real-users benchmark

    User->>CLI: run_comprehensive_benchmark.py --mode real-users
    CLI->>RealBenchmark: run(top_k, score_filter, limit)
    RealBenchmark->>DB: SELECT QuestionAnswer WHERE score=5 AND embedding IS NOT NULL
    DB-->>RealBenchmark: real_questions
    loop Для каждого реального вопроса
        RealBenchmark->>RealBenchmark: encoder.encode(question)
        RealBenchmark->>DB: SELECT Chunk ORDER BY cosine_distance LIMIT top_k
        DB-->>RealBenchmark: retrieved_chunks
        RealBenchmark->>RealBenchmark: compare with QuestionAnswer.confluence_url
        RealBenchmark->>RealBenchmark: compute recall, precision, mrr, ndcg
    end
    RealBenchmark-->>CLI: real_user_metrics
    CLI->>DB: INSERT INTO benchmark_runs

    Note over User,Dashboard: Фаза 7: Просмотр результатов

    User->>CLI: run_dashboard.py
    CLI->>Dashboard: main
    Dashboard->>DB: SELECT benchmark_runs ORDER BY timestamp
    DB-->>Dashboard: all_runs
    Dashboard->>Dashboard: create_interface
    Dashboard-->>User: Gradio UI на http://localhost:7860
```

---

## Описание шагов пайплайна

### Фаза 1: Генерация эмбеддингов

**Шаг 1-2:** Пользователь запускает `generate_embeddings.py --chunks`. CLI создаёт экземпляр `EmbeddingGenerator`.

**Шаг 3:** `EmbeddingGenerator` запрашивает из БД все чанки без эмбеддингов.

**Шаг 4-6:** Для каждого чанка генерируется эмбеддинг через `SentenceTransformer` и сохраняется в БД.

**Шаг 7-8:** Возвращается статистика: total, processed, skipped, errors.

**Реализация:** `benchmarks/utils/embedding_generator.py`, метод `generate_for_chunks()`.

---

### Фаза 2: Генерация датасета (synthetic)

**Шаг 9-10:** Пользователь запускает `generate_dataset.py --max-questions N`. CLI запрашивает чанки с эмбеддингами.

**Шаг 11-14:** Для каждого чанка (до достижения N) вызывается `LLMJudge.generate_question_from_chunk()`, который генерирует вопрос и идеальный ответ на основе текста чанка.

**Шаг 15-16:** Датасет сохраняется в файл `benchmarks/data/dataset_YYYYMMDD_HHMMSS.json`.

**Реализация:** `benchmarks/generate_dataset.py`, функция `generate_synthetic_dataset()`.

### Альтернативные режимы генерации

| Режим | Команда | Описание |
|-------|---------|----------|
| Synthetic | `--mode synthetic` | Генерация вопросов из чанков через LLM |
| From Real Questions | `--mode from-real-questions` | Использование реальных вопросов из QuestionAnswer |
| Score 5 | `--mode from-real-questions-score-5` | Только вопросы с оценкой score=5 |
| Export Annotation | `--mode export-annotation` | Экспорт для ручной аннотации |

При использовании режимов `from-real-questions` или `from-real-questions-score-5`:
- Вопросы берутся из таблицы `QuestionAnswer`
- Для каждого вопроса выполняется similarity search для нахождения релевантных чанков
- Ground truth answer берётся из `QuestionAnswer.answer`
- Сохраняются `user_score` и `question_answer_id` для трассировки

---

### Фаза 3: Запуск бенчмарков (Tier 1)

**Шаг 17-19:** Пользователь запускает `run_comprehensive_benchmark.py --tier 1`. Загружается датасет и создаётся `RAGBenchmark`.

**Шаг 20-23:** Для каждого вопроса:
- Генерируется эмбеддинг вопроса
- Выполняется векторный поиск через `Chunk.embedding.cosine_distance()`
- Вычисляются метрики: HitRate@K, MRR, NDCG@K

**Шаг 24-25:** Результаты сохраняются в `benchmark_runs` таблицу.

**Реализация:** `benchmarks/models/rag_benchmark.py`, метод `run_tier_1()`.

---

### Фаза 4: Запуск бенчмарков (Tier 2)

**Шаг 26-28:** Пользователь запускает `run_comprehensive_benchmark.py --tier 2`. Создаётся `RAGBenchmark`.

**Шаг 29-33:** Для каждого вопроса:
- Определяется контекст из датасета (`chunk_text`, `relevant_chunk_ids`, или `chunk_id`)
- Вызывается `qa.main.get_answer()` для генерации ответа (импорт из qa!)
- `LLMJudge` оценивает faithfulness (фактичность) и answer_relevance (релевантность)
- Вычисляются алгоритмические метрики: ROUGE-1, ROUGE-L, BLEU

**Шаг 34-35:** Результаты сохраняются.

**Реализация:** `benchmarks/models/rag_benchmark.py`, метод `run_tier_2()`, использует импорт `get_answer` из `qa.main`.

---

### Фаза 5: Запуск бенчмарков (Tier 3)

**Шаг 36-38:** Пользователь запускает `run_comprehensive_benchmark.py --tier 3`.

**Шаг 39-44:** Для каждого вопроса:
- Выполняется retrieval через `qa.confluence_retrieving.get_chunk()` (импорт из production!)
- Генерируется ответ через `qa.main.get_answer()`
- `LLMJudge` оценивает E2E качество
- Вычисляется cosine similarity между системным и эталонным ответом
- Вычисляются алгоритмические метрики: ROUGE-1, ROUGE-L, BLEU

**Шаг 45-46:** Результаты сохраняются.

**Реализация:** `benchmarks/models/rag_benchmark.py`, метод `run_tier_3()`, использует импорт `get_chunk` из `qa.confluence_retrieving`.

**Важно:** Tier 3 точно воспроизводит текущую production реализацию (1 чанк, top-1).

---

### Фаза 6: Real-users benchmark

**Шаг 47-49:** Пользователь запускает `run_comprehensive_benchmark.py --mode real-users`. Создаётся `RealQueriesBenchmark`.

**Шаг 50-54:** Для каждого реального вопроса из `QuestionAnswer` с `score=5`:
- Генерируется эмбеддинг вопроса
- Выполняется retrieval top-k
- Retrieved URL сравниваются с `QuestionAnswer.confluence_url`
- Вычисляются retrieval-метрики

**Шаг 55-56:** Результаты сохраняются.

**Реализация:** `benchmarks/models/real_queries_benchmark.py`, метод `run()`.

---

### Фаза 6a: Запуск бенчмарков (Tier 0)

**Шаг 56a:** Пользователь запускает `run_comprehensive_benchmark.py --tier 0`. Создаётся `RAGBenchmark`.

**Шаг 57-59:** Для кластерного анализа эмбеддингов:
- Извлекаются тексты и метки кластеров из датасета
- Вычисляются эмбеддинги для каждого текста
- Рассчитываются intra-cluster similarity (внутриклассовое сходство)
- Рассчитываются inter-cluster distance (межклассовое расстояние)
- Вычисляется silhouette score

**Шаг 60:** Результаты сохраняются.

**Реализация:** `benchmarks/models/rag_benchmark.py`, метод `run_tier_0()`.

---

### Фаза 6b: Запуск бенчмарков (Tier Judge)

**Шаг 61:** Пользователь запускает `run_comprehensive_benchmark.py --tier judge`. Создаётся `RAGBenchmark`.

**Шаг 62-64:** Для оценки качества LLM-судьи:
- Используется `LLMJudge` из `benchmarks.utils.llm_judge` (свой код, НЕ из qa!)
- Вызывается `judge.evaluate_faithfulness()` первый раз
- При включенной проверке консистентности — повторный вызов
- Сравниваются оценки для вычисления consistency score
- Замеряется latency каждого вызова

**Шаг 65:** Результаты сохраняются.

**Реализация:** `benchmarks/models/rag_benchmark.py`, метод `run_tier_judge()`, использует `benchmarks.utils.llm_judge.LLMJudge`.

---

### Фаза 6b: Запуск бенчмарков (Tier Judge Pipeline)

**Шаг 62:** Пользователь запускает `run_comprehensive_benchmark.py --tier judge_pipeline`. Создаётся `RAGBenchmark`.

**Шаг 63-65:** Для оценки production judge (Mistral):
- Загружается датасет `dataset_judge_pipeline_*.json` с парами (question, answer, ground_truth_show)
- Для каждой пары вызывается production judge через `qa.config.Config.get_judge_prompt()` и `Config.get_judge_headers()` (импорт из qa!)
- Сравнивается решение judge ("Yes"/"No") с ground truth
- Вычисляются метрики: accuracy, precision, recall, F1

**Шаг 66:** Результаты сохраняются.

**Реализация:** `benchmarks/models/rag_benchmark.py`, метод `run_tier_judge_pipeline()`.

**Отличия от Tier Judge:**
- **Tier Judge** тестирует `BENCHMARKS_JUDGE_*` (Qwen) — свой код в benchmarks, НЕ импортируется из qa
- **Tier Judge Pipeline** тестирует `JUDGE_*` (Mistral) — импорт из `qa.config.Config` — тестируется production judge

---

### Фаза 6c: Запуск бенчмарков (Tier UX)

**Шаг 66:** Пользователь запускает `run_comprehensive_benchmark.py --tier ux`. Создаётся `RAGBenchmark`.

**Шаг 67-69:** Для оценки пользовательского опыта:
- Загружаются вопросы с `score=5` из `QuestionAnswer`
- Для каждого вопроса в сессии ищется похожий вопрос в кэше
- Вычисляется cache hit rate (попадания в кэш)
- Для многотуровых сессий вычисляется context preservation

**Шаг 70:** Результаты сохраняются.

**Реализация:** `benchmarks/models/rag_benchmark.py`, метод `run_tier_ux()`.

---

### Фаза 7: Просмотр результатов

**Шаг 57-60:** Пользователь запускает `run_dashboard.py`. Dashboard загружает все запуски из `benchmark_runs` и создаёт Gradio интерфейс.

**Реализация:** `benchmarks/dashboard.py`, класс `RAGBenchmarkDashboard`.

---

## Компоненты системы

### RAG-модуль (qa/)

| Файл | Назначение |
|------|------------|
| `qa/main.py` | Генерация ответов через Mistral API |
| `qa/config.py` | Конфигурация, промпты, переменные окружения |
| `qa/database.py` | Модели SQLAlchemy (Chunk, QuestionAnswer, BenchmarkRun) |

### Benchmark-модуль (benchmarks/)

| Файл | Назначение |
|------|------------|
| `run_comprehensive_benchmark.py` | CLI для запуска бенчмарков |
| `generate_dataset.py` | Генерация synthetic датасета |
| `generate_embeddings.py` | Генерация эмбеддингов |
| `run_dashboard.py` | Запуск дашборда |
| `dashboard.py` | Gradio дашборд |
| `models/rag_benchmark.py` | Tier 0/1/2/3/Judge/UX бенчмарки |
| `models/real_queries_benchmark.py` | Real-users benchmark |
| `utils/llm_judge.py` | LLM-судья для оценки качества |
| `utils/evaluator.py` | Метрики: HitRate, MRR, NDCG |
| `utils/text_metrics.py` | Алгоритмические метрики: ROUGE, BLEU, BERTScore |
| `utils/embedding_generator.py` | Генерация эмбеддингов |

---

## Таблицы базы данных

### chunk

| Поле | Тип | Описание |
|------|-----|----------|
| id | int | PK |
| confluence_url | text | URL источника |
| text | text | Текст чанка |
| embedding | vector(1024) | Векторное представление |

### question_answer

| Поле | Тип | Описание |
|------|-----|----------|
| id | int | PK |
| question | text | Вопрос пользователя |
| answer | text | Ответ системы |
| confluence_url | text | URL источника |
| score | int | Оценка пользователя (1-5) |
| embedding | vector(1024) | Вектор вопроса |
| user_id | int | FK -> user.id |

### benchmark_runs

| Поле | Тип | Описание |
|------|-----|----------|
| id | int | PK |
| timestamp | datetime | Время запуска |
| git_branch | text | Ветка git |
| git_commit_hash | text | Hash коммита |
| run_author | text | Автор запуска |
| dataset_file | text | Имя файла датасета |
| dataset_type | text | synthetic/manual/real-users |
| judge_model | text | Модель для судейства (Qwen) |
| generation_model | text | Модель для генерации (Mistral) |
| tier_0_metrics | json | Метрики Tier 0 (Embedding Quality) |
| tier_1_metrics | json | Метрики Tier 1 (Retrieval) |
| tier_2_metrics | json | Метрики Tier 2 (Generation) |
| tier_3_metrics | json | Метрики Tier 3 (End-to-End) |
| tier_judge_metrics | json | Метрики Tier Judge (LLM Quality, Qwen) |
| tier_judge_pipeline_metrics | json | Метрики Tier Judge Pipeline (Production Judge, Mistral) |
| tier_ux_metrics | json | Метрики Tier UX (User Experience) |
| real_user_metrics | json | Метрики real-users |
| overall_status | text | Итоговый статус |

---

## Связанные документы

- [RUN_MODES_LOCAL_VS_DOCKER.md](RUN_MODES_LOCAL_VS_DOCKER.md) — режимы работы
- [CLI_REFERENCE.md](CLI_REFERENCE.md) — справочник по CLI командам
- [SMOKE_SCENARIO_LOCAL.md](SMOKE_SCENARIO_LOCAL.md) — инструкции для локального режима
- [SMOKE_SCENARIO_DOCKER.md](SMOKE_SCENARIO_DOCKER.md) — инструкции для Docker режима
