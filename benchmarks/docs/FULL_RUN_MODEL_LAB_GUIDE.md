# Полный прогон и Model Lab: финальный сбор всех метрик

Практическое руководство для полного прогона benchmark-системы и сравнения моделей (generation / benchmark judge / production judge) в одном цикле.

Документ ориентирован на **полный режим**:

- synthetic датасет на всех чанках (в текущей БД: около 202);
- source analytics по всей таблице `question_answer` (5000+ вопросов);
- сравнение нескольких наборов моделей (до ~10 в каждой категории).

---

## 1. Предпосылки

- Ветка: `feat/benchmark-rag-quality`
- Рабочая директория: `Submodules/voproshalych`
- Используемый режим:
  - `docker-compose.benchmarks.yml` поднимает только `db` + `db-migrate`
  - все benchmark-скрипты запускаются локально через `uv run`

---

## 2. Подготовка API ключей и моделей

Скопируйте шаблон и заполните ключи:

```bash
cd Submodules/voproshalych
cp .env.benchmark-models.example .env.benchmark-models
```

Заполните production и compare секции в `.env.benchmark-models`.

Важно: вы можете хранить **все ключи разом** в одном файле
(`PROVIDER_MISTRAL_API_KEY`, `PROVIDER_OPENROUTER_API_KEY`,
 `PROVIDER_DEEPSEEK_API_KEY`, `PROVIDER_ALIBABA_API_KEY`) и выбирать нужный
токен на запуск через `--generation-api-key-var`,
`--production-judge-api-key-var`, `--benchmark-judge-api-key-var`.

Также вы можете определить списки моделей в переменных окружения:
- `MISTRAL_GEN_MODELS`, `OPENROUTER_GEN_MODELS`, `DEEPSEEK_GEN_MODELS`, `ALIBABA_GEN_MODELS`
- `MISTRAL_JUDGE_MODELS`, `OPENROUTER_JUDGE_MODELS`, `DEEPSEEK_JUDGE_MODELS`, `ALIBABA_JUDGE_MODELS`
- `MISTRAL_BM_JUDGE_MODELS`, `OPENROUTER_BM_JUDGE_MODELS`, `DEEPSEEK_BM_JUDGE_MODELS`, `ALIBABA_BM_JUDGE_MODELS`

И использовать их через флаги:
- `--generation-model-source mistral` — загрузит все модели из `MISTRAL_GEN_MODELS`
- `--production-judge-model-source openrouter` — загрузит все модели из `OPENROUTER_JUDGE_MODELS`
- `--judge-model-source deepseek` — загрузит все модели из `DEEPSEEK_BM_JUDGE_MODELS`

Важно:

- `.env.benchmark-models` не коммитится;
- раннер автоматически подхватывает `.env.benchmark-models`.

---

## 3. Поднять инфраструктуру и подготовить данные

```bash
cd Submodules/voproshalych

# Поднять БД
docker compose -f docker-compose.benchmarks.yml up -d db db-migrate

# Установить зависимости
uv sync

# Загрузить дамп БД
uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump

# Сгенерировать эмбеддинги
uv run python benchmarks/generate_embeddings.py --chunks
uv run python benchmarks/generate_embeddings.py --check-coverage
```

Если нужен «чистый» прогон от пустой БД:

```bash
cd Submodules/voproshalych

# Очистить таблицы
uv run python benchmarks/load_database_dump.py --drop-tables-only

# Загрузить дамп
uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump

# Сгенерировать эмбеддинги
uv run python benchmarks/generate_embeddings.py --chunks
uv run python benchmarks/generate_embeddings.py --check-coverage
```

**Важно:** После генерации датасета (шаг 4) запомните имя файла — оно нужно для всех последующих команд.

---

## 4. Генерация synthetic датасета на всех чанках

Полный датасет по всем чанкам (в текущем состоянии БД ориентировано: 202):

```bash
cd Submodules/voproshalych
uv run python benchmarks/generate_dataset.py --mode synthetic --max-questions 202
```

Опционально экспорт в аннотацию:

```bash
cd Submodules/voproshalych
uv run python benchmarks/generate_dataset.py --mode export-annotation --output benchmarks/data/dataset_for_annotation_full.json
```

---

## 5. Полный baseline прогон (single-model)

Рекомендуется сначала зафиксировать baseline без model-sweep, чтобы иметь точку сравнения.
Модели берутся из `.env.benchmark-models`:

- `GENERATION_MODEL` — модель для генерации ответов
- `JUDGE_MODEL` — production judge модель
- `BENCHMARKS_JUDGE_MODEL` — benchmark judge модель

```bash
cd Submodules/voproshalych

# Сначала получите имя последнего датасета:
ls benchmarks/data/dataset_synthetic_*.json | tail -1

# Затем запустите бенчмарк (замените <TIMESTAMP> на реальное):
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all \
  --mode synthetic \
  --limit 202 \
  --dataset benchmarks/data/dataset_synthetic_<TIMESTAMP>.json \
  --consistency-runs 2 \
  --judge-eval-mode reasoned \
  --analyze-utilization \
  --utilization-questions-source real \
  --utilization-question-limit 100000 \
  --utilization-top-k 10 \
  --analyze-topics \
  --topics-question-limit 100000 \
  --topics-count 20 \
  --topics-top-k 5 \
  --analyze-domain \
  --domain-limit 100000
```

Пояснения к full-limit для real-user аналитики:

- `100000` выбрано как верхний «захват всего», чтобы покрыть таблицу 5000+ без ручного пересчёта;
- скрипты сами ограничиваются фактическим числом строк.

---

## 6. Model Lab: сравнение всех видов моделей

Ниже показан workflow для списка до 10 моделей в каждой категории.

### 6.1 Подготовить списки моделей

Пример (замените на свои):

- Generation models (`GEN_MODELS`): 10 шт.
- Benchmark judge models (`BM_JUDGE_MODELS`): 10 шт.
- Production judge models (`PR_JUDGE_MODELS`): 10 шт.

### 6.2 Полный cartesian sweep (осторожно с объёмом)

> **Важно:** Для запуска multi-model sweep нужно:
> 1. Заполнить списки моделей в `.env.benchmark-models` (например `MISTRAL_GEN_MODELS=model1,model2,model3`)
> 2. Использовать флаги `--generation-model-source`, `--production-judge-model-source`, `--benchmark-judge-model-source`
>
> Примеры см. в секции 6.4 ниже.

```bash
cd Submodules/voproshalych

# Сначала получите имя последнего датасета:
# ls benchmarks/data/dataset_synthetic_*.json | tail -1

# Затем запустите бенчмарк (замените <TIMESTAMP> на реальное):
uv run python benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic --limit 202 --dataset benchmarks/data/dataset_synthetic_<TIMESTAMP>.json --consistency-runs 2 --judge-eval-mode reasoned --generation-models "m1,m2,m3,m4,m5,m6,m7,m8,m9,m10" --judge-models "j1,j2,j3,j4,j5,j6,j7,j8,j9,j10" --production-judge-models "p1,p2,p3,p4,p5,p6,p7,p8,p9,p10" --analyze-utilization --utilization-questions-source real --utilization-question-limit 100000 --utilization-top-k 10 --analyze-topics --topics-question-limit 100000 --topics-count 20 --topics-top-k 5 --analyze-domain --domain-limit 100000
```

Это создаёт комбинации `10 x 10 x 10 = 1000` запусков в одном прогоне (`model_runs[]`). По времени и стоимости это очень тяжело.

### 6.3 Практичный рекомендованный режим (поэтапно)

1) **Generation + Benchmark Judge sweep** (фиксируем production judge):

```bash
cd Submodules/voproshalych

# Получите имя последнего датасета и вставьте ниже:
uv run python benchmarks/run_comprehensive_benchmark.py --tier 2 --mode synthetic --limit 202 --dataset benchmarks/data/dataset_synthetic_<TIMESTAMP>.json --consistency-runs 2 --judge-eval-mode reasoned --generation-models "m1,m2,m3,m4,m5,m6,m7,m8,m9,m10" --judge-models "j1,j2,j3,j4,j5,j6,j7,j8,j9,j10" --production-judge-models "p_base"
```

2) **Production Judge sweep** (фиксируем generation и benchmark judge):

```bash
cd Submodules/voproshalych

# Получите имя последнего датасета и вставьте ниже:
uv run python benchmarks/run_comprehensive_benchmark.py --tier judge_pipeline --mode synthetic --limit 202 --dataset benchmarks/data/dataset_synthetic_<TIMESTAMP>.json --generation-models "m_base" --judge-models "j_base" --production-judge-models "p1,p2,p3,p4,p5,p6,p7,p8,p9,p10"
```

Пример с выбором конкретных токенов и model-source:

```bash
cd Submodules/voproshalych

# Получите имя последнего датасета и вставьте ниже:
uv run python benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic --limit 202 --dataset benchmarks/data/dataset_synthetic_<TIMESTAMP>.json --generation-api-key-var PROVIDER_DEEPSEEK_API_KEY --generation-model-source deepseek --production-judge-api-key-var PROVIDER_MISTRAL_API_KEY --production-judge-model-source mistral --benchmark-judge-api-key-var PROVIDER_OPENROUTER_API_KEY --judge-model-source openrouter
```

### 6.4 Model-source подход (рекомендуется для multi-model)

Для использования нескольких моделей заполните списки в `.env.benchmark-models`:

```bash
# Generation модели
MISTRAL_GEN_MODELS=open-mistral-nemo,mistral-small-latest,mistral-medium-latest
OPENROUTER_GEN_MODELS=openai/gpt-4o-mini,openai/gpt-4.1-mini

# Production Judge модели
MISTRAL_JUDGE_MODELS=mistral-small-latest,mistral-medium-latest

# Benchmark Judge модели
ALIBABA_BM_JUDGE_MODELS=qwen-turbo
```

Затем запускайте с флагами `--*-model-source`:

```bash
cd Submodules/voproshalych

# Generation sweep (фиксированные judge модели)
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier 2 \
  --mode synthetic \
  --limit 202 \
  --dataset benchmarks/data/dataset_synthetic_<TIMESTAMP>.json \
  --generation-model-source mistral \
  --production-judge-model-source mistral \
  --benchmark-judge-model-source alibaba
```

Пример с ручным списком моделей (старый способ):

```bash
cd Submodules/voproshalych

# Получите имя последнего датасета и вставьте ниже:
uv run python benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic --limit 202 --dataset benchmarks/data/dataset_synthetic_<TIMESTAMP>.json --generation-api-key-var PROVIDER_DEEPSEEK_API_KEY --generation-api-url-var PROVIDER_DEEPSEEK_API_URL --production-judge-api-key-var PROVIDER_MISTRAL_API_KEY --benchmark-judge-base-url-var PROVIDER_OPENROUTER_BASE_URL --generation-models "deepseek-chat,mistral-small-latest" --judge-models "openai/gpt-4o-mini,openai/gpt-4.1-mini" --production-judge-models "mistral-small-latest,mistral-medium-latest"
```

Такой подход почти всегда лучше по времени/стоимости и интерпретируемости.

---

## 7. Отдельные full-run источниковые аналитики

Если нужно прогнать source analytics отдельно, без tier-бенчмарков:

```bash
cd Submodules/voproshalych

uv run python benchmarks/analyze_chunk_utilization.py --questions-source real --question-limit 100000 --top-k 10 --output benchmarks/reports/utilization_full.json
uv run python benchmarks/analyze_topic_coverage.py --question-limit 100000 --n-topics 20 --top-k 5 --output benchmarks/reports/topic_coverage_full.json
uv run python benchmarks/analyze_real_users_domain.py --limit 100000 --output benchmarks/reports/real_users_domain_analysis_full.json
```

---

## 8. Просмотр результатов и сравнение в дашборде

```bash
cd Submodules/voproshalych
uv run python benchmarks/run_dashboard.py
```

Открой `http://localhost:7860` и проверьте вкладки:

- `Run Details`
- `LLM Comparison` (generation)
- `Judge Comparison` (benchmark judge)
- `Prod Judge Comparison` (production judge)
- `Chunk Utilization`
- `Topic Coverage`
- `Domain Insights`

Ключевые production-judge метрики для выбора модели:

| Метрика | Тип | Назначение |
|--------|-----|------------|
| `accuracy` | float | Общая точность |
| `precision` | float | Precision для класса "Yes" |
| `recall` | float | Recall для класса "Yes" |
| `f1_score` | float | Баланс precision/recall |
| `avg_latency_ms` | float | Средняя задержка |

---

## 9. Где лежат итоговые файлы

- `benchmarks/reports/benchmark_runs.json` — реестр запусков
- `benchmarks/reports/rag_benchmark_<timestamp>.json` — подробный отчёт
- `benchmarks/reports/rag_benchmark_<timestamp>.md` — markdown отчёт
- `benchmarks/reports/*_full.json` — full source analytics

---

## 10. Завершение

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml down
```

---

## 11. Короткий checklist «финальный сбор цифр»

1. Поднять БД, загрузить дамп, сгенерировать эмбеддинги.
2. Сгенерировать synthetic dataset на 202 чанка.
3. Прогнать baseline `--tier all` + full source analytics.
4. Прогнать model-sweep (этапно или cartesian).
5. Проверить сравнения на вкладках `LLM/Judge/Prod Judge Comparison`.
6. Зафиксировать лучшие модели по метрикам и latency.
