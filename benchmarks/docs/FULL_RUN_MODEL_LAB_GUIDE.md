# Полный прогон и Model Lab: финальный сбор всех метрик

Практическое руководство для полного прогона benchmark-системы и сравнения
моделей (generation / benchmark judge / production judge) в одном цикле.

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
(`..._COMPARE_API_KEY_01..10`) и выбирать нужный токен на запуск через
`--generation-api-key-var`, `--production-judge-api-key-var`,
`--benchmark-judge-api-key-var`.

Важно:

- `.env.benchmark-models` не коммитится;
- раннер автоматически подхватывает `.env.benchmark-models`.

---

## 3. Поднять инфраструктуру и подготовить данные

```bash
cd Submodules/voproshalych/benchmarks
make up
make install-local
make load-dump-local
make generate-embeddings-local
```

Если нужен «чистый» прогон от пустой БД:

```bash
cd Submodules/voproshalych/benchmarks
make drop-tables-local
make load-dump-local
make generate-embeddings-local
```

---

## 4. Генерация synthetic датасета на всех чанках

Полный датасет по всем чанкам (в текущем состоянии БД ориентир: 202):

```bash
cd Submodules/voproshalych/benchmarks
make run-local SCRIPT=benchmarks/generate_dataset.py ARGS="--mode synthetic --max-questions 202"
```

Опционально экспорт в аннотацию:

```bash
cd Submodules/voproshalych/benchmarks
make run-local SCRIPT=benchmarks/generate_dataset.py ARGS="--mode export-annotation --output benchmarks/data/dataset_for_annotation_full.json"
```

---

## 5. Полный baseline прогон (single-model)

Рекомендуется сначала зафиксировать baseline без model-sweep, чтобы иметь
точку сравнения.

```bash
cd Submodules/voproshalych/benchmarks
make run-local SCRIPT=benchmarks/run_comprehensive_benchmark.py ARGS="--tier all --mode synthetic --limit 202 --consistency-runs 2 --judge-eval-mode reasoned --analyze-utilization --utilization-questions-source real --utilization-question-limit 100000 --utilization-top-k 10 --analyze-topics --topics-question-limit 100000 --topics-count 20 --topics-top-k 5 --analyze-domain --domain-limit 100000"
```

Пояснения к full-limit для real-user аналитики:

- `100000` выбрано как верхний «захват всего», чтобы покрыть таблицу 5000+
  без ручного пересчёта;
- скрипты сами ограничат фактическим числом строк.

---

## 6. Model Lab: сравнение всех видов моделей

Ниже показан workflow для списка до 10 моделей в каждой категории.

### 6.1 Подготовить списки моделей

Пример (замените на свои):

- Generation models (`GEN_MODELS`): 10 шт.
- Benchmark judge models (`BM_JUDGE_MODELS`): 10 шт.
- Production judge models (`PR_JUDGE_MODELS`): 10 шт.

### 6.2 Полный cartesian sweep (осторожно с объёмом)

```bash
cd Submodules/voproshalych/benchmarks
make run-local SCRIPT=benchmarks/run_comprehensive_benchmark.py ARGS="--tier all --mode synthetic --limit 202 --consistency-runs 2 --judge-eval-mode reasoned --generation-models \"m1,m2,m3,m4,m5,m6,m7,m8,m9,m10\" --judge-models \"j1,j2,j3,j4,j5,j6,j7,j8,j9,j10\" --production-judge-models \"p1,p2,p3,p4,p5,p6,p7,p8,p9,p10\" --analyze-utilization --utilization-questions-source real --utilization-question-limit 100000 --analyze-topics --topics-question-limit 100000 --topics-count 20 --analyze-domain --domain-limit 100000"
```

Это создаёт комбинации `10 x 10 x 10 = 1000` запусков в одном прогоне
(`model_runs[]`). По времени и стоимости это очень тяжело.

### 6.3 Практичный рекомендованный режим (поэтапно)

1) **Generation + Benchmark Judge sweep** (фиксируем production judge):

```bash
cd Submodules/voproshalych/benchmarks
make run-local SCRIPT=benchmarks/run_comprehensive_benchmark.py ARGS="--tier 2 --mode synthetic --limit 202 --consistency-runs 2 --judge-eval-mode reasoned --generation-models \"m1,m2,m3,m4,m5,m6,m7,m8,m9,m10\" --judge-models \"j1,j2,j3,j4,j5,j6,j7,j8,j9,j10\" --production-judge-models \"p_base\""
```

2) **Production Judge sweep** (фиксируем generation и benchmark judge):

```bash
cd Submodules/voproshalych/benchmarks
make run-local SCRIPT=benchmarks/run_comprehensive_benchmark.py ARGS="--tier judge_pipeline --mode synthetic --limit 202 --generation-models \"m_base\" --judge-models \"j_base\" --production-judge-models \"p1,p2,p3,p4,p5,p6,p7,p8,p9,p10\""

Пример с выбором конкретных токенов из единого файла:

```bash
cd Submodules/voproshalych/benchmarks
make run-local SCRIPT=benchmarks/run_comprehensive_benchmark.py ARGS="--tier all --mode synthetic --limit 202 --generation-api-key-var GENERATION_COMPARE_API_KEY_03 --generation-api-url-var GENERATION_COMPARE_API_URL_03 --production-judge-api-key-var PRODUCTION_JUDGE_COMPARE_API_KEY_02 --benchmark-judge-api-key-var BENCHMARK_JUDGE_COMPARE_API_KEY_07 --benchmark-judge-base-url-var BENCHMARK_JUDGE_COMPARE_BASE_URL_07 --generation-models \"m1,m2\" --judge-models \"j1,j2\" --production-judge-models \"p1,p2\""
```
```

Такой подход почти всегда лучше по времени/стоимости и интерпретируемости.

---

## 7. Отдельные full-run источниковые аналитики

Если нужно прогнать source analytics отдельно, без tier-бенчмарков:

```bash
cd Submodules/voproshalych/benchmarks
make run-local SCRIPT=benchmarks/analyze_chunk_utilization.py ARGS="--questions-source real --question-limit 100000 --top-k 10 --output benchmarks/reports/utilization_full.json"
make run-local SCRIPT=benchmarks/analyze_topic_coverage.py ARGS="--question-limit 100000 --n-topics 20 --top-k 5 --output benchmarks/reports/topic_coverage_full.json"
make run-local SCRIPT=benchmarks/analyze_real_users_domain.py ARGS="--limit 100000 --output benchmarks/reports/real_users_domain_analysis_full.json"
```

---

## 8. Просмотр результатов и сравнение в дашборде

```bash
cd Submodules/voproshalych/benchmarks
make run-dashboard-local
```

Открой `http://localhost:7860` и проверь вкладки:

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
cd Submodules/voproshalych/benchmarks
make down
```

---

## 11. Короткий checklist «финальный сбор цифр»

1. Поднять БД, загрузить дамп, сгенерировать эмбеддинги.
2. Сгенерировать synthetic dataset на 202 чанка.
3. Прогнать baseline `--tier all` + full source analytics.
4. Прогнать model-sweep (этапно или cartesian).
5. Проверить сравнения на вкладках `LLM/Judge/Prod Judge Comparison`.
6. Зафиксировать лучшие модели по метрикам и latency.
