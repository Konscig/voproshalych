# Руководство: сравнение LLM моделей в benchmarks

Документ описывает, как запускать сравнение моделей для:

1. **Generation LLM** (Tier 2, Tier 3),
2. **Benchmark Judge LLM** (Tier Judge, LLM-as-a-Judge для метрик),
3. **Production Judge LLM** (Tier Judge Pipeline).

Важно: для production judge сравнения нужен dataset
`benchmarks/data/dataset_judge_pipeline_*.json`.

---

## 1. Где хранить API ключи

Рекомендуемый подход:

- базовые переменные хранить в `.env.docker`;
- отдельные переменные для model-lab хранить в `.env.benchmark-models`
  (файл читается `run_comprehensive_benchmark.py`, не коммитится).

Шаблон: `.env.benchmark-models.example` (в корне `Submodules/voproshalych`).

Файл `.env.benchmark-models.example` уже содержит:

- production + compare секции для generation / production judge /
  benchmark judge;
- централизованные ключи по платформам: `mistral.ai`, `openrouter.ai`,
  `platform.deepseek.com`, `alibabacloud.com`;
- active aliases, используемые кодом по умолчанию.

Ключевая идея: храните **все** токены в одном файле и выбирайте нужный
на запуск через флаги выбора имени env-переменной.

Пример `.env.benchmark-models` (фрагмент):

```dotenv
# Provider vault
PROVIDER_MISTRAL_API_KEY=...
PROVIDER_OPENROUTER_API_KEY=...
PROVIDER_DEEPSEEK_API_KEY=...
PROVIDER_ALIBABA_API_KEY=...

# Role aliases
GENERATION_API_KEY=${PROVIDER_MISTRAL_API_KEY}
JUDGE_API=${PROVIDER_MISTRAL_API_KEY}
BENCHMARKS_JUDGE_API_KEY=${PROVIDER_OPENROUTER_API_KEY}
```

Пояснение по ключам и моделям:

- если один API-ключ покрывает несколько моделей провайдера, достаточно
  одного ключа и CSV списка моделей (`--generation-models`, `--judge-models`,
  `--production-judge-models`);
- если для разных моделей нужны разные ключи/провайдеры, запускайте батчи
  отдельно (по 1-2 провайдера за прогон).

---

## 2. CLI для сравнения моделей

Есть два способа указать модели для сравнения:

### Способ А: model-source (рекомендуемый)

Укажите провайдер — скрипт автоматически загрузит модели из env-переменных:

- `--generation-model-source` — mistral / openrouter / deepseek / alibaba
- `--judge-model-source` — mistral / openrouter / deepseek / alibaba  
- `--production-judge-model-source` — mistral / openrouter / deepseek / alibaba

Модели берутся из переменных:
- `*_GEN_MODELS` (generation)
- `*_BM_JUDGE_MODELS` (benchmark judge)
- `*_JUDGE_MODELS` (production judge)

### Способ Б: ручной CSV (старый)

- `--generation-models` — список generation моделей через запятую;
- `--judge-models` — список benchmark judge моделей;
- `--production-judge-models` — список production judge моделей.

Выбор токенов (из единого файла ключей):

- `--generation-api-key-var`
- `--production-judge-api-key-var`
- `--benchmark-judge-api-key-var`
- `--generation-api-url-var`
- `--benchmark-judge-base-url-var`

### Пример (model-source)

```bash
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all \
  --mode synthetic \
  --limit 10 \
  --judge-eval-mode reasoned \
  --generation-api-key-var PROVIDER_DEEPSEEK_API_KEY \
  --generation-model-source deepseek \
  --production-judge-api-key-var PROVIDER_MISTRAL_API_KEY \
  --production-judge-model-source mistral \
  --benchmark-judge-api-key-var PROVIDER_OPENROUTER_API_KEY \
  --judge-model-source openrouter
```

### Пример (CSV, старый способ)

```bash
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all \
  --mode synthetic \
  --limit 10 \
  --judge-eval-mode reasoned \
  --generation-api-key-var PROVIDER_DEEPSEEK_API_KEY \
  --generation-api-url-var PROVIDER_DEEPSEEK_API_URL \
  --production-judge-api-key-var PROVIDER_MISTRAL_API_KEY \
  --benchmark-judge-api-key-var PROVIDER_OPENROUTER_API_KEY \
  --benchmark-judge-base-url-var PROVIDER_OPENROUTER_BASE_URL \
  --generation-models "mistral-small-latest,mistral-medium-latest" \
  --judge-models "openai/gpt-4o-mini,openai/gpt-4.1-mini" \
  --production-judge-models "mistral-small-latest,mistral-medium-latest"
```

### Что будет сохранено

В `benchmarks/reports/benchmark_runs.json` появится поле `model_runs[]` с
комбинациями моделей и метриками запуска.

Каждый элемент содержит:

- `generation_model`
- `judge_model` (benchmark judge)
- `production_judge_model`
- `judge_eval_mode`
- `metrics` (все tier-метрики по комбинации)

---

## 3. Вкладки дашборда для model-lab

После запуска с multi-model флагами в dashboard доступны вкладки:

1. **LLM Comparison** — сравнение generation моделей по Tier 2/3 метрикам.
2. **Judge Comparison** — сравнение benchmark judge моделей (`consistency`,
   `error_rate`, `latency`).
3. **Prod Judge Comparison** — сравнение production judge моделей по метрикам:

   - `accuracy`
   - `precision`
   - `recall`
   - `f1_score`
   - `avg_latency_ms`

---

## 4. Рекомендуемый workflow выбора моделей

1. Зафиксировать synthetic/manual датасет.
2. Запустить небольшой sweep (`limit=10-30`) для отсева.
3. Запустить полный sweep (`limit=200+`) на shortlist моделей.
4. Сравнить результаты в трёх вкладках model-lab.
5. Выбрать:
   - лучшую generation модель,
   - лучшую benchmark judge модель,
   - лучшую production judge модель.

---

## 5. Ограничения и интерпретация

- Сравнение корректно только при одинаковом датасете и одинаковых флагах.
- Для fair-сравнения фиксируйте `judge_eval_mode` и `consistency_runs`.
- Если у модели нет доступа по API ключу, комбинация не должна входить в
  финальный отбор.
