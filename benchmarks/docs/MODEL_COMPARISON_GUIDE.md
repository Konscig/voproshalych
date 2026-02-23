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

Пример `.env.benchmark-models`:

```dotenv
# LLM для генерации
MISTRAL_API=...
MISTRAL_MODEL=mistral-small-latest

# Production judge
JUDGE_API=...
JUDGE_MODEL=mistral-small-latest

# Benchmark judge
BENCHMARKS_JUDGE_API_KEY=...
BENCHMARKS_JUDGE_MODEL=openai/gpt-4o-mini
BENCHMARKS_JUDGE_BASE_URL=https://openrouter.ai/api/v1
```

---

## 2. CLI для сравнения моделей

Используйте `run_comprehensive_benchmark.py` с CSV списками:

- `--generation-models` — список generation моделей;
- `--judge-models` — список benchmark judge моделей;
- `--production-judge-models` — список production judge моделей.

### Пример (тестовый)

```bash
uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all \
  --mode synthetic \
  --limit 10 \
  --judge-eval-mode reasoned \
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
