# Smoke-сценарий для benchmarks

Полный цикл проверки: synthetic, manual и real-users режимы.

## 0) Поднять контейнеры

Для запуска с бенчмарками:

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml up -d --build
```

Для запуска только основного приложения:

```bash
cd Submodules/voproshalych
docker compose up -d --build
```

Дождитесь здорового состояния всех сервисов:
```bash
docker compose -f docker-compose.benchmarks.yml ps
```

## 1) Подготовка БД

```bash
uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump
```

Если нужно только удалить таблицы без загрузки дампа:

```bash
uv run python benchmarks/load_database_dump.py --drop-tables-only
```

**Важно:** При загрузке дампа таблицы автоматически очищаются перед загрузкой.

## 2) Эмбеддинги

```bash
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --chunks
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --check-coverage
```

## 3) Synthetic dataset + benchmark

```bash
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py --max-questions 20
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic --limit 10
```

## 4) Manual dataset + benchmark

Создайте файл `benchmarks/data/manual_dataset_smoke.json` с 3-5 записями.

```bash
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode manual \
  --manual-dataset benchmarks/data/manual_dataset_smoke.json \
  --limit 5
```

## 5) Real users benchmark

```bash
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 50 --top-k 10
```

## 6) Проверка результатов

```bash
docker compose -f docker-compose.benchmarks.yml run --rm -p 7860:7860 benchmarks uv run python benchmarks/run_dashboard.py
```

Дашборд доступен по адресу: `http://localhost:7860`

Ожидаемо:
- в `benchmarks/reports/` появляются свежие `rag_benchmark_*.json/.md`;
- в `benchmark_runs` появляются новые записи с `dataset_type`;
- в дашборде видны метрики и графики по новым запускам.

## Остановка

```bash
docker compose -f docker-compose.benchmarks.yml down
```
