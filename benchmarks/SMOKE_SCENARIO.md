# Smoke-сценарий для benchmarks

Короткая проверка полного цикла: synthetic, manual и real-users режимы.

## 0) Подготовка окружения

```bash
cd Submodules/voproshalych
uv sync
```

## 1) Подготовка БД

```bash
python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump
```

## 2) Эмбеддинги

```bash
python benchmarks/generate_embeddings.py --chunks
python benchmarks/generate_embeddings.py --check-coverage
```

## 3) Synthetic dataset + benchmark

```bash
python benchmarks/generate_dataset.py --max-questions 20
python benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic --limit 10
```

## 4) Manual dataset + benchmark

Создайте файл `benchmarks/data/manual_dataset_smoke.json` с 3-5 записями.

```bash
python benchmarks/run_comprehensive_benchmark.py --tier all --mode manual --manual-dataset benchmarks/data/manual_dataset_smoke.json --limit 5
```

## 5) Real users benchmark

```bash
python benchmarks/run_comprehensive_benchmark.py --mode real-users --real-score 5 --real-limit 50 --top-k 10
```

## 6) Проверка результатов

```bash
python benchmarks/run_dashboard.py
```

Ожидаемо:
- в `benchmarks/reports/` появляются свежие `rag_benchmark_*.json/.md`;
- в `benchmark_runs` появляются новые записи с `dataset_type`;
- в дашборде видны метрики и графики по новым запускам.
