# Smoke-сценарий для benchmarks

Полный цикл проверки: synthetic, manual и real-users режимы.

## 0) Поднять контейнеры

```bash
cd Submodules/voproshalych
docker compose up -d --build
```

Дождитесь здорового состояния всех сервисов:
```bash
docker compose ps
```

## 1) Подготовка БД

```bash
python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump
```

Если нужно только удалить таблицы без загрузки дампа:

```bash
python benchmarks/load_database_dump.py --drop-tables-only
```

**Важно:** При загрузке дампа таблицы автоматически очищаются перед загрузкой.

## 2) Эмбеддинги

```bash
cd Submodules/voproshalych
docker run --rm \
  --network voproshalych_chatbot-conn \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  virtassist/qa:latest \
  python benchmarks/generate_embeddings.py --chunks

docker run --rm \
  --network voproshalych_chatbot-conn \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  virtassist/qa:latest \
  python benchmarks/generate_embeddings.py --check-coverage
```

## 3) Synthetic dataset + benchmark

```bash
cd Submodules/voproshalych
docker run --rm \
  --network voproshalych_chatbot-conn \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  virtassist/qa:latest \
  python benchmarks/generate_dataset.py --max-questions 500

docker run --rm \
  --network voproshalych_chatbot-conn \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  -e BENCHMARK_GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)" \
  -e BENCHMARK_GIT_COMMIT_HASH="$(git rev-parse --short HEAD)" \
  -e BENCHMARK_RUN_AUTHOR="$(git config user.name)" \
  virtassist/qa:latest \
  python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic --limit 10
```

## 4) Manual dataset + benchmark

Создайте файл `benchmarks/data/manual_dataset_smoke.json` с 3-5 записями.

```bash
cd Submodules/voproshalych
docker run --rm \
  --network voproshalych_chatbot-conn \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  -e BENCHMARK_GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)" \
  -e BENCHMARK_GIT_COMMIT_HASH="$(git rev-parse --short HEAD)" \
  -e BENCHMARK_RUN_AUTHOR="$(git config user.name)" \
  virtassist/qa:latest \
  python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode manual \
  --manual-dataset benchmarks/data/manual_dataset_smoke.json \
  --limit 5
```

## 5) Real users benchmark

```bash
cd Submodules/voproshalych
docker run --rm \
  --network voproshalych_chatbot-conn \
  -v "$PWD:/workspace" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace \
  -e BENCHMARK_GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD)" \
  -e BENCHMARK_GIT_COMMIT_HASH="$(git rev-parse --short HEAD)" \
  -e BENCHMARK_RUN_AUTHOR="$(git config user.name)" \
  virtassist/qa:latest \
  python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 50 --top-k 10
```

## 6) Проверка результатов

```bash
cd Submodules/voproshalych
docker run --rm \
  --network voproshalych_chatbot-conn \
  -v "$PWD:/workspace" \
  -w /workspace \
  -p 7860:7860 \
  virtassist/qa:latest \
  python benchmarks/run_dashboard.py
```

Дашборд доступен по адресу: `http://localhost:7860`

Ожидаемо:
- в `benchmarks/reports/` появляются свежие `rag_benchmark_*.json/.md`;
- в `benchmark_runs` появляются новые записи с `dataset_type`;
- в дашборде видны метрики и графики по новым запускам.
