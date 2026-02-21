# Руководство по очистке прогонов бенчмарков

## Где хранятся данные

### 1. Локальные файлы (хост-машина)

**Отчёты:** `benchmarks/reports/`
```
benchmarks/reports/rag_benchmark_*.json
benchmarks/reports/rag_benchmark_*.md
```

**Датасеты:** `benchmarks/data/`
```
benchmarks/data/dataset_*.json
benchmarks/data/manual_dataset_*.json
```

### 2. Docker Volumes

| Volume | Что хранит | Путь в контейнере |
|--------|-----------|-------------------|
| `benchmarks-reports` | Отчёты JSON/MD | `/usr/src/app/benchmarks/reports` |
| `benchmarks-data` | Датасеты | `/usr/src/app/benchmarks/data` |
| `db-data` | База данных (включая `benchmark_runs`) | — |
| `model-cache` | Кэш моделей эмбеддингов | `/root/.cache/huggingface` |

### 3. База данных (таблица benchmark_runs)

**Важно:** База данных называется `virtassist`, не `postgres`!

```sql
\c virtassist
SELECT * FROM benchmark_runs ORDER BY timestamp DESC;
```

---

## Очистка всех прогонов

### Вариант 1: Полная очистка (удалить ВСЁ)

```bash
cd Submodules/voproshalych

# 1. Удалить локальные отчёты
rm -f benchmarks/reports/rag_benchmark_*.json
rm -f benchmarks/reports/rag_benchmark_*.md

# 2. Удалить локальные датасеты (если нужно)
rm -f benchmarks/data/dataset_*.json
rm -f benchmarks/data/manual_dataset_*.json

# 3. Очистить таблицу benchmark_runs в БД
docker compose -f docker-compose.benchmarks.yml exec -T db psql -U postgres -d virtassist -c "DELETE FROM benchmark_runs;"

# 4. Удалить Docker volumes (опционально)
docker compose -f docker-compose.benchmarks.yml down -v

# Примечание: -v удалит ВСЕ volumes, включая db-data
```

### Вариант 2: Очистить только старые запуски (оставить последние N)

```bash
# Оставить только последние 5 запусков
docker compose -f docker-compose.benchmarks.yml exec -T db psql -U postgres -d virtassist -c "
DELETE FROM benchmark_runs 
WHERE id NOT IN (
    SELECT id FROM benchmark_runs 
    ORDER BY timestamp DESC 
    LIMIT 5
);"
```

### Вариант 3: Очистить отчёты, но оставить историю в БД

```bash
# Только файлы
rm -f benchmarks/reports/rag_benchmark_*.json
rm -f benchmarks/reports/rag_benchmark_*.md

# В Docker volumes
docker run --rm -v benchmarks-reports:/data alpine rm -f /data/rag_benchmark_*.json /data/rag_benchmark_*.md
```

### Вариант 4: Удалить только определённый запуск

```bash
# Найти ID запуска
docker compose -f docker-compose.benchmarks.yml exec -T db psql -U postgres -d virtassist -c "SELECT id, timestamp, git_branch FROM benchmark_runs ORDER BY timestamp DESC LIMIT 10;"

# Удалить по ID
docker compose -f docker-compose.benchmarks.yml exec -T db psql -U postgres -d virtassist -c "DELETE FROM benchmark_runs WHERE id = <ID>;"
```

---

## Очистка только для конкретного сценария

### Перед smoke-тестом

```bash
cd Submodules/voproshalych

# Очистить старые отчёты
rm -f benchmarks/reports/rag_benchmark_*.json benchmarks/reports/rag_benchmark_*.md

# Очистить БД от старых запусков
docker compose -f docker-compose.benchmarks.yml exec -T db psql -U postgres -d virtassist -c "DELETE FROM benchmark_runs;"

# Запустить бенчмарк заново
make COMPOSE_FILE=../docker-compose.benchmarks.yml run-benchmarks
```

---

## Проверка после очистки

```bash
# Проверить что отчёты удалены
ls benchmarks/reports/

# Проверить что таблица пуста
docker compose -f docker-compose.benchmarks.yml exec -T db psql -U postgres -d virtassist -c "SELECT COUNT(*) FROM benchmark_runs;"

# Должно быть: 0
```

---

## Рекомендации

1. **Перед важным прогоном** — очищайте старые запуски, чтобы видеть только актуальные метрики
2. **Для долгосрочного хранения** — экспортируйте нужные запуски в отдельную директорию перед очисткой
3. **Для сравнения веток** — не очищайте, а добавляйте новые запуски с разными `git_branch`

---

## Экспорт запуска перед очисткой

```bash
# Экспорт конкретного запуска в файл
docker compose -f docker-compose.benchmarks.yml exec -T db psql -U postgres -d virtassist -c "\COPY (SELECT * FROM benchmark_runs WHERE id = <ID>) TO '/tmp/run_<ID>.csv' WITH CSV HEADER"

# Копировать из контейнера
docker compose -f docker-compose.benchmarks.yml cp db:/tmp/run_<ID>.csv ./benchmark_run_<ID>.csv
```
