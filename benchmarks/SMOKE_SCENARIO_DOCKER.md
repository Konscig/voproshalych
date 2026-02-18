# Smoke-сценарий для benchmarks (Контейнеризация с Docker)

Полный цикл проверки: synthetic, manual и real-users режимы с использованием `docker-compose.benchmarks.yml`.

## Предварительные требования

- Docker и Docker Compose установлены
- `.env.docker` файл настроен с переменными окружения
- Доступен PostgreSQL в сети Docker

## 0) Поднять контейнеры

Для запуска полного стека с бенчмарками:

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml up -d --build
```

Для запуска только основного приложения (без бенчмарков):

```bash
cd Submodules/voproshalych
docker compose up -d --build
```

Дождитесь здорового состояния всех сервисов:

```bash
docker compose -f docker-compose.benchmarks.yml ps
```

Ожидайте пока все сервисы будут в статусе `healthy` или `running`.

## 1) Проверка .env.docker

Убедитесь что `.env.docker` настроен:

```bash
cd Submodules/voproshalych
cat .env.docker
```

Обязательные переменные:
- `MISTRAL_API`, `MISTRAL_MODEL`
- `BENCHMARKS_JUDGE_API_KEY` (или `JUDGE_API`)
- `EMBEDDING_MODEL_PATH` (путь к модели или используеться HF)

## 2) Подготовка БД

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml exec -T benchmarks uv run python benchmarks/load_database_dump.py \
  --dump benchmarks/data/dump/virtassist_backup_20260213.dump
```

Если нужно только удалить таблицы без загрузки дампа:

```bash
docker compose -f docker-compose.benchmarks.yml exec -T benchmarks uv run python benchmarks/load_database_dump.py \
  --drop-tables-only
```

**Важно:** При загрузке дампа таблицы автоматически очищаются перед загрузкой.

## 3) Генерация эмбеддингов

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --chunks

docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_embeddings.py --check-coverage
```

**Примечание:** Эмбеддинги будут кэшироваться в volume `benchmarks-cache`.

## 4) Synthetic dataset + benchmark

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/generate_dataset.py --max-questions 20

docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode synthetic --limit 10
```

## 5) Manual dataset + benchmark

Создайте файл `benchmarks/data/manual_dataset_smoke.json` с 3-5 записями на хосте:

```bash
cd Submodules/voproshalych
nano benchmarks/data/manual_dataset_smoke.json
```

```bash
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --tier all --mode manual \
  --manual-dataset benchmarks/data/manual_dataset_smoke.json \
  --limit 5
```

## 6) Real users benchmark

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml exec benchmarks uv run python benchmarks/run_comprehensive_benchmark.py \
  --mode real-users --real-score 5 --real-limit 50 --top-k 10
```

## 7) Проверка результатов через дашборд

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml run --rm -p 7860:7860 benchmarks uv run python benchmarks/run_dashboard.py
```

Дашборд доступен по адресу: `http://localhost:7860`

Ожидаемо:
- в volume `benchmarks-reports` появляются свежие `rag_benchmark_*.json/.md`;
- в PostgreSQL таблице `benchmark_runs` появляются новые записи с `dataset_type`;
- в дашборде видны метрики и графики по новым запускам.

## 8) Просмотр логов

Для просмотра логов сервиса benchmarks:

```bash
docker compose -f docker-compose.benchmarks.yml logs -f benchmarks
```

Для просмотра логов всех сервисов:

```bash
docker compose -f docker-compose.benchmarks.yml logs -f
```

## 9) Проверка volumes

Проверьте что данные сохраняются в volumes:

```bash
docker volume inspect benchmarks-cache benchmarks-reports benchmarks-data
```

## 10) Остановка контейнеров

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml down
```

Для остановки с удалением volumes:

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml down -v
```

## Быстрые команды через Makefile

```bash
cd Submodules/voproshalych/benchmarks

# Установка зависимостей (локально)
make install

# Управление стеком
make up-benchmarks    # Поднять стек с бенчмарками
make down-benchmarks   # Остановить стек с бенчмарками

# Бенчмарки
make generate-embeddings
make generate-dataset
make run-benchmarks

# Дашборд
make run-dashboard

# Статус и логи
make ps               # Показать статус сервисов
make logs             # Показать логи benchmarks
make help             # Справка
```

## Устранение проблем

### Проблема: "Контейнер не запускается"

Проверьте логи:

```bash
docker compose -f docker-compose.benchmarks.yml logs benchmarks
```

Проверьте зависимости:

```bash
docker compose -f docker-compose.benchmarks.yml ps
```

### Проблема: "Нет соединения с БД"

Убедитесь что сервис `db` healthy:

```bash
docker compose -f docker-compose.benchmarks.yml ps
```

Проверьте переменные окружения в `.env.docker`:

```bash
docker compose -f docker-compose.benchmarks.yml exec benchmarks env | grep POSTGRES
```

### Проблема: "Модели не кэшируются"

Проверьте volume `benchmarks-cache`:

```bash
docker volume ls | grep benchmarks
```

Кэш хранится в `benchmarks-cache:/root/.cache`.

### Проблема: "Нет отчётов"

Проверьте volume `benchmarks-reports`:

```bash
docker run --rm -v benchmarks-reports:/data alpine ls -lh /data
```

## Мониторинг ресурсов

Для мониторинга ресурсов контейнеров:

```bash
docker stats
```

Для мониторинга только сервиса benchmarks:

```bash
docker stats virtassist-benchmarks
```

## Пересборка образа

Если нужно пересобрать образ benchmarks:

```bash
cd Submodules/voproshalych
docker compose -f docker-compose.benchmarks.yml build --no-cache benchmarks
```

Для пересборки с очисткой всех кэшей:

```bash
docker compose -f docker-compose.benchmarks.yml build --no-cache --pull benchmarks
docker system prune -f
```

## Архитектура сети

```
┌─────────────────────────────────────────┐
│          chatbot-conn network          │
├─────────────────────────────────────────┤
│  db (5432)                          │
│  db-migrate                           │
│  qa                                    │
│  chatbot                              │
│  adminpanel (80)                      │
│  max                                  │
│  benchmarks (7860)                    │
└─────────────────────────────────────────┘
```

Все сервисы в одной сети `chatbot-conn` для взаимодействия.

## Поведение контейнера benchmarks

**CMD в Dockerfile:** `tail -f /dev/null`

**Почему это нужно:**
- Контейнер benchmarks предназначен для ручного выполнения команд через `docker compose exec`
- Без CMD контейнер бы завершался сразу после старта
- `tail -f /dev/null` поддерживает контейнер в состоянии `running` бесконечно

**Использование:**
- Для запуска команд: `docker compose exec benchmarks uv run python ...`
- Для запуска дашборда: `docker compose run --rm -p 7860:7860 benchmarks uv run python benchmarks/run_dashboard.py`

**Важно:** Контейнер не является сервисом-демоном, а интерактивной средой для выполнения задач.


## Политики перезапуска сервисов

| Сервис | Policy | Поведение |
|---------|---------|------------|
| db | unless-stopped | Автоматический перезапуск |
| qa | unless-stopped | Автоматический перезапуск |
| chatbot | unless-stopped | Автоматический перезапуск |
| adminpanel | unless-stopped | Автоматический перезапуск |
| max | unless-stopped | Автоматический перезапуск |
| **benchmarks** | **unless-stopped** | **Контейнер остаётся running для exec команд** |

**Важно:** Сервис `benchmarks` имеет `restart: unless-stopped`, что позволяет:
- Контейнеру оставаться в состоянии `running` после запуска
- Выполнять команды через `docker compose exec`
- Запускать дашборд через `make run-dashboard`

