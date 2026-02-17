ROOT_DIR := ..
COMPOSE_FILE ?= docker-compose.yml

.PHONY: load-dump drop-tables generate-embeddings generate-dataset run-benchmarks run-dashboard install up down

install:
	uv sync

up:
	docker compose -f $(COMPOSE_FILE) up -d --build

down:
	docker compose -f $(COMPOSE_FILE) down

up-benchmarks:
	docker compose -f docker-compose.benchmarks.yml up -d --build

down-benchmarks:
	docker compose -f docker-compose.benchmarks.yml down

load-dump:
	uv run python $(ROOT_DIR)/benchmarks/load_database_dump.py --dump $(ROOT_DIR)/benchmarks/data/dump/virtassist_backup_20260213.dump

drop-tables:
	uv run python $(ROOT_DIR)/benchmarks/load_database_dump.py --drop-tables-only

generate-embeddings:
	docker compose -f $(COMPOSE_FILE) exec benchmarks uv run python benchmarks/generate_embeddings.py --chunks

generate-dataset:
	docker compose -f $(COMPOSE_FILE) exec benchmarks uv run python benchmarks/generate_dataset.py --max-questions 500

run-benchmarks:
	docker compose -f $(COMPOSE_FILE) exec -e BENCHMARK_GIT_BRANCH=$$(git rev-parse --abbrev-ref HEAD) \
		-e BENCHMARK_GIT_COMMIT_HASH=$$(git rev-parse --short HEAD) \
		-e BENCHMARK_RUN_AUTHOR=$$(git config user.name) \
		benchmarks uv run python benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic

run-dashboard:
	docker compose -f $(COMPOSE_FILE) run --rm -p 7860:7860 benchmarks uv run python benchmarks/run_dashboard.py

ps:
	docker compose -f $(COMPOSE_FILE) ps

logs:
	docker compose -f $(COMPOSE_FILE) logs -f benchmarks

.PHONY: help
help:
	@echo "Доступные команды:"
	@echo "  install              - Установить зависимости через uv sync"
	@echo "  up                  - Поднять все сервисы (использует COMPOSE_FILE)"
	@echo "  down                - Остановить все сервисы (использует COMPOSE_FILE)"
	@echo "  up-benchmarks       - Поднять все сервисы с бенчмарками (docker-compose.benchmarks.yml)"
	@echo "  down-benchmarks     - Остановить все сервисы с бенчмарками"
	@echo "  load-dump           - Загрузить дамп БД"
	@echo "  drop-tables         - Удалить таблицы БД"
	@echo "  generate-embeddings - Сгенерировать эмбеддинги"
	@echo "  generate-dataset    - Сгенерировать датасет"
	@echo "  run-benchmarks      - Запустить бенчмарки"
	@echo "  run-dashboard       - Запустить дашборд"
	@echo "  ps                  - Показать статус сервисов"
	@echo "  logs                - Показать логи сервиса benchmarks"
	@echo ""
	@echo "Переменные:"
	@echo "  COMPOSE_FILE         - Docker compose файл (по умолчанию: docker-compose.yml)"
	@echo ""
	@echo "Примеры:"
	@echo "  make up-benchmarks              # Поднять с бенчмарками"
	@echo "  make COMPOSE_FILE=docker-compose.benchmarks.yml run-dashboard  # Запуск дашборда с указанием файла"
