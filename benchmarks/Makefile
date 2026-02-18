ROOT_DIR := ..
COMPOSE_FILE ?= docker-compose.yml

.PHONY: install load-dump drop-tables generate-embeddings generate-dataset run-benchmarks run-dashboard up down ps logs
.PHONY: install-local load-dump-local drop-tables-local generate-embeddings-local generate-dataset-local run-benchmarks-local run-dashboard-local

# Локальный режим (без контейнеризации)
install-local:
	uv sync

load-dump-local:
	uv run python $(ROOT_DIR)/benchmarks/load_database_dump.py --dump $(ROOT_DIR)/benchmarks/data/dump/virtassist_backup_20260213.dump

drop-tables-local:
	uv run python $(ROOT_DIR)/benchmarks/load_database_dump.py --drop-tables-only

generate-embeddings-local:
	uv run python $(ROOT_DIR)/benchmarks/generate_embeddings.py --chunks
	uv run python $(ROOT_DIR)/benchmarks/generate_embeddings.py --check-coverage

generate-dataset-local:
	uv run python $(ROOT_DIR)/benchmarks/generate_dataset.py --max-questions 20

run-benchmarks-local:
	uv run python $(ROOT_DIR)/benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic --limit 10

run-dashboard-local:
	uv run python $(ROOT_DIR)/benchmarks/run_dashboard.py

# Docker режим (используется COMPOSE_FILE, по умолчанию docker-compose.benchmarks.yml)
up:
	docker compose -f $(COMPOSE_FILE) up -d --build

down:
	docker compose -f $(COMPOSE_FILE) down

load-dump:
	docker compose -f $(COMPOSE_FILE) exec -T benchmarks uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump

drop-tables:
	docker compose -f $(COMPOSE_FILE) exec -T benchmarks uv run python benchmarks/load_database_dump.py --drop-tables-only

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
	@echo ""
	@echo "Локальный режим (без контейнеризации):"
	@echo "  install-local              - Установить зависимости через uv sync"
	@echo "  load-dump-local            - Загрузить дамп БД"
	@echo "  drop-tables-local          - Удалить таблицы БД"
	@echo "  generate-embeddings-local  - Сгенерировать эмбеддинги"
	@echo "  generate-dataset-local     - Сгенерировать датасет (20 вопросов)"
	@echo "  run-benchmarks-local       - Запустить бенчмарки (synthetic mode, 10 вопросов)"
	@echo "  run-dashboard-local        - Запустить дашборд локально"
	@echo ""
	@echo "Docker режим (используется COMPOSE_FILE):"
	@echo "  up                         - Поднять все сервисы (использует COMPOSE_FILE)"
	@echo "  down                       - Остановить все сервисы (использует COMPOSE_FILE)"
	@echo "  load-dump                  - Загрузить дамп БД через Docker"
	@echo "  drop-tables                - Удалить таблицы БД через Docker"
	@echo "  generate-embeddings        - Сгенерировать эмбеддинги через Docker"
	@echo "  generate-dataset           - Сгенерировать датасет через Docker (500 вопросов)"
	@echo "  run-benchmarks             - Запустить бенчмарки через Docker"
	@echo "  run-dashboard              - Запустить дашборд через Docker"
	@echo "  ps                         - Показать статус сервисов"
	@echo "  logs                       - Показать логи сервиса benchmarks"
	@echo ""
	@echo "Переменные:"
	@echo "  COMPOSE_FILE                - Docker compose файл (по умолчанию: docker-compose.yml)"
	@echo ""
	@echo "Примеры:"
	@echo ""
	@echo "Локальный режим:"
	@echo "  make install-local"
	@echo "  make load-dump-local"
	@echo "  make generate-embeddings-local"
	@echo "  make generate-dataset-local"
	@echo "  make run-benchmarks-local"
	@echo "  make run-dashboard-local"
	@echo ""
	@echo "Docker режим:"
	@echo "  make COMPOSE_FILE=docker-compose.benchmarks.yml up"
	@echo "  make COMPOSE_FILE=docker-compose.benchmarks.yml load-dump"
	@echo "  make COMPOSE_FILE=docker-compose.benchmarks.yml generate-embeddings"
	@echo "  make COMPOSE_FILE=docker-compose.benchmarks.yml generate-dataset"
	@echo "  make COMPOSE_FILE=docker-compose.benchmarks.yml run-benchmarks"
	@echo "  make COMPOSE_FILE=docker-compose.benchmarks.yml run-dashboard"
	@echo ""
	@echo "Быстрый старт для Docker:"
	@echo "  cd Submodules/voproshalych/benchmarks"
	@echo "  make COMPOSE_FILE=../docker-compose.benchmarks.yml up"
	@echo "  make COMPOSE_FILE=../docker-compose.benchmarks.yml load-dump"
	@echo "  make COMPOSE_FILE=../docker-compose.benchmarks.yml generate-embeddings"
	@echo "  make COMPOSE_FILE=../docker-compose.benchmarks.yml generate-dataset"
	@echo "  make COMPOSE_FILE=../docker-compose.benchmarks.yml run-benchmarks"
	@echo "  make COMPOSE_FILE=../docker-compose.benchmarks.yml run-dashboard"
