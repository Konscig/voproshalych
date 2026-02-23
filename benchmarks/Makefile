ROOT_DIR := ..
COMPOSE_FILE := ../docker-compose.benchmarks.yml

.PHONY: up down install-local load-dump-local drop-tables-local
.PHONY: generate-embeddings-local generate-dataset-local
.PHONY: generate-dataset-real-local generate-dataset-score5-local
.PHONY: run-benchmarks-local run-dashboard-local run-local help

# Сервисы инфраструктуры (только БД + миграции)
up:
	docker compose -f $(COMPOSE_FILE) up -d db db-migrate

down:
	docker compose -f $(COMPOSE_FILE) down

# Локальный режим (UV + локальные скрипты)
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
	uv run python $(ROOT_DIR)/benchmarks/generate_dataset.py --mode synthetic --max-questions 10

generate-dataset-real-local:
	uv run python $(ROOT_DIR)/benchmarks/generate_dataset.py --mode from-real-questions --max-questions 10

generate-dataset-score5-local:
	uv run python $(ROOT_DIR)/benchmarks/generate_dataset.py --mode from-real-questions-score-5 --max-questions 10

run-benchmarks-local:
	@GIT_BRANCH=$$(git -C $(ROOT_DIR) rev-parse --abbrev-ref HEAD) && \
	GIT_COMMIT=$$(git -C $(ROOT_DIR) rev-parse --short HEAD) && \
	GIT_AUTHOR=$$(git -C $(ROOT_DIR) config user.name) && \
	echo "Using git branch: $$GIT_BRANCH, commit: $$GIT_COMMIT, author: $$GIT_AUTHOR" && \
	BENCHMARK_GIT_BRANCH=$$GIT_BRANCH BENCHMARK_GIT_COMMIT_HASH=$$GIT_COMMIT BENCHMARK_RUN_AUTHOR=$$GIT_AUTHOR \
	uv run python $(ROOT_DIR)/benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic --limit 10 --analyze-utilization --analyze-topics --analyze-domain

run-dashboard-local:
	uv run python $(ROOT_DIR)/benchmarks/run_dashboard.py

run-local:
	@test -n "$(SCRIPT)" || (echo "Usage: make run-local SCRIPT=benchmarks/script.py ARGS=\"...\"" && exit 1)
	uv run python $(ROOT_DIR)/$(SCRIPT) $(ARGS)

help:
	@echo "Доступные команды benchmarks:"
	@echo ""
	@echo "Инфраструктура:"
	@echo "  make up                           - Поднять db и db-migrate"
	@echo "  make down                         - Остановить сервисы"
	@echo ""
	@echo "Локальные скрипты (UV):"
	@echo "  make install-local                - Установить зависимости"
	@echo "  make load-dump-local              - Загрузить дамп БД"
	@echo "  make drop-tables-local            - Очистить таблицы"
	@echo "  make generate-embeddings-local    - Эмбеддинги чанков + coverage"
	@echo "  make generate-dataset-local       - Synthetic dataset (10 вопросов)"
	@echo "  make generate-dataset-real-local  - Dataset из real вопросов (10)"
	@echo "  make generate-dataset-score5-local - Dataset score=5 (10)"
	@echo "  make run-benchmarks-local         - Комплексный тестовый запуск"
	@echo "  make run-dashboard-local          - Запустить дашборд"
	@echo "  make run-local SCRIPT=... ARGS=... - Запуск произвольного скрипта"
