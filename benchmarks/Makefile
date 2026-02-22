ROOT_DIR := .. 

.PHONY: install-local load-dump-local drop-tables-local generate-embeddings-local generate-dataset-local generate-dataset-real-local generate-dataset-score5-local run-benchmarks-local run-dashboard-local run-local help up down

# Поднять/опустить БД
up:
	docker compose -f docker-compose.benchmarks.yml up -d db db-migrate

down:
	docker compose -f docker-compose.benchmarks.yml down

# Локальный режим
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
	uv run python $(ROOT_DIR)/benchmarks/generate_dataset.py --mode synthetic --max-questions 20

generate-dataset-real-local:
	uv run python $(ROOT_DIR)/benchmarks/generate_dataset.py --mode from-real-questions --max-questions 20

generate-dataset-score5-local:
	uv run python $(ROOT_DIR)/benchmarks/generate_dataset.py --mode from-real-questions-score-5 --max-questions 20

run-benchmarks-local:
	@GIT_BRANCH=$$(git rev-parse --abbrev-ref HEAD) && \
	GIT_COMMIT=$$(git rev-parse --short HEAD) && \
	GIT_AUTHOR=$$(git config user.name) && \
	echo "Using git branch: $$GIT_BRANCH, commit: $$GIT_COMMIT, author: $$GIT_AUTHOR" && \
	BENCHMARK_GIT_BRANCH=$$GIT_BRANCH BENCHMARK_GIT_COMMIT_HASH=$$GIT_COMMIT BENCHMARK_RUN_AUTHOR=$$GIT_AUTHOR uv run python $(ROOT_DIR)/benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic

run-dashboard-local:
	uv run python $(ROOT_DIR)/benchmarks/run_dashboard.py

run-local:
	@test -n "$(SCRIPT)" || (echo "Usage: make run-local SCRIPT=benchmarks/script.py ARGS=\"...\"" && exit 1)
	uv run python $(ROOT_DIR)/$(SCRIPT) $(ARGS)

.PHONY: help
help:
	@echo "Доступные команды:"
	@echo ""
	@echo "Поднять/опустить БД:"
	@echo "  make up                         - Поднять БД и миграции"
	@echo "  make down                       - Остановить БД"
	@echo ""
	@echo "Локальный режим:"
	@echo "  make install-local              - Установить зависимости через uv sync"
	@echo "  make load-dump-local           - Загрузить дамп БД"
	@echo "  make drop-tables-local         - Удалить таблицы БД"
	@echo "  make generate-embeddings-local - Сгенерировать эмбеддинги"
	@echo "  make generate-dataset-local    - Сгенерировать synthetic датасет (20 вопросов)"
	@echo "  make generate-dataset-real-local - Сгенерировать датасет из реальных вопросов (20 вопросов)"
	@echo "  make generate-dataset-score5-local - Сгенерировать датасет из вопросов с оценкой 5"
	@echo "  make run-benchmarks-local      - Запустить бенчмарки (synthetic mode)"
	@echo "  make run-dashboard-local      - Запустить дашборд"
	@echo "  make run-local                 - Запустить произвольный Python script"
	@echo ""
	@echo "Примеры:"
	@echo ""
	@echo "Полный цикл:"
	@echo "  make up"
	@echo "  make install-local"
	@echo "  make load-dump-local"
	@echo "  make generate-embeddings-local"
	@echo "  make generate-dataset-local"
	@echo "  make run-benchmarks-local"
	@echo "  make run-dashboard-local"
	@echo ""
	@echo "Запуск произвольного скрипта:"
	@echo "  make run-local SCRIPT=benchmarks/generate_dataset.py ARGS=\"--mode synthetic --max-questions 10\""
