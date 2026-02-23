ROOT_DIR := ..
COMPOSE_FILE := ../docker-compose.benchmarks.yml

.PHONY: up down install-local load-dump-local drop-tables-local
.PHONY: generate-embeddings-local generate-dataset-local
.PHONY: run-benchmarks-local run-dashboard-local run-local help

# Сервисы инфраструктуры (только БД + миграции)
up:
	docker compose -f $(COMPOSE_FILE) up -d db db-migrate

down:
	docker compose -f $(COMPOSE_FILE) down

# Локальный режим (UV + локальные скрипты)
install-local:
	cd $(ROOT_DIR) && uv sync

load-dump-local:
	cd $(ROOT_DIR) && uv run python benchmarks/load_database_dump.py --dump benchmarks/data/dump/virtassist_backup_20260213.dump

drop-tables-local:
	cd $(ROOT_DIR) && uv run python benchmarks/load_database_dump.py --drop-tables-only

generate-embeddings-local:
	cd $(ROOT_DIR) && uv run python benchmarks/generate_embeddings.py --chunks
	cd $(ROOT_DIR) && uv run python benchmarks/generate_embeddings.py --check-coverage

generate-dataset-local:
	cd $(ROOT_DIR) && uv run python benchmarks/generate_dataset.py --mode synthetic --max-questions 20

generate-dataset-real-local:
	cd $(ROOT_DIR) && uv run python benchmarks/generate_dataset.py --mode from-real-questions --max-questions 20

generate-dataset-score5-local:
	cd $(ROOT_DIR) && uv run python benchmarks/generate_dataset.py --mode from-real-questions-score-5 --max-questions 20

run-benchmarks-local:
	@GIT_BRANCH=$$(git -C $(ROOT_DIR) rev-parse --abbrev-ref HEAD) && \
	GIT_COMMIT=$$(git -C $(ROOT_DIR) rev-parse --short HEAD) && \
	GIT_AUTHOR=$$(git -C $(ROOT_DIR) config user.name) && \
	echo "Using git branch: $$GIT_BRANCH, commit: $$GIT_COMMIT, author: $$GIT_AUTHOR" && \
	cd $(ROOT_DIR) && BENCHMARK_GIT_BRANCH=$$GIT_BRANCH BENCHMARK_GIT_COMMIT_HASH=$$GIT_COMMIT BENCHMARK_RUN_AUTHOR=$$GIT_AUTHOR uv run python benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic --limit 10 --analyze-utilization --analyze-topics --analyze-domain

run-dashboard-local:
	cd $(ROOT_DIR) && uv run python benchmarks/run_dashboard.py

run-local:
	@test -n "$(SCRIPT)" || (echo "Usage: make run-local SCRIPT=benchmarks/script.py ARGS=\"...\"" && exit 1)
	cd $(ROOT_DIR) && uv run python $(SCRIPT) $(ARGS)

help:
	@echo "Доступные команды benchmarks:"
	@echo ""
	@echo "Поднять/опустить БД:"
	@echo "  make up                         - Поднять db и db-migrate"
	@echo "  make down                       - Остановить БД"
	@echo ""
	@echo "Локальные скрипты (UV):"
	@echo "  make install-local              - Установить зависимости"
	@echo "  make load-dump-local              - Загрузить дамп БД"
	@echo "  make drop-tables-local            - Очистить таблицы"
	@echo "  make generate-embeddings-local - Сгенерировать эмбеддинги"
	@echo "  make generate-dataset-local       - Synthetic dataset (20 вопросов)"
	@echo "  make generate-dataset-real-local - Реальные вопросы (20 вопросов)"
	@echo "  make generate-dataset-score5-local - Вопросы с оценкой 5 (20 вопросов)"
	@echo "  make run-benchmarks-local          - Комплексный тестовый запуск"
	@echo "  make run-dashboard-local          - Запустить дашборд"
	@echo "  make run-local SCRIPT=... ARGS=... - Запустить произвольный скрипт"
	@echo ""
	@echo "Примеры:"
	@echo ""
	@echo "Полный цикл:"
	@echo "  cd Submodules/voproshalych/benchmarks"
	@echo "  make up"
	@echo "  make install-local"
	@echo "  make load-dump-local"
	@echo "  make generate-embeddings-local"
	@echo "  make generate-dataset-local"
	@echo "  make run-benchmarks-local"
	@echo "  make run-dashboard-local"
	@echo ""
	@echo "Запуск произвольного скрипта:"
	@echo "  cd Submodules/voproshalych/benchmarks"
	@echo "  make run-local SCRIPT=benchmarks/generate_dataset.py ARGS=\"--mode synthetic --max-questions 10\""
