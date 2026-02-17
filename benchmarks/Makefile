ROOT_DIR := ..

.PHONY: load-dump drop-tables generate-embeddings generate-dataset run-benchmarks run-dashboard install

install:
	uv sync

load-dump:
	uv run python $(ROOT_DIR)/benchmarks/load_database_dump.py --dump $(ROOT_DIR)/benchmarks/data/dump/virtassist_backup_20260213.dump

drop-tables:
	uv run python $(ROOT_DIR)/benchmarks/load_database_dump.py --drop-tables-only

generate-embeddings:
	docker compose exec benchmarks uv run python benchmarks/generate_embeddings.py --chunks

generate-dataset:
	docker compose exec benchmarks uv run python benchmarks/generate_dataset.py --max-questions 500

run-benchmarks:
	docker compose exec -e BENCHMARK_GIT_BRANCH=$$(git rev-parse --abbrev-ref HEAD) \
		-e BENCHMARK_GIT_COMMIT_HASH=$$(git rev-parse --short HEAD) \
		-e BENCHMARK_RUN_AUTHOR=$$(git config user.name) \
		benchmarks uv run python benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic

run-dashboard:
	docker compose run --rm -p 7860:7860 benchmarks uv run python benchmarks/run_dashboard.py
