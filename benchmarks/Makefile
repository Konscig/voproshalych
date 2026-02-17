ROOT_DIR := ..

.PHONY: load-dump generate-embeddings generate-dataset run-benchmarks run-dashboard

load-dump:
	python $(ROOT_DIR)/benchmarks/load_database_dump.py --dump $(ROOT_DIR)/benchmarks/data/dump/virtassist_backup_20260213.dump

generate-embeddings:
	docker run --rm \
	  --network voproshalych_chatbot-conn \
	  -v "$(PWD):/workspace" \
	  -v "$$HOME/.cache/huggingface:/root/.cache/huggingface" \
	  -w /workspace \
	  virtassist/qa:latest \
	  python $(ROOT_DIR)/benchmarks/generate_embeddings.py --chunks

generate-dataset:
	docker run --rm \
	  --network voproshalych_chatbot-conn \
	  -v "$(PWD):/workspace" \
	  -v "$$HOME/.cache/huggingface:/root/.cache/huggingface" \
	  -w /workspace \
	  virtassist/qa:latest \
	  python $(ROOT_DIR)/benchmarks/generate_dataset.py --max-questions 500

run-benchmarks:
	docker run --rm \
	  --network voproshalych_chatbot-conn \
	  -v "$(PWD):/workspace" \
	  -v "$$HOME/.cache/huggingface:/root/.cache/huggingface" \
	  -w /workspace \
	  -e BENCHMARK_GIT_BRANCH="$$(git rev-parse --abbrev-ref HEAD)" \
	  -e BENCHMARK_GIT_COMMIT_HASH="$$(git rev-parse --short HEAD)" \
	  -e BENCHMARK_RUN_AUTHOR="$$(git config user.name)" \
	  virtassist/qa:latest \
	  python $(ROOT_DIR)/benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic

run-dashboard:
	docker run --rm \
	  --network voproshalych_chatbot-conn \
	  -v "$(PWD):/workspace" \
	  -w /workspace \
	  -p 7860:7860 \
	  virtassist/qa:latest \
	  python $(ROOT_DIR)/benchmarks/run_dashboard.py
