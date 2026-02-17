PYTHON ?= python
ROOT_DIR := ..

.PHONY: load-dump generate-embeddings generate-dataset run-benchmarks run-dashboard

load-dump:
	$(PYTHON) $(ROOT_DIR)/benchmarks/load_database_dump.py --dump $(ROOT_DIR)/benchmarks/data/dump/virtassist_backup_20260213.dump

generate-embeddings:
	$(PYTHON) $(ROOT_DIR)/benchmarks/generate_embeddings.py --chunks

generate-dataset:
	$(PYTHON) $(ROOT_DIR)/benchmarks/generate_dataset.py --max-questions 500

run-benchmarks:
	$(PYTHON) $(ROOT_DIR)/benchmarks/run_comprehensive_benchmark.py --tier all --mode synthetic

run-dashboard:
	$(PYTHON) $(ROOT_DIR)/benchmarks/run_dashboard.py
