"""CLI скрипт для запуска комплексных бенчмарков RAG-системы.

Использование:
    python run_comprehensive_benchmark.py --tier all --limit 50
    python run_comprehensive_benchmark.py --tier 1 --dataset benchmarks/data/dataset_20260216_143000.json
"""

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from qa.config import Config
from qa.database import (
    BenchmarkRun,
    Chunk,
    create_engine,
    ensure_benchmark_runs_schema,
)
from benchmarks.models.rag_benchmark import RAGBenchmark
from benchmarks.utils.llm_judge import LLMJudge
from benchmarks.utils.report_generator import ReportGenerator
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Явно загружаем .env.docker для локального использования (для разработки)
# В Docker используется .env.docker автоматически через docker compose
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.docker")


def check_prerequisites(engine, judge: LLMJudge, dataset_path: str) -> bool:
    """Проверить предварительные условия для запуска бенчмарков.

    Args:
        engine: Движок базы данных
        judge: LLM-судья
        dataset_path: Путь к датасету

    Returns:
        True если все условия выполнены
    """
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    logger.info("Проверка предварительных условий...")

    with Session(engine) as session:
        total_chunks = session.scalars(select(Chunk)).all()
        chunks_with_embeddings = [
            c for c in total_chunks if c.embedding is not None and len(c.embedding) > 0
        ]

        logger.info(f"Всего чанков: {len(total_chunks)}")
        logger.info(f"Чанков с эмбеддингами: {len(chunks_with_embeddings)}")

        if len(chunks_with_embeddings) == 0:
            logger.error(
                "❌ Нет чанков с эмбеддингами! "
                "Запустите: python benchmarks/generate_embeddings.py --chunks"
            )
            return False

    if not os.path.exists(dataset_path):
        logger.error(
            f"❌ Датасет не найден: {dataset_path}\n"
            "Сгенерируйте датасет: python benchmarks/generate_dataset.py --num-samples 50"
        )
        return False

    logger.info("✅ Все предварительные условия выполнены")
    return True


def load_dataset(dataset_path: str, limit: Optional[int] = None) -> list:
    """Загрузить датасет из JSON файла.

    Args:
        dataset_path: Путь к файлу датасета
        limit: Ограничение количества записей

    Returns:
        Список записей датасета
    """
    logger.info(f"Загрузка датасета из {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]

    logger.info(f"Загружено {len(dataset)} записей")

    return dataset


def resolve_dataset_path(dataset_path: str) -> str:
    """Разрешить путь к датасету, учитывая versioned файлы по умолчанию."""
    if dataset_path != "benchmarks/data/golden_dataset_synthetic.json":
        return dataset_path

    if os.path.exists(dataset_path):
        return dataset_path

    candidates = sorted(glob.glob("benchmarks/data/dataset_*.json"))
    if candidates:
        latest = candidates[-1]
        logger.info("Используем последний versioned датасет: %s", latest)
        return latest

    return dataset_path


def run_benchmark(
    engine,
    encoder: SentenceTransformer,
    judge: LLMJudge,
    dataset: list,
    tier: str,
    top_k: int = 10,
):
    """Запустить бенчмарк.

    Args:
        engine: Движок базы данных
        encoder: Модель для генерации эмбеддингов
        judge: LLM-судья
        dataset: Датасет
        tier: Уровень бенчмарка (1, 2, 3, all)
        top_k: Количество результатов для поиска (Tier 1)

    Returns:
        Результаты бенчмарка
    """
    benchmark = RAGBenchmark(engine, encoder, judge)

    if tier == "all":
        return benchmark.run_all_tiers(dataset, top_k=top_k)
    elif tier == "1":
        return {"tier_1": benchmark.run_tier_1(dataset, top_k=top_k)}
    elif tier == "2":
        return {"tier_2": benchmark.run_tier_2(dataset)}
    elif tier == "3":
        return {"tier_3": benchmark.run_tier_3(dataset)}
    else:
        raise ValueError(f"Неизвестный уровень бенчмарка: {tier}")


def save_results(results: dict, output_dir: str, dataset_name: str, engine):
    """Сохранить результаты бенчмарка.

    Args:
        results: Результаты бенчмарка
        output_dir: Директория для сохранения
        dataset_name: Имя датасета
        engine: Движок базы данных
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_builtin(value: Any) -> Any:
        """Преобразовать numpy-типы в стандартные типы Python."""
        if isinstance(value, dict):
            return {k: to_builtin(v) for k, v in value.items()}
        if isinstance(value, list):
            return [to_builtin(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    normalized_results: dict = to_builtin(results)

    report_generator = ReportGenerator()
    run_metadata = report_generator.get_run_metadata()
    overall_status = report_generator.evaluate_overall_status(normalized_results)

    artifact_payload = dict(normalized_results)
    artifact_payload["run_metadata"] = run_metadata
    artifact_payload["overall_status"] = overall_status
    artifact_payload["dataset_file"] = os.path.basename(dataset_name)

    json_path = os.path.join(output_dir, f"rag_benchmark_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact_payload, f, ensure_ascii=False, indent=2)

    markdown_path = os.path.join(output_dir, f"rag_benchmark_{timestamp}.md")
    markdown_report = report_generator.generate_benchmark_report(
        normalized_results,
        dataset_name=dataset_name,
        metadata=run_metadata,
    )

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    ensure_benchmark_runs_schema(engine)

    from sqlalchemy.orm import Session

    with Session(engine) as session:
        session.add(
            BenchmarkRun(
                timestamp=datetime.now(),
                git_branch=run_metadata["git_branch"],
                git_commit_hash=run_metadata["git_commit_hash"],
                run_author=run_metadata["run_author"],
                dataset_file=os.path.basename(dataset_name),
                tier_1_metrics=normalized_results.get("tier_1"),
                tier_2_metrics=normalized_results.get("tier_2"),
                tier_3_metrics=normalized_results.get("tier_3"),
                overall_status=overall_status,
            )
        )
        session.commit()

    logger.info(f"✅ Результаты сохранены:")
    logger.info(f"   JSON: {json_path}")
    logger.info(f"   Markdown: {markdown_path}")
    logger.info("   DB table: benchmark_runs")


def print_results(results: dict):
    """Вывести результаты бенчмарка в консоль.

    Args:
        results: Результаты бенчмарка
    """
    print("\n" + "=" * 60)
    print("RAG BENCHMARK RESULTS")
    print("=" * 60 + "\n")

    for tier_name, tier_results in results.items():
        print(f"📊 {tier_name.upper()}")
        print("-" * 60)

        for key, value in tier_results.items():
            if key == "tier":
                continue
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print()

    print("=" * 60 + "\n")


def main():
    """Главная функция CLI скрипта."""
    parser = argparse.ArgumentParser(
        description="Запуск комплексных бенчмарков RAG-системы"
    )

    parser.add_argument(
        "--tier",
        type=str,
        choices=["1", "2", "3", "all"],
        required=True,
        help="Уровень бенчмарка (1, 2, 3 или all)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmarks/data/golden_dataset_synthetic.json",
        help=(
            "Путь к файлу датасета. Если используется путь по умолчанию и файл "
            "не найден, будет выбран последний benchmarks/data/dataset_*.json"
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничение количества записей из датасета",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Количество результатов для поиска (Tier 1, default: 10)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/reports",
        help="Директория для сохранения результатов",
    )

    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Пропустить проверку предварительных условий",
    )

    args = parser.parse_args()

    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)

    model_path = Config.EMBEDDING_MODEL_PATH
    encoder = SentenceTransformer(model_path, device="cpu")

    judge = LLMJudge()

    resolved_dataset = resolve_dataset_path(args.dataset)

    if not args.skip_checks:
        if not check_prerequisites(engine, judge, resolved_dataset):
            sys.exit(1)

    dataset = load_dataset(resolved_dataset, args.limit)

    logger.info(f"Запуск бенчмарка Tier {args.tier} с {len(dataset)} записями")

    results = run_benchmark(engine, encoder, judge, dataset, args.tier, args.top_k)

    save_results(results, args.output_dir, dataset_name=resolved_dataset, engine=engine)
    print_results(results)

    logger.info("✅ Бенчмарк успешно завершён")


if __name__ == "__main__":
    main()
