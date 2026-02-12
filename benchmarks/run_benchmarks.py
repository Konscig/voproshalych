"""CLI скрипт для запуска бенчмарков.

Использование:
    python run_benchmarks.py --tier 1 --dataset golden_set
    python run_benchmarks.py --tier 2 --dataset questions_with_url
"""

import argparse
import logging
import os
from datetime import datetime

from qa.config import Config
from qa.database import create_engine
from sentence_transformers import SentenceTransformer

from benchmarks.models.retrieval_benchmark import RetrievalBenchmark
from benchmarks.utils.data_loader import DataLoader
from benchmarks.utils.report_generator import ReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    """Главная функция CLI скрипта."""
    parser = argparse.ArgumentParser(description="Запуск бенчмарков для Вопрошалыча")

    parser.add_argument(
        "--tier",
        type=int,
        choices=[1, 2],
        help="Tier бенчмарка (1 - поиск похожих вопросов, 2 - поиск чанков)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=[
            "golden_set",
            "low_quality",
            "questions_with_url",
            "recent",
        ],
        help="Датасет для бенчмарка",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Количество результатов для поиска (default: 10)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/reports",
        help="Директория для сохранения отчётов",
    )

    args = parser.parse_args()

    # Проверяем обязательные аргументы
    if not args.tier or not args.dataset:
        parser.print_help()
        return

    # Создаём директорию для отчётов
    os.makedirs(args.output_dir, exist_ok=True)

    # Создаём движок базы данных
    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)

    # Загружаем модель эмбеддингов
    model_path = "saved_models/multilingual-e5-large-wikiutmn"
    encoder = SentenceTransformer(model_path, device="cpu")

    # Загружаем данные
    loader = DataLoader(engine)

    if args.dataset == "golden_set":
        questions = loader.load_golden_set()
        dataset_name = "Золотой набор (score=5)"
    elif args.dataset == "low_quality":
        questions = loader.load_low_quality_set(limit=200)
        dataset_name = "Низкое качество (score=1, 200)"
    elif args.dataset == "questions_with_url":
        questions = loader.load_questions_with_url()
        dataset_name = "Вопросы с URL"
    elif args.dataset == "recent":
        questions = loader.load_recent_questions(months=6, limit=1000)
        dataset_name = "Последние 6 месяцев (1000)"
    else:
        logging.error(f"Неизвестный датасет: {args.dataset}")
        return

    # Подготавливаем ground truth
    ground_truth_urls = loader.prepare_ground_truth_urls(questions)

    # Формируем список вопросов
    test_questions = [qa["question"] for qa in questions]

    logging.info(f"Запуск бенчмарка Tier {args.tier} с {len(test_questions)} вопросами")

    # Создаём бенчмарк
    benchmark = RetrievalBenchmark(engine, encoder)

    # Запускаем бенчмарк
    if args.tier == 1:
        metrics = benchmark.run_tier_1(
            test_questions=test_questions,
            ground_truth_urls=ground_truth_urls,
            top_k=args.top_k,
        )
        tier_name = "Tier 1 (поиск похожих вопросов)"
    else:
        metrics = benchmark.run_tier_2(
            test_questions=test_questions,
            ground_truth_urls=ground_truth_urls,
            top_k=args.top_k,
        )
        tier_name = "Tier 2 (поиск чанков)"

    # Выводим результаты
    logging.info(f"\nРезультаты бенчмарка {tier_name}:")
    for metric, value in sorted(metrics.items()):
        logging.info(f"  {metric}: {value:.4f}")

    # Генерируем отчёт
    generator = ReportGenerator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Markdown отчёт
    report = generator.generate_retrieval_report(
        metrics=metrics,
        dataset_name=f"{dataset_name} ({len(test_questions)} вопросов)",
        tier=str(args.tier),
    )

    report_path = os.path.join(
        args.output_dir, f"retrieval_tier{args.tier}_{args.dataset}_{timestamp}.md"
    )
    generator.save_report(report, report_path)

    # JSON метрики
    json_path = os.path.join(
        args.output_dir, f"retrieval_tier{args.tier}_{args.dataset}_{timestamp}.json"
    )
    generator.save_metrics_json(metrics, json_path)

    logging.info(f"\nОтчёт сохранён: {report_path}")
    logging.info(f"Метрики сохранены: {json_path}")


if __name__ == "__main__":
    main()
