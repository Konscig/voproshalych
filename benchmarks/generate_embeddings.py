"""CLI скрипт для генерации эмбеддингов.

Использование:
    python generate_embeddings.py --all
    python generate_embeddings.py --score 5
    python generate_embeddings.py --check-coverage
"""

import argparse
import logging
import sys

from qa.config import Config
from qa.database import create_engine
from sentence_transformers import SentenceTransformer

from benchmarks.utils.embedding_generator import EmbeddingGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    """Главная функция CLI скрипта."""
    parser = argparse.ArgumentParser(description="Генерация эмбеддингов для вопросов")

    parser.add_argument(
        "--all",
        action="store_true",
        help="Генерировать эмбеддинги для всех вопросов",
    )

    parser.add_argument(
        "--score",
        type=int,
        choices=[1, 5],
        help="Генерировать эмбеддинги для вопросов с указанной оценкой",
    )

    parser.add_argument(
        "--check-coverage",
        action="store_true",
        help="Проверить покрытие эмбеддингами в базе данных",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Перезаписывать существующие эмбеддинги",
    )

    args = parser.parse_args()

    # Проверяем аргументы
    if not any([args.all, args.score, args.check_coverage]):
        parser.print_help()
        sys.exit(1)

    # Создаём движок базы данных
    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)

    # Загружаем модель эмбеддингов
    model_path = "saved_models/multilingual-e5-large-wikiutmn"
    encoder = SentenceTransformer(model_path, device="cpu")

    # Создаём генератор
    generator = EmbeddingGenerator(engine, encoder)

    # Выполняем запрос
    if args.check_coverage:
        stats = generator.check_coverage()
        print(f"\nСтатистика покрытия эмбеддингов:")
        print(f"Всего вопросов: {stats['total_questions']}")
        print(f"С эмбеддингами: {stats['with_embeddings']}")
        print(f"Без эмбеддингов: {stats['without_embeddings']}")
        print(f"Покрытие: {stats['coverage_percent']:.2f}%")

    elif args.all:
        stats = generator.generate_for_all_questions(overwrite=args.overwrite)
        print(f"\nРезультат генерации:")
        print(f"Всего вопросов: {stats['total_questions']}")
        print(f"Обработано: {stats['processed']}")
        print(f"Пропущено: {stats['skipped']}")
        print(f"Ошибок: {stats['errors']}")

    elif args.score:
        stats = generator.generate_for_all_questions(
            score_filter=args.score,
            overwrite=args.overwrite,
        )
        print(f"\nРезультат генерации для score={args.score}:")
        print(f"Всего вопросов: {stats['total_questions']}")
        print(f"Обработано: {stats['processed']}")
        print(f"Пропущено: {stats['skipped']}")
        print(f"Ошибок: {stats['errors']}")


if __name__ == "__main__":
    main()
