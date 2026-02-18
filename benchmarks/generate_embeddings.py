"""CLI скрипт для генерации эмбеддингов.

Использование:
    python generate_embeddings.py --all
    python generate_embeddings.py --score 5
    python generate_embeddings.py --chunks
    python generate_embeddings.py --check-coverage
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qa.config import Config
from qa.database import create_engine

from benchmarks.utils.embedding_generator import EmbeddingGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Явно загружаем .env.docker для локального использования (для разработки)
# В Docker используется .env.docker автоматически через docker compose
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.docker")


def main():
    """Главная функция CLI скрипта."""
    parser = argparse.ArgumentParser(
        description="Генерация эмбеддингов для вопросов и чанков"
    )

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
        "--chunks",
        action="store_true",
        help="Генерировать эмбеддинги для всех чанков",
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

    if not any([args.all, args.score, args.chunks, args.check_coverage]):
        parser.print_help()
        sys.exit(1)

    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)

    if args.check_coverage:
        # Для проверки покрытия модель не нужна
        generator = EmbeddingGenerator(engine, encoder=None)
        stats = generator.check_coverage()
        print(f"\nСтатистика покрытия эмбеддингов (вопросы):")
        print(f"Всего вопросов: {stats['total_questions']}")
        print(f"С эмбеддингами: {stats['with_embeddings']}")
        print(f"Без эмбеддингов: {stats['without_embeddings']}")
        print(f"Покрытие: {stats['coverage_percent']:.2f}%")

        chunks_stats = generator.check_chunks_coverage()
        print(f"\nСтатистика покрытия эмбеддингов (чанки):")
        print(f"Всего чанков: {chunks_stats['total_chunks']}")
        print(f"С эмбеддингами: {chunks_stats['with_embeddings']}")
        print(f"Без эмбеддингов: {chunks_stats['without_embeddings']}")
        print(f"Покрытие: {chunks_stats['coverage_percent']:.2f}%")
        return

    # Для генерации эмбеддингов нужно загрузить модель
    from sentence_transformers import SentenceTransformer

    model_path = Config.EMBEDDING_MODEL_PATH
    encoder = SentenceTransformer(model_path, device="cpu")
    generator = EmbeddingGenerator(engine, encoder)

    if args.chunks:
        stats = generator.generate_for_chunks(overwrite=args.overwrite)
        print(f"\nРезультат генерации для чанков:")
        print(f"Всего чанков: {stats['total_chunks']}")
        print(f"Обработано: {stats['processed']}")
        print(f"Пропущено: {stats['skipped']}")
        print(f"Ошибок: {stats['errors']}")

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
