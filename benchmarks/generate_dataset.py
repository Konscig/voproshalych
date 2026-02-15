"""Скрипт для генерации синтетического золотого стандарта.

Использует LLM-судью для генерации вопросов и идеальных ответов на основе чанков.
"""

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qa.config import Config
from qa.database import Chunk, create_engine
from benchmarks.utils.llm_judge import LLMJudge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def generate_synthetic_dataset(
    engine, num_samples: int, output_path: str, skip_existing: bool = True
):
    """Сгенерировать синтетический датасет.

    Args:
        engine: Движок базы данных
        num_samples: Количество сэмплов для генерации
        output_path: Путь для сохранения датасета
        skip_existing: Пропускать чанки, для которых уже есть вопросы
    """
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    judge = LLMJudge()

    with Session(engine) as session:
        all_chunks = session.scalars(select(Chunk)).all()
        logger.info(f"Всего чанков в БД: {len(all_chunks)}")

        chunks_with_embeddings = [
            c for c in all_chunks if c.embedding is not None and len(c.embedding) > 0
        ]
        logger.info(f"Чанков с эмбеддингами: {len(chunks_with_embeddings)}")

        if len(chunks_with_embeddings) < num_samples:
            logger.warning(
                f"Недостаточно чанков с эмбеддингами: "
                f"требуется {num_samples}, доступно {len(chunks_with_embeddings)}"
            )
            num_samples = len(chunks_with_embeddings)

        random.seed(42)
        selected_chunks = random.sample(chunks_with_embeddings, num_samples)

        dataset = []

        for i, chunk in enumerate(selected_chunks, 1):
            try:
                result = judge.generate_question_from_chunk(chunk.text)

                dataset_item = {
                    "chunk_id": chunk.id,
                    "chunk_text": chunk.text[:500],
                    "question": result["question"],
                    "ground_truth_answer": result["ground_truth_answer"],
                    "confluence_url": chunk.confluence_url,
                }

                dataset.append(dataset_item)

                logger.info(f"[{i}/{num_samples}] Сгенерировано: {result['question']}")

                if i % 10 == 0:
                    logger.info(f"Прогресс: {i}/{num_samples}")

            except Exception as e:
                logger.error(f"Ошибка для чанка {chunk.id}: {e}")
                continue

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Датасет сохранён: {output_path}")
    logger.info(f"📊 Сгенерировано {len(dataset)} пар вопрос-ответ")


def main():
    """Главная функция CLI скрипта."""
    parser = argparse.ArgumentParser(
        description="Генерация синтетического датасета для бенчмарков"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Количество сэмплов для генерации (default: 50)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/data/golden_dataset_synthetic.json",
        help="Путь для сохранения датасета",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Только проверить существующий датасет",
    )

    args = parser.parse_args()

    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)

    if args.check_only:
        if os.path.exists(args.output):
            with open(args.output, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            print(f"Датасет существует: {args.output}")
            print(f"Количество записей: {len(dataset)}")
            if dataset:
                print(f"\nПример записи:")
                print(json.dumps(dataset[0], ensure_ascii=False, indent=2))
        else:
            print(f"Датасет не найден: {args.output}")
        return

    generate_synthetic_dataset(engine, args.num_samples, args.output)


if __name__ == "__main__":
    main()
