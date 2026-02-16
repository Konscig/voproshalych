"""Скрипт для генерации синтетического золотого стандарта.

Использует LLM-судью для генерации вопросов и идеальных ответов на основе чанков.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from qa.config import Config
from qa.database import Chunk, create_engine
from benchmarks.utils.llm_judge import LLMJudge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Явно загружаем .env.docker для локального использования (для разработки)
# В Docker используется .env.docker автоматически через docker compose
load_dotenv(dotenv_path=".env.docker")


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
    from sqlalchemy import func, select
    from sqlalchemy.orm import Session

    judge = LLMJudge()

    def normalize_question(question: str) -> str:
        return " ".join(question.lower().split())

    def is_too_similar(candidate: str, existing: set[str]) -> bool:
        normalized_candidate = normalize_question(candidate)
        if normalized_candidate in existing:
            return True
        for existing_question in existing:
            if (
                SequenceMatcher(None, normalized_candidate, existing_question).ratio()
                >= 0.92
            ):
                return True
        return False

    with Session(engine) as session:
        total_chunks = session.scalars(select(func.count(Chunk.id))).one()
        logger.info(f"Всего чанков в БД: {total_chunks}")

        available_chunks = session.scalars(
            select(Chunk)
            .where(Chunk.embedding.isnot(None))
            .where(Chunk.text.isnot(None))
            .order_by(func.random())
        ).all()

        logger.info(f"Чанков с эмбеддингами: {len(available_chunks)}")

        if len(available_chunks) < num_samples:
            logger.warning(
                f"Недостаточно чанков с эмбеддингами: "
                f"требуется {num_samples}, доступно {len(available_chunks)}"
            )
            num_samples = len(available_chunks)

        dataset = []
        seen_questions: set[str] = set()

        for i, chunk in enumerate(available_chunks, 1):
            if len(dataset) >= num_samples:
                break
            try:
                result = judge.generate_question_from_chunk(chunk.text)
                question = result["question"].strip()

                if is_too_similar(question, seen_questions):
                    logger.info(
                        "Дубликат/слишком похожий вопрос, пропускаем: %s", question
                    )
                    continue

                seen_questions.add(normalize_question(question))

                dataset_item = {
                    "chunk_id": chunk.id,
                    "chunk_text": chunk.text[:500],
                    "question": question,
                    "ground_truth_answer": result["ground_truth_answer"],
                    "confluence_url": chunk.confluence_url,
                }

                dataset.append(dataset_item)

                logger.info(f"[{len(dataset)}/{num_samples}] Сгенерировано: {question}")

                if len(dataset) % 10 == 0:
                    logger.info(f"Прогресс: {len(dataset)}/{num_samples}")

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
        default=100,
        help="Количество сэмплов для генерации (default: 100)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь для сохранения датасета (по умолчанию versioned dataset_YYYYMMDD_HHMMSS.json)",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Только проверить существующий датасет",
    )

    args = parser.parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"benchmarks/data/dataset_{timestamp}.json"

    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)

    if args.check_only:
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            print(f"Датасет существует: {output_path}")
            print(f"Количество записей: {len(dataset)}")
            if dataset:
                print(f"\nПример записи:")
                print(json.dumps(dataset[0], ensure_ascii=False, indent=2))
        else:
            print(f"Датасет не найден: {output_path}")
        return

    generate_synthetic_dataset(engine, args.num_samples, output_path)


if __name__ == "__main__":
    main()
