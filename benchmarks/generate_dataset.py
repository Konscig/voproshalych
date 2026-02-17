"""Скрипт для генерации синтетического золотого стандарта.

Использует LLM-судью для генерации вопросов и идеальных ответов на основе чанков.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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


def _load_existing_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Загрузить существующий датасет для инкрементального дополнения."""
    if not dataset_path or not os.path.exists(dataset_path):
        return []

    with open(dataset_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        logger.warning("Файл %s не является списком, пропускаем", dataset_path)
        return []

    return payload


def generate_synthetic_dataset(
    engine,
    max_questions: int,
    output_path: str,
    skip_existing_dataset: str | None = None,
    generation_attempts: int = 3,
):
    """Сгенерировать синтетический датасет.

    Args:
        engine: Движок базы данных
        max_questions: Максимальное количество пар вопрос-ответ
        output_path: Путь для сохранения датасета
        skip_existing_dataset: Путь к существующему датасету для дополнения
        generation_attempts: Количество попыток генерации для одного чанка
    """
    from sqlalchemy import func, select
    from sqlalchemy.orm import Session

    judge = LLMJudge()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generation_errors: List[Dict[str, Any]] = []

    existing_dataset = _load_existing_dataset(skip_existing_dataset or "")
    existing_chunk_ids = {
        item.get("chunk_id")
        for item in existing_dataset
        if isinstance(item, dict) and item.get("chunk_id") is not None
    }
    dataset: List[Dict[str, Any]] = [
        item for item in existing_dataset if isinstance(item, dict)
    ]

    if existing_dataset:
        logger.info(
            "Загружен существующий датасет: %s записей, %s уникальных chunk_id",
            len(existing_dataset),
            len(existing_chunk_ids),
        )

    with Session(engine) as session:
        total_chunks = session.scalar(select(func.count(Chunk.id))) or 0
        logger.info(f"Всего чанков в БД: {total_chunks}")

        available_chunks = session.scalars(
            select(Chunk)
            .where(Chunk.embedding.isnot(None))
            .where(Chunk.text.isnot(None))
            .where(Chunk.text != "")
            .order_by(Chunk.id.asc())
        ).all()

        available_chunks_total = len(available_chunks)
        logger.info(f"Чанков с эмбеддингами: {available_chunks_total}")

        for i, chunk in enumerate(available_chunks, 1):
            if len(dataset) >= max_questions:
                logger.info("Достигнут лимит max_questions=%s", max_questions)
                break
            if chunk.id in existing_chunk_ids:
                continue

            try:
                result = None
                last_error: Exception | None = None
                for attempt in range(1, generation_attempts + 1):
                    try:
                        result = judge.generate_question_from_chunk(chunk.text)
                        break
                    except Exception as error:  # noqa: PERF203
                        last_error = error
                        logger.warning(
                            "Ошибка генерации chunk_id=%s (attempt %s/%s): %s",
                            chunk.id,
                            attempt,
                            generation_attempts,
                            error,
                        )

                if result is None:
                    generation_errors.append(
                        {
                            "chunk_id": chunk.id,
                            "reason": str(last_error or "generation_failed"),
                        }
                    )
                    continue

                dataset_item = {
                    "chunk_id": chunk.id,
                    "chunk_text": chunk.text[:500],
                    "question": result["question"].strip(),
                    "ground_truth_answer": result["ground_truth_answer"],
                    "confluence_url": chunk.confluence_url,
                }

                dataset.append(dataset_item)
                existing_chunk_ids.add(chunk.id)

                logger.info(
                    "[%s/%s] chunk_id=%s, сгенерирован вопрос",
                    len(dataset),
                    max_questions,
                    chunk.id,
                )

                if len(dataset) % 10 == 0:
                    logger.info(f"Прогресс: {len(dataset)}/{max_questions}")

            except Exception as e:
                logger.error(f"Ошибка для чанка {chunk.id}: {e}")
                generation_errors.append({"chunk_id": chunk.id, "reason": str(e)})
                continue

            if i % 200 == 0:
                logger.info("Проверено чанков: %s/%s", i, available_chunks_total)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    error_report_path = f"benchmarks/data/dataset_errors_{timestamp}.json"
    with open(error_report_path, "w", encoding="utf-8") as f:
        json.dump(generation_errors, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Датасет сохранён: {output_path}")
    logger.info(f"📊 Сгенерировано {len(dataset)} пар вопрос-ответ")
    logger.info("📛 Ошибок генерации: %s", len(generation_errors))
    logger.info("🧾 Отчёт об ошибках: %s", error_report_path)

    print("\n=== Итог генерации датасета ===")
    print(f"Всего чанков с эмбеддингом: {available_chunks_total}")
    print(f"Успешно сгенерировано пар: {len(dataset)}")
    print(f"Ошибок генерации: {len(generation_errors)}")
    print(f"Файл датасета: {output_path}")
    print(f"Файл ошибок: {error_report_path}")


def main():
    """Главная функция CLI скрипта."""
    parser = argparse.ArgumentParser(
        description="Генерация синтетического датасета для бенчмарков"
    )

    parser.add_argument(
        "--max-questions",
        type=int,
        default=500,
        help="Максимальное количество пар вопрос-ответ (default: 500)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Устаревший флаг (алиас для --max-questions)",
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

    parser.add_argument(
        "--skip-existing-dataset",
        type=str,
        default=None,
        help=(
            "Путь к существующему датасету. Если указан и файл существует, "
            "чанки из него будут пропущены"
        ),
    )

    parser.add_argument(
        "--generation-attempts",
        type=int,
        default=3,
        help="Количество попыток генерации вопроса для одного чанка",
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

    max_questions = args.max_questions
    if args.num_samples is not None:
        max_questions = args.num_samples
        logger.warning(
            "Флаг --num-samples устарел, используйте --max-questions. "
            "Используем max_questions=%s",
            max_questions,
        )

    generate_synthetic_dataset(
        engine,
        max_questions=max_questions,
        output_path=output_path,
        skip_existing_dataset=args.skip_existing_dataset,
        generation_attempts=max(1, args.generation_attempts),
    )


if __name__ == "__main__":
    main()
