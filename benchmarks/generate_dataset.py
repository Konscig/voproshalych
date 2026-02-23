"""Скрипт для генерации датасета для RAG-бенчмарков.

Поддерживает различные режимы генерации:
- synthetic: генерация вопросов из чанков через LLM
- export-annotation: экспорт synthetic/manual датасета для ручной аннотации
"""

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _generate_item_id(text: str, prefix: str = "") -> str:
    """Генерирует уникальный ID на основе хеша текста."""
    hash_obj = hashlib.md5(text.encode("utf-8"))
    return f"{prefix}{hash_obj.hexdigest()[:12]}"


def _create_dataset_item(
    question: str,
    ground_truth_answer: str,
    chunk_id: Optional[int] = None,
    chunk_text: Optional[str] = None,
    confluence_url: Optional[str] = None,
    source: str = "synthetic",
    question_source: str = "synthetic",
    relevant_chunk_ids: Optional[List[int]] = None,
    user_score: Optional[int] = None,
    question_answer_id: Optional[int] = None,
    is_relevant_chunk_matched: Optional[int] = None,
) -> Dict[str, Any]:
    """Создать标准化ную запись датасета с метаданными."""
    item = {
        "id": _generate_item_id(question, f"{source[:3]}_"),
        "source": source,
        "question_source": question_source,
        "question": question.strip(),
        "ground_truth_answer": ground_truth_answer.strip(),
    }

    if chunk_id is not None:
        item["chunk_id"] = chunk_id
    if chunk_text:
        item["chunk_text"] = chunk_text[:500] if len(chunk_text) > 500 else chunk_text
    if confluence_url:
        item["confluence_url"] = confluence_url
    if relevant_chunk_ids:
        item["relevant_chunk_ids"] = relevant_chunk_ids
    if user_score is not None:
        item["user_score"] = user_score
    if question_answer_id is not None:
        item["question_answer_id"] = question_answer_id
    if is_relevant_chunk_matched is not None:
        item["is_relevant_chunk_matched"] = is_relevant_chunk_matched

    return item


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

                dataset_item = _create_dataset_item(
                    question=result["question"],
                    ground_truth_answer=result["ground_truth_answer"],
                    chunk_id=chunk.id,
                    chunk_text=chunk.text,
                    confluence_url=chunk.confluence_url,
                    source="synthetic",
                    question_source="synthetic",
                )

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

    logger.info(f"✅ Датасет сохранён: {output_path}")
    logger.info(f"📊 Сгенерировано {len(dataset)} пар вопрос-ответ")
    if generation_errors:
        logger.warning("📛 Ошибок генерации: %s", len(generation_errors))
        for err in generation_errors[:5]:
            logger.warning(f"  - {err}")
        if len(generation_errors) > 5:
            logger.warning(f"  ... и ещё {len(generation_errors) - 5} ошибок")

    print("\n=== Итог генерации датасета ===")
    print(f"Всего чанков с эмбеддингом: {available_chunks_total}")
    print(f"Успешно сгенерировано пар: {len(dataset)}")
    if generation_errors:
        print(f"Ошибок генерации: {len(generation_errors)}")
    print(f"Файл датасета: {output_path}")


def export_for_annotation(
    input_dataset: str,
    output_path: str,
    include_annotations: bool = False,
):
    """Экспортировать датасет в формат для ручной аннотации.

    Создает CSV/JSONL файл, удобный для редактирования аннотатором.

    Args:
        input_dataset: Путь к входному датасету
        output_path: Путь для сохранения файла аннотации
        include_annotations: Включить существующие аннотации из БД
    """
    dataset = _load_existing_dataset(input_dataset)
    if not dataset:
        logger.error("Датасет не найден: %s", input_dataset)
        return

    export_data = []
    for item in dataset:
        export_item = {
            "id": item.get("id", ""),
            "question": item.get("question", ""),
            "ground_truth_answer": item.get("ground_truth_answer", ""),
            "source": item.get("source", "unknown"),
            "question_source": item.get("question_source", "unknown"),
            "question_answer_id": item.get("question_answer_id"),
            "chunk_id": item.get("chunk_id"),
            "chunk_text": item.get("chunk_text", ""),
            "confluence_url": item.get("confluence_url"),
            "relevant_chunk_ids": item.get("relevant_chunk_ids", []),
            "relevant_urls": item.get("relevant_urls", []),
            "user_score": item.get("user_score"),
            "is_relevant_chunk_matched": item.get("is_relevant_chunk_matched"),
            "is_question_ok": 1,
            "is_answer_ok": 1,
            "is_chunk_ok": 1,
            "notes": "",
        }
        export_data.append(export_item)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if output_path.endswith(".csv"):
        import csv

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            if export_data:
                writer = csv.DictWriter(
                    f, fieldnames=list(export_data[0].keys()), extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(export_data)
    else:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Экспорт для аннотации: {output_path}")
    print(f"Экспортировано {len(export_data)} записей в {output_path}")


def main():
    """Главная функция CLI скрипта."""
    parser = argparse.ArgumentParser(
        description="Генерация датасета для RAG-бенчмарков"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "synthetic",
            "export-annotation",
        ],
        default="synthetic",
        help="Режим генерации датасета: synthetic (из чанков), export-annotation (экспорт для ручной аннотации)",
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

    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)

    if args.mode == "export-annotation":
        input_dataset = args.output or "benchmarks/data/dataset_latest.json"
        export_path = f"benchmarks/data/annotation_{timestamp}.json"
        export_for_annotation(input_dataset, export_path)
        return

    if args.check_only:
        output_path = args.output or f"benchmarks/data/dataset_{timestamp}.json"
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

    mode_prefix = {
        "synthetic": "synthetic",
    }.get(args.mode, "synthetic")

    output_path = (
        args.output or f"benchmarks/data/dataset_{mode_prefix}_{timestamp}.json"
    )

    max_questions = args.max_questions
    if args.num_samples is not None:
        max_questions = args.num_samples
        logger.warning(
            "Флаг --num-samples устарел, используйте --max-questions. "
            "Используем max_questions=%s",
            max_questions,
        )

    if args.mode == "synthetic":
        generate_synthetic_dataset(
            engine,
            max_questions=max_questions,
            output_path=output_path,
            skip_existing_dataset=args.skip_existing_dataset,
            generation_attempts=max(1, args.generation_attempts),
        )


if __name__ == "__main__":
    main()
