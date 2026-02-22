"""CLI и функции анализа использования чанков в retrieval."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from qa.config import Config
from qa.database import Chunk, QuestionAnswer, create_engine

logger = logging.getLogger(__name__)


def _build_synthetic_questions(engine: Engine, limit: int) -> list[dict[str, str]]:
    """Сгенерировать простой synthetic-набор вопросов из текста чанков."""
    with Session(engine) as session:
        chunks = session.scalars(
            select(Chunk)
            .where(Chunk.embedding.isnot(None), Chunk.text.isnot(None))
            .order_by(Chunk.id.asc())
            .limit(limit)
        ).all()

    questions: list[dict[str, str]] = []
    for chunk in chunks:
        text = (chunk.text or "").strip().replace("\n", " ")
        if not text:
            continue
        snippet = text[:120]
        questions.append(
            {
                "text": f"О чём этот материал: {snippet}?",
            }
        )
    return questions


def _load_real_questions(engine: Engine, limit: int) -> list[dict[str, str]]:
    """Загрузить реальные вопросы из таблицы QuestionAnswer."""
    with Session(engine) as session:
        items = session.scalars(
            select(QuestionAnswer)
            .where(
                QuestionAnswer.embedding.isnot(None),
                QuestionAnswer.question.isnot(None),
                QuestionAnswer.question != "",
            )
            .order_by(QuestionAnswer.id.asc())
            .limit(limit)
        ).all()
    return [{"text": item.question.strip()} for item in items if item.question]


def analyze_chunk_utilization(
    engine: Engine,
    encoder: SentenceTransformer,
    questions_source: str = "synthetic",
    question_limit: int = 500,
    top_k: int = 10,
) -> dict[str, Any]:
    """Проанализировать использование чанков retrieval-слоем.

    Args:
        engine: SQLAlchemy engine
        encoder: Модель эмбеддингов
        questions_source: Источник вопросов: synthetic или real
        question_limit: Ограничение числа вопросов
        top_k: Размер top-k retrieval

    Returns:
        Сводка utilization-метрик
    """
    with Session(engine) as session:
        all_chunk_ids = set(
            session.scalars(select(Chunk.id).where(Chunk.embedding.isnot(None))).all()
        )

    if questions_source == "real":
        questions = _load_real_questions(engine, question_limit)
    else:
        questions = _build_synthetic_questions(engine, question_limit)

    used_chunk_ids: set[int] = set()

    with Session(engine) as session:
        for question in questions:
            text = question.get("text", "").strip()
            if not text:
                continue
            query_embedding = encoder.encode(text)
            top_chunks = session.scalars(
                select(Chunk)
                .where(Chunk.embedding.isnot(None))
                .order_by(Chunk.embedding.cosine_distance(query_embedding))
                .limit(top_k)
            ).all()
            used_chunk_ids.update(chunk.id for chunk in top_chunks)

    total_chunks = len(all_chunk_ids)
    used_chunks = len(used_chunk_ids)

    return {
        "total_chunks": total_chunks,
        "used_chunks": used_chunks,
        "unused_chunks": max(total_chunks - used_chunks, 0),
        "utilization_rate": (used_chunks / total_chunks) if total_chunks else 0.0,
        "questions_source": questions_source,
        "question_count": len(questions),
        "top_k": top_k,
        "used_chunk_ids": sorted(used_chunk_ids),
    }


def main() -> None:
    """CLI-вход для анализа использования чанков."""
    parser = argparse.ArgumentParser(description="Анализ использования чанков")
    parser.add_argument(
        "--questions-source",
        choices=["synthetic", "real"],
        default="synthetic",
        help="Источник вопросов: synthetic | real",
    )
    parser.add_argument(
        "--question-limit",
        type=int,
        default=500,
        help="Лимит количества вопросов",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Сколько чанков брать в retrieval",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/reports/utilization.json",
        help="Путь для сохранения результата JSON",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
    encoder = SentenceTransformer(Config.EMBEDDING_MODEL_PATH, device="cpu")

    result = analyze_chunk_utilization(
        engine=engine,
        encoder=encoder,
        questions_source=args.questions_source,
        question_limit=args.question_limit,
        top_k=args.top_k,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
    logger.info("Результат сохранён: %s", output_path)


if __name__ == "__main__":
    main()
