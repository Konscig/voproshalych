"""CLI и функции анализа покрытия тем retrieval-слоем."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from qa.config import Config
from qa.database import Chunk, QuestionAnswer, create_engine

logger = logging.getLogger(__name__)


def _prepare_question_items(engine: Engine, limit: int) -> list[dict[str, Any]]:
    """Загрузить вопросы с эмбеддингами для кластеризации по темам."""
    with Session(engine) as session:
        rows = session.scalars(
            select(QuestionAnswer)
            .where(
                QuestionAnswer.embedding.isnot(None),
                QuestionAnswer.question.isnot(None),
                QuestionAnswer.question != "",
            )
            .order_by(QuestionAnswer.id.asc())
            .limit(limit)
        ).all()

    items: list[dict[str, Any]] = []
    for row in rows:
        if row.embedding is None or not row.question:
            continue
        items.append(
            {
                "id": row.id,
                "text": row.question.strip(),
                "embedding": np.array(row.embedding, dtype=float),
            }
        )
    return items


def analyze_topic_coverage(
    engine: Engine,
    encoder: SentenceTransformer,
    question_limit: int = 2000,
    n_topics: int = 20,
    top_k: int = 5,
) -> dict[str, Any]:
    """Проанализировать покрытие тем чанками retrieval-пайплайна."""
    questions = _prepare_question_items(engine, question_limit)
    if not questions:
        return {
            "n_topics": 0,
            "topic_coverage": [],
            "total_questions": 0,
            "avg_chunks_per_topic": 0.0,
            "top_k": top_k,
        }

    question_embeddings = np.array([item["embedding"] for item in questions])
    n_clusters = max(1, min(n_topics, len(question_embeddings)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    topic_labels = kmeans.fit_predict(question_embeddings)

    topic_coverage: list[dict[str, Any]] = []

    with Session(engine) as session:
        for topic_id in range(n_clusters):
            topic_questions = [
                item
                for item, label in zip(questions, topic_labels)
                if label == topic_id
            ]
            if not topic_questions:
                continue

            unique_chunks: set[int] = set()
            unique_urls: set[str] = set()

            for question in topic_questions:
                query_embedding = encoder.encode(question["text"])
                top_chunks = session.scalars(
                    select(Chunk)
                    .where(Chunk.embedding.isnot(None))
                    .order_by(Chunk.embedding.cosine_distance(query_embedding))
                    .limit(top_k)
                ).all()

                for chunk in top_chunks:
                    unique_chunks.add(chunk.id)
                    if chunk.confluence_url:
                        unique_urls.add(chunk.confluence_url)

            topic_coverage.append(
                {
                    "topic_id": int(topic_id),
                    "question_count": len(topic_questions),
                    "unique_chunks": len(unique_chunks),
                    "unique_urls": len(unique_urls),
                    "sample_questions": [
                        item["text"][:100] for item in topic_questions[:5]
                    ],
                }
            )

    avg_chunks = (
        float(
            sum(item["unique_chunks"] for item in topic_coverage) / len(topic_coverage)
        )
        if topic_coverage
        else 0.0
    )

    return {
        "n_topics": n_clusters,
        "topic_coverage": topic_coverage,
        "total_questions": len(questions),
        "avg_chunks_per_topic": avg_chunks,
        "top_k": top_k,
    }


def main() -> None:
    """CLI-вход для анализа покрытия тем."""
    parser = argparse.ArgumentParser(description="Анализ покрытия тем")
    parser.add_argument(
        "--question-limit",
        type=int,
        default=2000,
        help="Лимит количества вопросов",
    )
    parser.add_argument(
        "--n-topics",
        type=int,
        default=20,
        help="Количество тем (кластеров KMeans)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Размер top-k retrieval для оценки покрытия",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/reports/topic_coverage.json",
        help="Путь для сохранения JSON результата",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
    encoder = SentenceTransformer(Config.EMBEDDING_MODEL_PATH, device="cpu")

    result = analyze_topic_coverage(
        engine=engine,
        encoder=encoder,
        question_limit=args.question_limit,
        n_topics=args.n_topics,
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
