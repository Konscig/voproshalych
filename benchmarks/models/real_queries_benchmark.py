"""Модуль метрик retrieval по реальным пользовательским вопросам."""

from __future__ import annotations

from typing import Dict, List, Set

from sentence_transformers import SentenceTransformer
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from qa.database import Chunk, QuestionAnswer
from benchmarks.utils.evaluator import compute_retrieval_metrics


class RealQueriesBenchmark:
    """Компонент оценки retrieval по real-user-data.

    Использует реальные вопросы из таблицы `question_answer` и сравнивает
    retrieved URL из `Chunk` с URL вопроса в качестве прокси ground truth.
    """

    def __init__(self, engine: Engine, encoder: SentenceTransformer):
        """Инициализировать benchmark-компонент.

        Args:
            engine: Движок базы данных SQLAlchemy.
            encoder: Модель эмбеддингов для запросов.
        """
        self.engine = engine
        self.encoder = encoder

    def run(
        self,
        top_k: int = 10,
        score_filter: int | None = 5,
        limit: int = 500,
        latest: bool = False,
    ) -> Dict[str, float]:
        """Вычислить retrieval-метрики по реальным вопросам.

        Args:
            top_k: Размер top-k в retrieval.
            score_filter: Фильтр по score (например, 5). None отключает фильтр.
            limit: Максимальное количество вопросов.
            latest: Брать последние вопросы по `created_at`.

        Returns:
            Словарь с агрегированными retrieval-метриками.
        """
        with Session(self.engine) as session:
            query = select(QuestionAnswer).where(QuestionAnswer.embedding.isnot(None))

            if score_filter is not None:
                query = query.where(QuestionAnswer.score == score_filter)

            query = query.where(QuestionAnswer.confluence_url.isnot(None))
            if latest:
                query = query.order_by(QuestionAnswer.created_at.desc())
            else:
                query = query.order_by(QuestionAnswer.id.asc())

            questions = session.scalars(query.limit(limit)).all()

        test_keys: List[str] = []
        retrieved_results: Dict[str, List[str]] = {}
        ground_truth_urls: Dict[str, Set[str]] = {}

        with Session(self.engine) as session:
            for qa in questions:
                if not qa.question or not qa.confluence_url:
                    continue

                key = str(qa.id)
                embedding = self.encoder.encode(qa.question)
                chunks = session.scalars(
                    select(Chunk)
                    .where(Chunk.embedding.isnot(None))
                    .order_by(Chunk.embedding.cosine_distance(embedding))
                    .limit(top_k)
                ).all()

                test_keys.append(key)
                retrieved_results[key] = [
                    chunk.confluence_url for chunk in chunks if chunk.confluence_url
                ]
                ground_truth_urls[key] = {qa.confluence_url}

        metrics = compute_retrieval_metrics(
            test_questions=test_keys,
            retrieved_results=retrieved_results,
            ground_truth_urls=ground_truth_urls,
            k_values=[1, 3, 5, 10],
        )
        metrics.update(
            {
                "tier": "real_users",
                "total_queries": len(test_keys),
                "top_k": top_k,
                "score_filter": score_filter,
                "latest": latest,
            }
        )

        return metrics
