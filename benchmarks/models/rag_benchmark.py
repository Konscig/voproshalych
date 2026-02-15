"""Модуль бенчмарков RAG-системы.

Реализует три уровня тестирования:
- Tier 1: Retrieval Accuracy (качество поиска через pgvector)
- Tier 2: Generation Quality (качество генерации через Mistral)
- Tier 3: End-to-End (полный пайплайн)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from qa.config import Config
from qa.database import Chunk
from benchmarks.utils.llm_judge import LLMJudge

logger = logging.getLogger(__name__)


class RAGBenchmark:
    """Комплексный бенчмарк для RAG-системы.

    Attributes:
        engine: Движок базы данных SQLAlchemy
        encoder: Модель для генерации эмбеддингов
        judge: LLM-судья для оценки качества
        get_answer_func: Реальная функция генерации ответов из проекта (Mistral)
    """

    def __init__(
        self,
        engine: Engine,
        encoder: SentenceTransformer,
        judge: Optional[LLMJudge] = None,
        get_answer_func: Optional[callable] = None,
    ):
        """Инициализировать бенчмарк.

        Args:
            engine: Движок базы данных
            encoder: Модель для генерации эмбеддингов
            judge: LLM-судья (создастся автоматически если None)
            get_answer_func: Функция генерации ответов (по умолчанию из qa.main)
        """
        self.engine = engine
        self.encoder = encoder
        self.judge = judge or LLMJudge()

        if get_answer_func is None:
            from qa.main import get_answer

            def _get_answer_wrapper(question: str, context: str) -> str:
                """Обёртка для вызова реальной функции генерации ответов.

                Args:
                    question: Вопрос пользователя
                    context: Контекст (текст чанка)

                Returns:
                    Сгенерированный ответ
                """
                return get_answer(
                    dialog_history=[], knowledge_base=context, question=question
                )

            self.get_answer_func = _get_answer_wrapper
        else:
            self.get_answer_func = get_answer_func

        logger.info("RAGBenchmark инициализирован с реальной функцией генерации")

    def run_tier_1(
        self,
        dataset: List[Dict],
        top_k: int = 10,
    ) -> Dict[str, float]:
        """Выполнить бенчмарк Tier 1 (Retrieval Accuracy).

        Проверяет, находит ли векторный поиск через pgvector правильный чанк.
        ИСПОЛЬЗУЮТСЯ РЕАЛЬНЫЕ SQL ЗАПРОСЫ К БАЗЕ ДАННЫХ.

        Args:
            dataset: Список записей с chunk_id, question
            top_k: Количество результатов для поиска

        Returns:
            Словарь с метриками
        """
        logger.info(f"Запуск Tier 1 (Retrieval Accuracy) с {len(dataset)} записями")
        logger.info("Используются реальные SQL запросы с pgvector")

        hit_rates = {k: 0 for k in [1, 5, 10]}
        reciprocal_ranks = []

        for item in dataset:
            chunk_id = item["chunk_id"]
            question = item["question"]

            question_embedding = self.encoder.encode(question)

            with Session(self.engine) as session:
                top_chunks = session.scalars(
                    select(Chunk)
                    .order_by(Chunk.embedding.cosine_distance(question_embedding))
                    .limit(top_k)
                ).all()

                top_chunk_ids = [c.id for c in top_chunks]

                for k in [1, 5, 10]:
                    if k <= top_k and chunk_id in top_chunk_ids[:k]:
                        hit_rates[k] += 1

                if chunk_id in top_chunk_ids:
                    rank = top_chunk_ids.index(chunk_id) + 1
                    reciprocal_ranks.append(1.0 / rank)
                else:
                    reciprocal_ranks.append(0.0)

        total = len(dataset)

        metrics = {
            "tier": 1,
            "total_queries": total,
            "hit_rate@1": hit_rates[1] / total if total > 0 else 0,
            "hit_rate@5": hit_rates[5] / total if total > 0 else 0,
            "hit_rate@10": hit_rates[10] / total if total > 0 else 0,
            "mrr": np.mean(reciprocal_ranks) if reciprocal_ranks else 0,
        }

        logger.info(f"Tier 1 завершён. MRR: {metrics['mrr']:.4f}")

        return metrics

    def run_tier_2(
        self,
        dataset: List[Dict],
    ) -> Dict[str, float]:
        """Выполнить бенчмарк Tier 2 (Generation Quality).

        Оценивает качество генерации ответов на идеальном контексте
        С ИСПОЛЬЗОВАНИЕМ РЕАЛЬНОЙ ФУНКЦИИ ГЕНЕРАЦИИ (Mistral).

        Args:
            dataset: Список записей с question, ground_truth_answer, chunk_text

        Returns:
            Словарь с метриками
        """
        logger.info(f"Запуск Tier 2 (Generation Quality) с {len(dataset)} записями")
        logger.info("Используются реальные вызовы Mistral через qa.main.get_answer")

        faithfulness_scores = []
        relevance_scores = []
        errors = 0

        for item in dataset:
            try:
                question = item["question"]
                context = item["chunk_text"]

                system_answer = self.get_answer_func(question, context)

                if not system_answer:
                    logger.warning(f"Пустой ответ для вопроса: {question[:50]}...")
                    errors += 1
                    continue

                faithfulness = self.judge.evaluate_faithfulness(
                    question=question, context=context, answer=system_answer
                )

                relevance = self.judge.evaluate_answer_relevance(
                    question=question, answer=system_answer
                )

                faithfulness_scores.append(faithfulness)
                relevance_scores.append(relevance)

                logger.debug(
                    f"Вопрос: {question[:50]}... | "
                    f"Faithfulness: {faithfulness:.2f} | "
                    f"Relevance: {relevance:.2f}"
                )

            except Exception as e:
                logger.error(f"Ошибка оценки для вопроса '{item['question']}': {e}")
                errors += 1

        metrics = {
            "tier": 2,
            "total_queries": len(dataset),
            "successful_evaluations": len(faithfulness_scores),
            "errors": errors,
            "avg_faithfulness": np.mean(faithfulness_scores)
            if faithfulness_scores
            else 0,
            "avg_answer_relevance": np.mean(relevance_scores)
            if relevance_scores
            else 0,
        }

        logger.info(f"Tier 2 завершён. Faithfulness: {metrics['avg_faithfulness']:.2f}")

        return metrics

    def run_tier_3(
        self,
        dataset: List[Dict],
    ) -> Dict[str, float]:
        """Выполнить бенчмарк Tier 3 (End-to-End).

        Оценивает полный пайплайн RAG-системы (поиск + генерация).
        ИСПОЛЬЗУЮТСЯ РЕАЛЬНЫЕ SQL ЗАПРОСЫ + MISTRAL ГЕНЕРАЦИЯ.

        Args:
            dataset: Список записей с question, ground_truth_answer

        Returns:
            Словарь с метриками
        """
        logger.info(f"Запуск Tier 3 (End-to-End) с {len(dataset)} записями")
        logger.info("Используются реальные SQL запросы + Mistral генерация")

        e2e_scores = []
        semantic_similarities = []
        errors = 0

        for item in dataset:
            try:
                question = item["question"]
                ground_truth = item["ground_truth_answer"]

                question_embedding = self.encoder.encode(question)

                with Session(self.engine) as session:
                    retrieved_chunk = session.scalars(
                        select(Chunk)
                        .order_by(Chunk.embedding.cosine_distance(question_embedding))
                        .limit(1)
                    ).first()

                if not retrieved_chunk:
                    logger.warning(f"Чанк не найден для вопроса: {question[:50]}...")
                    errors += 1
                    continue

                retrieved_context = retrieved_chunk.text

                system_answer = self.get_answer_func(question, retrieved_context)

                if not system_answer:
                    logger.warning(f"Пустой ответ для вопроса: {question[:50]}...")
                    errors += 1
                    continue

                e2e_score = self.judge.evaluate_e2e_quality(
                    question=question,
                    system_answer=system_answer,
                    ground_truth_answer=ground_truth,
                )

                system_embedding = self.encoder.encode(system_answer)
                gt_embedding = self.encoder.encode(ground_truth)

                norm_system = np.linalg.norm(system_embedding)
                norm_gt = np.linalg.norm(gt_embedding)

                if norm_system > 0 and norm_gt > 0:
                    cosine_sim = np.dot(system_embedding, gt_embedding) / (
                        norm_system * norm_gt
                    )
                else:
                    cosine_sim = 0.0

                e2e_scores.append(e2e_score)
                semantic_similarities.append(cosine_sim)

                logger.debug(
                    f"E2E Score: {e2e_score:.2f} | Similarity: {cosine_sim:.4f}"
                )

            except Exception as e:
                logger.error(f"Ошибка E2E для вопроса '{item['question']}': {e}")
                errors += 1

        metrics = {
            "tier": 3,
            "total_queries": len(dataset),
            "successful_evaluations": len(e2e_scores),
            "errors": errors,
            "avg_e2e_score": np.mean(e2e_scores) if e2e_scores else 0,
            "avg_semantic_similarity": np.mean(semantic_similarities)
            if semantic_similarities
            else 0,
        }

        logger.info(f"Tier 3 завершён. E2E Score: {metrics['avg_e2e_score']:.2f}")

        return metrics

    def run_all_tiers(
        self,
        dataset: List[Dict],
        top_k: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """Выполнить все уровни тестирования.

        Args:
            dataset: Список записей датасета
            top_k: Количество результатов для поиска (Tier 1)

        Returns:
            Словарь с метриками всех уровней
        """
        logger.info("Запуск комплексного бенчмарка (все уровни)")

        results = {}

        results["tier_1"] = self.run_tier_1(dataset, top_k=top_k)
        results["tier_2"] = self.run_tier_2(dataset)
        results["tier_3"] = self.run_tier_3(dataset)

        logger.info("✅ Все уровни тестирования завершены")

        return results
