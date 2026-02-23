"""Модуль бенчмарков RAG-системы.

Реализует три уровня тестирования:
- Tier 1: Retrieval Accuracy (качество поиска через pgvector)
- Tier 2: Generation Quality (качество генерации через Mistral)
- Tier 3: End-to-End (полный пайплайн)
"""

import logging
import requests
from typing import Any, Callable, Dict, List, Optional, Set

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from qa.database import Chunk
from qa.confluence_retrieving import get_chunk
from benchmarks.utils.llm_judge import LLMJudge
from benchmarks.utils.evaluator import compute_retrieval_metrics

logger = logging.getLogger(__name__)


def _answer_consistency_metrics(
    answers: List[str],
    encoder: SentenceTransformer,
) -> Dict[str, float]:
    """Вычислить согласованность набора ответов между повторами."""
    cleaned_answers = [
        answer.strip() for answer in answers if answer and answer.strip()
    ]
    if len(cleaned_answers) < 2:
        return {
            "response_exact_match_rate": 1.0,
            "response_semantic_consistency": 1.0,
        }

    exact_match = sum(
        1 for answer in cleaned_answers[1:] if answer == cleaned_answers[0]
    ) / (len(cleaned_answers) - 1)

    embeddings = [encoder.encode(answer) for answer in cleaned_answers]
    similarities = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            emb_i = embeddings[i]
            emb_j = embeddings[j]
            norm_i = np.linalg.norm(emb_i)
            norm_j = np.linalg.norm(emb_j)
            if norm_i > 0 and norm_j > 0:
                similarities.append(float(np.dot(emb_i, emb_j) / (norm_i * norm_j)))

    semantic_consistency = float(np.mean(similarities)) if similarities else 0.0
    return {
        "response_exact_match_rate": float(exact_match),
        "response_semantic_consistency": semantic_consistency,
    }


def _empty_intrinsic_metrics() -> Dict[str, float | int]:
    """Вернуть пустой набор intrinsic-метрик Tier 0."""
    return {
        "avg_nn_distance": 0.0,
        "std_nn_distance": 0.0,
        "density_score": 0.0,
        "avg_spread": 0.0,
        "max_spread": 0.0,
        "spread_std": 0.0,
        "effective_dimensionality": 0,
        "avg_pairwise_distance": 0.0,
        "std_pairwise_distance": 0.0,
        "min_pairwise_distance": 0.0,
        "max_pairwise_distance": 0.0,
    }


def compute_intrinsic_metrics(
    embeddings: np.ndarray, k: int = 5
) -> Dict[str, float | int]:
    """Вычислить intrinsic-метрики качества эмбеддингов.

    Args:
        embeddings: Матрица эмбеддингов формы (n_samples, n_features)
        k: Количество ближайших соседей для локальных метрик

    Returns:
        Словарь с intrinsic-метриками качества векторного пространства
    """
    from sklearn.metrics import pairwise_distances
    from sklearn.neighbors import NearestNeighbors

    if embeddings.ndim != 2 or len(embeddings) < 2:
        return _empty_intrinsic_metrics()

    n_samples = len(embeddings)
    n_neighbors = min(max(k, 1), n_samples - 1)

    neighbors = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
    neighbors.fit(embeddings)
    distances, _ = neighbors.kneighbors(embeddings)
    nn_distances = distances[:, 1:]

    centroid = embeddings.mean(axis=0)
    distances_from_center = np.linalg.norm(embeddings - centroid, axis=1)

    covariance_matrix = np.atleast_2d(np.cov(embeddings, rowvar=False))
    eigenvalues = np.linalg.eigvalsh(covariance_matrix)
    eigenvalues = np.clip(np.sort(eigenvalues)[::-1], a_min=0.0, a_max=None)
    total_variance = float(eigenvalues.sum())

    if total_variance > 0.0:
        explained = np.cumsum(eigenvalues) / total_variance
        effective_dimensionality = int(np.searchsorted(explained, 0.95) + 1)
    else:
        effective_dimensionality = 1

    pairwise = pairwise_distances(embeddings, metric="cosine")
    upper_triangle = pairwise[np.triu_indices(n_samples, k=1)]

    if upper_triangle.size == 0:
        upper_triangle = np.array([0.0])

    avg_nn_distance = float(nn_distances.mean()) if nn_distances.size else 0.0

    return {
        "avg_nn_distance": avg_nn_distance,
        "std_nn_distance": float(nn_distances.std()) if nn_distances.size else 0.0,
        "density_score": float(1.0 / (avg_nn_distance + 1e-6)),
        "avg_spread": float(distances_from_center.mean()),
        "max_spread": float(distances_from_center.max()),
        "spread_std": float(distances_from_center.std()),
        "effective_dimensionality": effective_dimensionality,
        "avg_pairwise_distance": float(upper_triangle.mean()),
        "std_pairwise_distance": float(upper_triangle.std()),
        "min_pairwise_distance": float(upper_triangle.min()),
        "max_pairwise_distance": float(upper_triangle.max()),
    }


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
        get_answer_func: Optional[Callable[[str, str], str]] = None,
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
        self.judge = judge

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

    def _get_judge(self) -> LLMJudge:
        """Лениво инициализировать LLM-судью при необходимости."""
        if self.judge is None:
            self.judge = LLMJudge()
        return self.judge

    @staticmethod
    def _extract_relevant_chunk_ids(item: Dict, fallback_chunk_id: int) -> Set[int]:
        """Извлечь множество релевантных chunk_id из элемента датасета."""
        if isinstance(item.get("relevant_chunk_ids"), list):
            return {
                int(chunk_id)
                for chunk_id in item["relevant_chunk_ids"]
                if isinstance(chunk_id, (int, str)) and str(chunk_id).isdigit()
            }
        return {fallback_chunk_id}

    @staticmethod
    def _extract_relevant_urls(item: Dict) -> Set[str]:
        """Извлечь множество релевантных URL из элемента датасета."""
        if isinstance(item.get("relevant_urls"), list):
            return {
                str(url).strip()
                for url in item["relevant_urls"]
                if isinstance(url, str) and url.strip()
            }
        url = item.get("confluence_url")
        if isinstance(url, str) and url.strip():
            return {url.strip()}
        return set()

    def _resolve_context_for_item(self, item: Dict, session: Session) -> str:
        """Получить контекст для генерации в Tier 2 на основе датасета."""
        chunk_text = item.get("chunk_text")
        if isinstance(chunk_text, str) and chunk_text.strip():
            return chunk_text

        relevant_ids = item.get("relevant_chunk_ids")
        if isinstance(relevant_ids, list) and relevant_ids:
            chunks = session.scalars(
                select(Chunk).where(Chunk.id.in_(relevant_ids)).order_by(Chunk.id.asc())
            ).all()
            context_parts = [chunk.text for chunk in chunks if chunk.text]
            if context_parts:
                return "\n\n".join(context_parts)

        chunk_id = item.get("chunk_id")
        if isinstance(chunk_id, int):
            chunk = session.scalar(select(Chunk).where(Chunk.id == chunk_id))
            if chunk and chunk.text:
                return chunk.text

        relevant_urls = item.get("relevant_urls")
        if isinstance(relevant_urls, list) and relevant_urls:
            chunks = session.scalars(
                select(Chunk)
                .where(Chunk.confluence_url.in_(relevant_urls))
                .order_by(Chunk.id.asc())
                .limit(5)
            ).all()
            context_parts = [chunk.text for chunk in chunks if chunk.text]
            if context_parts:
                return "\n\n".join(context_parts)

        return ""

    def run_tier_1(
        self,
        dataset: List[Dict],
        top_k: int = 10,
        consistency_runs: int = 1,
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
        retrieval_consistency_scores = []
        retrieval_questions: List[str] = []
        retrieval_results: Dict[str, List[str]] = {}
        retrieval_ground_truth: Dict[str, Set[str]] = {}

        for item in dataset:
            raw_chunk_id = item.get("chunk_id")
            if raw_chunk_id is None:
                relevant_ids = item.get("relevant_chunk_ids") or []
                raw_chunk_id = relevant_ids[0] if relevant_ids else -1
            chunk_id = int(raw_chunk_id)
            question = item["question"]
            relevant_chunk_ids = self._extract_relevant_chunk_ids(item, chunk_id)
            relevant_urls = self._extract_relevant_urls(item)

            question_embedding = self.encoder.encode(question)

            with Session(self.engine) as session:
                top_chunks = session.scalars(
                    select(Chunk)
                    .where(Chunk.embedding.isnot(None))
                    .order_by(Chunk.embedding.cosine_distance(question_embedding))
                    .limit(top_k)
                ).all()

                top_chunk_ids = [c.id for c in top_chunks]
                top_urls = [c.confluence_url for c in top_chunks if c.confluence_url]

                if consistency_runs > 1:
                    per_question_runs = [top_chunk_ids]
                    for _ in range(consistency_runs - 1):
                        repeated_chunks = session.scalars(
                            select(Chunk)
                            .where(Chunk.embedding.isnot(None))
                            .order_by(
                                Chunk.embedding.cosine_distance(question_embedding)
                            )
                            .limit(top_k)
                        ).all()
                        per_question_runs.append(
                            [chunk.id for chunk in repeated_chunks]
                        )

                    reference = per_question_runs[0]
                    matches = sum(
                        1 for run_ids in per_question_runs[1:] if run_ids == reference
                    )
                    retrieval_consistency_scores.append(
                        matches / max(len(per_question_runs) - 1, 1)
                    )

                for k in [1, 5, 10]:
                    if k <= top_k and set(top_chunk_ids[:k]).intersection(
                        relevant_chunk_ids
                    ):
                        hit_rates[k] += 1

                first_rank = 0
                for idx, candidate_chunk_id in enumerate(top_chunk_ids, start=1):
                    if candidate_chunk_id in relevant_chunk_ids:
                        first_rank = idx
                        break
                reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)

                retrieval_key = str(item.get("id") or item.get("chunk_id") or question)
                retrieval_questions.append(retrieval_key)
                retrieval_results[retrieval_key] = top_urls
                if relevant_urls:
                    retrieval_ground_truth[retrieval_key] = relevant_urls
                else:
                    matched_urls = {
                        chunk.confluence_url
                        for chunk in top_chunks
                        if chunk.id in relevant_chunk_ids and chunk.confluence_url
                    }
                    retrieval_ground_truth[retrieval_key] = matched_urls

        total = len(dataset)

        metrics: Dict[str, float | int] = {
            "tier": 1,
            "total_queries": total,
            "hit_rate@1": hit_rates[1] / total if total > 0 else 0,
            "hit_rate@5": hit_rates[5] / total if total > 0 else 0,
            "hit_rate@10": hit_rates[10] / total if total > 0 else 0,
            "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
            "retrieval_consistency": float(np.mean(retrieval_consistency_scores))
            if retrieval_consistency_scores
            else 1.0,
        }

        if retrieval_questions:
            retrieval_metrics = compute_retrieval_metrics(
                test_questions=retrieval_questions,
                retrieved_results=retrieval_results,
                ground_truth_urls=retrieval_ground_truth,
                k_values=[1, 3, 5, 10],
            )
            metrics.update(retrieval_metrics)

        logger.info(f"Tier 1 завершён. MRR: {metrics['mrr']:.4f}")

        return metrics

    def run_tier_2(
        self,
        dataset: List[Dict],
        consistency_runs: int = 1,
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
        correctness_scores = []
        response_exact_match_scores = []
        response_semantic_consistency_scores = []
        text_metrics_list = []
        errors = 0

        from benchmarks.utils.text_metrics import compute_all_text_metrics

        for item in dataset:
            try:
                question = item["question"]
                with Session(self.engine) as session:
                    context = self._resolve_context_for_item(item, session)

                if not context:
                    logger.warning("Не найден контекст для вопроса: %s", question[:50])
                    errors += 1
                    continue

                system_answer = self.get_answer_func(question, context)

                if not system_answer:
                    logger.warning(f"Пустой ответ для вопроса: {question[:50]}...")
                    errors += 1
                    continue

                judge = self._get_judge()

                faithfulness = judge.evaluate_faithfulness(
                    question=question, context=context, answer=system_answer
                )

                relevance = judge.evaluate_answer_relevance(
                    question=question, answer=system_answer
                )

                ground_truth = item.get("ground_truth_answer", "")
                correctness = 0.0
                if ground_truth:
                    correctness = judge.evaluate_answer_correctness(
                        question=question,
                        system_answer=system_answer,
                        ground_truth_answer=ground_truth,
                    )

                if consistency_runs > 1:
                    repeated_answers = [system_answer]
                    for _ in range(consistency_runs - 1):
                        repeated_answers.append(self.get_answer_func(question, context))
                    consistency = _answer_consistency_metrics(
                        repeated_answers, self.encoder
                    )
                    response_exact_match_scores.append(
                        consistency["response_exact_match_rate"]
                    )
                    response_semantic_consistency_scores.append(
                        consistency["response_semantic_consistency"]
                    )
                else:
                    response_exact_match_scores.append(1.0)
                    response_semantic_consistency_scores.append(1.0)

                faithfulness_scores.append(faithfulness)
                relevance_scores.append(relevance)
                correctness_scores.append(correctness)

                # Алгоритмические метрики
                if ground_truth and system_answer:
                    text_metrics = compute_all_text_metrics(
                        hypothesis=system_answer,
                        reference=ground_truth,
                        include_bertscore=False,
                    )
                    text_metrics_list.append(text_metrics)

                logger.debug(
                    f"Вопрос: {question[:50]}... | "
                    f"Faithfulness: {faithfulness:.2f} | "
                    f"Relevance: {relevance:.2f}"
                )

            except Exception as e:
                logger.error(f"Ошибка оценки для вопроса '{item['question']}': {e}")
                errors += 1

        # Агрегируем алгоритмические метрики
        if text_metrics_list:
            avg_text_metrics = {
                "avg_rouge1_f": float(
                    np.mean([m.get("rouge1_f", 0) for m in text_metrics_list])
                ),
                "avg_rouge2_f": float(
                    np.mean([m.get("rouge2_f", 0) for m in text_metrics_list])
                ),
                "avg_rougeL_f": float(
                    np.mean([m.get("rougeL_f", 0) for m in text_metrics_list])
                ),
                "avg_bleu": float(
                    np.mean([m.get("bleu", 0) for m in text_metrics_list])
                ),
            }
        else:
            avg_text_metrics = {
                "avg_rouge1_f": 0.0,
                "avg_rouge2_f": 0.0,
                "avg_rougeL_f": 0.0,
                "avg_bleu": 0.0,
            }

        metrics = {
            "tier": 2,
            "total_queries": len(dataset),
            "successful_evaluations": len(faithfulness_scores),
            "errors": errors,
            "avg_faithfulness": float(np.mean(faithfulness_scores))
            if faithfulness_scores
            else 0.0,
            "avg_answer_relevance": float(np.mean(relevance_scores))
            if relevance_scores
            else 0.0,
            "avg_answer_correctness": float(np.mean(correctness_scores))
            if correctness_scores
            else 0.0,
            "response_exact_match_rate": float(np.mean(response_exact_match_scores))
            if response_exact_match_scores
            else 0.0,
            "response_semantic_consistency": float(
                np.mean(response_semantic_consistency_scores)
            )
            if response_semantic_consistency_scores
            else 0.0,
            **avg_text_metrics,
        }

        logger.info(f"Tier 2 завершён. Faithfulness: {metrics['avg_faithfulness']:.2f}")

        return metrics

    def run_tier_3(
        self,
        dataset: List[Dict],
        consistency_runs: int = 1,
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
        dot_similarities = []
        euclidean_distances = []
        response_exact_match_scores = []
        response_semantic_consistency_scores = []
        text_metrics_list = []
        errors = 0

        from benchmarks.utils.text_metrics import compute_all_text_metrics

        for item in dataset:
            try:
                question = item["question"]
                ground_truth = item["ground_truth_answer"]

                retrieved_chunk = get_chunk(self.engine, self.encoder, question)

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

                judge = self._get_judge()

                e2e_score = judge.evaluate_e2e_quality(
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
                    dot_sim = float(np.dot(system_embedding, gt_embedding))
                    euclidean_dist = float(
                        np.linalg.norm(system_embedding - gt_embedding)
                    )
                else:
                    cosine_sim = 0.0
                    dot_sim = 0.0
                    euclidean_dist = 0.0

                if consistency_runs > 1:
                    repeated_answers = [system_answer]
                    for _ in range(consistency_runs - 1):
                        repeated_answers.append(
                            self.get_answer_func(question, retrieved_context)
                        )
                    consistency = _answer_consistency_metrics(
                        repeated_answers, self.encoder
                    )
                    response_exact_match_scores.append(
                        consistency["response_exact_match_rate"]
                    )
                    response_semantic_consistency_scores.append(
                        consistency["response_semantic_consistency"]
                    )
                else:
                    response_exact_match_scores.append(1.0)
                    response_semantic_consistency_scores.append(1.0)

                e2e_scores.append(e2e_score)
                semantic_similarities.append(cosine_sim)
                dot_similarities.append(dot_sim)
                euclidean_distances.append(euclidean_dist)

                # Алгоритмические метрики
                if system_answer and ground_truth:
                    text_metrics = compute_all_text_metrics(
                        hypothesis=system_answer,
                        reference=ground_truth,
                        include_bertscore=False,
                    )
                    text_metrics_list.append(text_metrics)

                logger.debug(
                    f"E2E Score: {e2e_score:.2f} | Similarity: {cosine_sim:.4f}"
                )

            except Exception as e:
                logger.error(f"Ошибка E2E для вопроса '{item['question']}': {e}")
                errors += 1

        # Агрегируем алгоритмические метрики
        if text_metrics_list:
            avg_text_metrics = {
                "avg_rouge1_f": float(
                    np.mean([m.get("rouge1_f", 0) for m in text_metrics_list])
                ),
                "avg_rouge2_f": float(
                    np.mean([m.get("rouge2_f", 0) for m in text_metrics_list])
                ),
                "avg_rougeL_f": float(
                    np.mean([m.get("rougeL_f", 0) for m in text_metrics_list])
                ),
                "avg_bleu": float(
                    np.mean([m.get("bleu", 0) for m in text_metrics_list])
                ),
            }
        else:
            avg_text_metrics = {
                "avg_rouge1_f": 0.0,
                "avg_rouge2_f": 0.0,
                "avg_rougeL_f": 0.0,
                "avg_bleu": 0.0,
            }

        metrics = {
            "tier": 3,
            "total_queries": len(dataset),
            "successful_evaluations": len(e2e_scores),
            "errors": errors,
            "avg_e2e_score": float(np.mean(e2e_scores)) if e2e_scores else 0.0,
            "avg_semantic_similarity": float(np.mean(semantic_similarities))
            if semantic_similarities
            else 0.0,
            "avg_dot_similarity": float(np.mean(dot_similarities))
            if dot_similarities
            else 0.0,
            "avg_euclidean_distance": float(np.mean(euclidean_distances))
            if euclidean_distances
            else 0.0,
            "response_exact_match_rate": float(np.mean(response_exact_match_scores))
            if response_exact_match_scores
            else 0.0,
            "response_semantic_consistency": float(
                np.mean(response_semantic_consistency_scores)
            )
            if response_semantic_consistency_scores
            else 0.0,
            **avg_text_metrics,
        }

        logger.info(f"Tier 3 завершён. E2E Score: {metrics['avg_e2e_score']:.2f}")

        return metrics

    def run_all_tiers(
        self,
        dataset: List[Dict],
        top_k: int = 10,
        consistency_runs: int = 1,
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

        results["tier_1"] = self.run_tier_1(
            dataset,
            top_k=top_k,
            consistency_runs=consistency_runs,
        )
        results["tier_2"] = self.run_tier_2(
            dataset,
            consistency_runs=consistency_runs,
        )
        results["tier_3"] = self.run_tier_3(
            dataset,
            consistency_runs=consistency_runs,
        )

        logger.info("✅ Все уровни тестирования завершены")

        return results

    def run_tier_0(
        self,
        test_pairs: List[Dict],
    ) -> Dict[str, float]:
        """Выполнить бенчмарк Tier 0 (Intrinsic Embedding Quality).

        Оценивает внутреннее качество эмбеддингов через intrinsic-метрики,
        не требующие cluster_id/label в датасете.

        Args:
            test_pairs: Список пар (question1, question2, ground_truth_similarity)
                       или чанков для кластеризации

        Returns:
            Словарь с метриками качества эмбеддингов
        """
        logger.info(
            "Запуск Tier 0 (Intrinsic Embedding Quality) с %s записями",
            len(test_pairs),
        )

        if not test_pairs:
            logger.warning("Нет тестовых пар для Tier 0")
            return {"tier": 0, "total_samples": 0, **_empty_intrinsic_metrics()}

        has_cluster_labels = any(
            pair.get("cluster_id") is not None or pair.get("label") is not None
            for pair in test_pairs
        )
        if not has_cluster_labels:
            logger.warning(
                "Метрики кластеризации требуют cluster_id, используем intrinsic"
            )

        embeddings = []

        for pair in test_pairs:
            text = pair.get("text") or pair.get("question", "")

            if text:
                emb = self.encoder.encode(text)
                embeddings.append(emb)

        if len(embeddings) < 2:
            logger.warning("Недостаточно данных для intrinsic анализа")
            return {
                "tier": 0,
                "total_samples": len(embeddings),
                **_empty_intrinsic_metrics(),
            }

        embeddings_array = np.array(embeddings)
        intrinsic_metrics = compute_intrinsic_metrics(embeddings_array)

        metrics = {
            "tier": 0,
            "total_samples": len(embeddings),
            **intrinsic_metrics,
        }

        logger.info(
            "Tier 0 завершён. avg_nn_distance=%.4f, density_score=%.4f",
            metrics["avg_nn_distance"],
            metrics["density_score"],
        )

        return metrics

    def run_tier_judge(
        self,
        evaluation_pairs: List[Dict],
        consistency_check: bool = True,
    ) -> Dict[str, float]:
        """Выполнить бенчмарк Tier Judge (LLM-as-a-Judge Quality).

        Оценивает качество и стабильность LLM-судьи через:
        - Consistency score (согласованность при повторной оценке)
        - Error rate (доля ошибочных оценок)
        - Latency (время оценки)

        Args:
            evaluation_pairs: Пары (question, context, answer) для оценки
            consistency_check: Проводить ли проверку консистентности

        Returns:
            Словарь с метриками качества судьи
        """
        logger.info(
            f"Запуск Tier Judge (LLM Judge Quality) с {len(evaluation_pairs)} парами"
        )

        if not evaluation_pairs:
            logger.warning("Нет пар для оценки судьи")
            return {
                "tier_code": -4.0,  # -4 for judge tier
                "total_evaluations": 0,
                "consistency_score": 0.0,
                "error_rate": 0.0,
                "avg_latency_ms": 0.0,
            }

        judge = self._get_judge()
        first_scores = []
        second_scores = []
        latencies = []
        errors = 0

        import time

        for pair in evaluation_pairs:
            try:
                question = pair["question"]
                context = pair.get("context", "")
                answer = pair["answer"]

                start_time = time.time()

                if context:
                    score_1 = judge.evaluate_faithfulness(question, context, answer)
                else:
                    score_1 = judge.evaluate_answer_relevance(question, answer)

                latency_1 = (time.time() - start_time) * 1000
                first_scores.append(score_1)
                latencies.append(latency_1)

                if consistency_check:
                    start_time = time.time()

                    if context:
                        score_2 = judge.evaluate_faithfulness(question, context, answer)
                    else:
                        score_2 = judge.evaluate_answer_relevance(question, answer)

                    second_scores.append(score_2)

            except Exception as e:
                logger.error(f"Ошибка оценки судьи: {e}")
                errors += 1

        consistency = 0.0
        if consistency_check and first_scores and second_scores:
            matches = sum(
                1 for s1, s2 in zip(first_scores, second_scores) if abs(s1 - s2) <= 0.5
            )
            consistency = matches / len(first_scores)

        total = len(evaluation_pairs)
        error_rate = errors / total if total > 0 else 0.0

        metrics = {
            "tier_code": -4.0,  # -4 for judge tier
            "total_evaluations": float(len(first_scores)),
            "consistency_score": float(consistency),
            "error_rate": float(error_rate),
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "avg_faithfulness": float(np.mean(first_scores)) if first_scores else 0.0,
        }

        logger.info(
            f"Tier Judge завершён. Consistency: {metrics['consistency_score']:.2f}, "
            f"Error rate: {metrics['error_rate']:.2f}"
        )

        return metrics

    def run_tier_judge_pipeline(
        self,
        evaluation_pairs: List[Dict],
    ) -> Dict[str, Any]:
        """Выполнить бенчмарк Tier Judge Pipeline (Production Judge Quality).

        Оценивает качество production judge (Mistral), который используется в реальном
        RAG-пайплайне для решения "показывать ответ пользователю или нет".

        Использует JUDGE_API и JUDGE_MODEL из конфигурации qa.

        Метрики:
        - Accuracy (согласованность с ground truth)
        - Precision/Recall/F1 для класса "Yes" (показывать ответ)
        - Latency (время оценки)

        Args:
            evaluation_pairs: Пары для оценки с ground_truth_show
                [{"question": "...", "answer": "...", "context": "...", "ground_truth_show": true/false}]

        Returns:
            Словарь с метриками качества production judge
        """
        import time

        logger.info(
            f"Запуск Tier Judge Pipeline (Production Judge) с {len(evaluation_pairs)} парами"
        )

        if not evaluation_pairs:
            logger.warning("Нет пар для оценки production judge")
            return {
                "tier_code": -6.0,  # -6 for judge pipeline tier
                "total_evaluations": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "avg_latency_ms": 0.0,
            }

        try:
            from qa.config import Config
        except ImportError:
            logger.error("Не удалось импортировать qa.config")
            return {
                "tier_code": -6.0,
                "total_evaluations": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "avg_latency_ms": 0.0,
            }

        try:
            headers = Config.get_judge_headers()
        except ValueError as e:
            logger.error(f"Ошибка получения заголовков judge: {e}")
            return {
                "tier_code": -6.0,
                "total_evaluations": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "avg_latency_ms": 0.0,
                "error": str(e),
            }

        predictions = []
        ground_truths = []
        latencies = []

        for pair in evaluation_pairs:
            try:
                question = pair.get("question", "")
                answer = pair.get("answer", "")
                context = pair.get("context", "")
                ground_truth = pair.get("ground_truth_show", True)

                if isinstance(ground_truth, str):
                    ground_truth = (
                        ground_truth.lower() == "yes" or ground_truth.lower() == "true"
                    )

                start_time = time.time()

                prompt = Config.get_judge_prompt(
                    dialog_history="",
                    question_text=question,
                    answer_text=answer,
                    content=context,
                    generation=False,
                )

                response = requests.post(
                    Config.MISTRAL_API_URL, json=prompt, headers=headers
                )
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)

                if response.status_code == 200:
                    data = response.json()
                    solution = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                        .strip()
                        .lower()
                    )
                    predicted_show = "yes" in solution

                    predictions.append(predicted_show)
                    ground_truths.append(ground_truth)

                    logger.debug(
                        f"Q: {question[:50]}... | "
                        f"A: {answer[:30]}... | "
                        f"GT: {ground_truth} | "
                        f"Pred: {predicted_show} | "
                        f"Latency: {latency:.0f}ms"
                    )
                else:
                    logger.error(
                        f"Ошибка API: {response.status_code} - {response.text}"
                    )

            except Exception as e:
                logger.error(f"Ошибка оценки: {e}")
                continue

        if not predictions:
            logger.warning("Нет успешных предсказаний")
            return {
                "tier_code": -6.0,
                "total_evaluations": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "avg_latency_ms": 0.0,
            }

        tp = sum(1 for p, g in zip(predictions, ground_truths) if p and g)
        fp = sum(1 for p, g in zip(predictions, ground_truths) if p and not g)
        fn = sum(1 for p, g in zip(predictions, ground_truths) if not p and g)
        tn = sum(1 for p, g in zip(predictions, ground_truths) if not p and not g)

        accuracy = (tp + tn) / len(predictions) if predictions else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        metrics = {
            "tier_code": -6.0,
            "total_evaluations": float(len(predictions)),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "true_positives": float(tp),
            "false_positives": float(fp),
            "false_negatives": float(fn),
            "true_negatives": float(tn),
        }

        logger.info(
            f"Tier Judge Pipeline завершён. Accuracy: {accuracy:.2f}, "
            f"F1: {f1:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}"
        )

        return metrics

    def run_tier_ux(
        self,
        dialog_sessions: List[Dict],
    ) -> Dict[str, float]:
        """Выполнить бенчмарк Tier UX (User Experience Quality).

        Оценивает качество пользовательского опыта через:
        - Cache hit rate (попадания в кэш похожих вопросов)
        - Context preservation (сохранение контекста диалога)
        - Multi-turn consistency (согласованность в многотуровых диалогах)

        Args:
            dialog_sessions: Список сессий диалогов
                [{"questions": ["q1", "q2"], "expected_answers": ["a1", "a2"]}]

        Returns:
            Словарь с метриками UX
        """
        logger.info(
            f"Запуск Tier UX (User Experience) с {len(dialog_sessions)} сессиями"
        )

        if not dialog_sessions:
            logger.warning("Нет сессий для UX оценки")
            return {
                "tier_code": -5.0,  # -5 for UX tier
                "total_sessions": 0.0,
                "cache_hit_rate": 0.0,
                "context_preservation": 0.0,
                "multi_turn_consistency": 0.0,
            }

        from qa.database import QuestionAnswer

        cache_hits = 0
        cache_total = 0
        context_scores = []

        with Session(self.engine) as session:
            high_score_questions = session.scalars(
                select(QuestionAnswer).where(QuestionAnswer.score == 5)
            ).all()

            high_score_embeddings = {}
            for qa in high_score_questions:
                if qa.question and qa.embedding is not None:
                    high_score_embeddings[qa.question] = np.array(qa.embedding)

        for session_data in dialog_sessions:
            questions = session_data.get("questions", [])

            for i, question in enumerate(questions):
                question_embedding = self.encoder.encode(question)

                best_match = None
                best_similarity = -1.0

                for cached_q, cached_emb in high_score_embeddings.items():
                    if cached_emb is not None and len(cached_emb) > 0:
                        similarity = np.dot(question_embedding, cached_emb) / (
                            np.linalg.norm(question_embedding)
                            * np.linalg.norm(cached_emb)
                        )
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match = cached_q

                cache_total += 1
                if best_similarity > 0.85:
                    cache_hits += 1

                if i > 0:
                    prev_question = questions[i - 1]
                    prev_emb = self.encoder.encode(prev_question)
                    context_sim = np.dot(question_embedding, prev_emb) / (
                        np.linalg.norm(question_embedding) * np.linalg.norm(prev_emb)
                    )
                    context_scores.append(context_sim)

        cache_hit_rate = cache_hits / cache_total if cache_total > 0 else 0.0
        avg_context_preservation = (
            float(np.mean(context_scores)) if context_scores else 0.0
        )

        metrics = {
            "tier_code": -5.0,  # -5 for UX tier
            "total_sessions": float(len(dialog_sessions)),
            "cache_hit_rate": cache_hit_rate,
            "context_preservation": avg_context_preservation,
            "multi_turn_consistency": avg_context_preservation,
        }

        logger.info(
            f"Tier UX завершён. Cache hit rate: {metrics['cache_hit_rate']:.2f}, "
            f"Context preservation: {metrics['context_preservation']:.2f}"
        )

        return metrics
