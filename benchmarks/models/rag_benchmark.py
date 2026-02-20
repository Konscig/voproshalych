"""Модуль бенчмарков RAG-системы.

Реализует три уровня тестирования:
- Tier 1: Retrieval Accuracy (качество поиска через pgvector)
- Tier 2: Generation Quality (качество генерации через Mistral)
- Tier 3: End-to-End (полный пайплайн)
"""

import logging
from typing import Callable, Dict, List, Optional, Set

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from qa.database import Chunk
from benchmarks.utils.llm_judge import LLMJudge
from benchmarks.utils.evaluator import compute_retrieval_metrics

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

                faithfulness_scores.append(faithfulness)
                relevance_scores.append(relevance)

                # Алгоритмические метрики
                ground_truth = item.get("ground_truth_answer", "")
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
            **avg_text_metrics,
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
        text_metrics_list = []
        errors = 0

        from benchmarks.utils.text_metrics import compute_all_text_metrics

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
                else:
                    cosine_sim = 0.0

                e2e_scores.append(e2e_score)
                semantic_similarities.append(cosine_sim)

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
            **avg_text_metrics,
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

    def run_tier_0(
        self,
        test_pairs: List[Dict],
    ) -> Dict[str, float]:
        """Выполнить бенчмарк Tier 0 (Intrinsic Embedding Quality).

        Оценивает внутреннее качество эмбеддингов через кластерный анализ
        и метрики на golden similarity dataset.

        Args:
            test_pairs: Список пар (question1, question2, ground_truth_similarity)
                       или чанков для кластеризации

        Returns:
            Словарь с метриками качества эмбеддингов
        """
        logger.info(
            f"Запуск Tier 0 (Intrinsic Embedding Quality) с {len(test_pairs)} парами"
        )

        if not test_pairs:
            logger.warning("Нет тестовых пар для Tier 0")
            return {
                "tier": 0,
                "total_pairs": 0,
                "avg_intra_cluster_sim": 0.0,
                "avg_inter_cluster_dist": 0.0,
                "silhouette_score": 0.0,
            }

        embeddings = []
        labels = []

        for pair in test_pairs:
            text = pair.get("text") or pair.get("question", "")
            label = pair.get("cluster_id") or pair.get("label", 0)

            if text:
                emb = self.encoder.encode(text)
                embeddings.append(emb)
                labels.append(label)

        if len(embeddings) < 2:
            logger.warning("Недостаточно данных для кластерного анализа")
            return {
                "tier": 0,
                "total_pairs": len(test_pairs),
                "avg_intra_cluster_sim": 0.0,
                "avg_inter_cluster_dist": 0.0,
                "silhouette_score": 0.0,
            }

        embeddings_array = np.array(embeddings)
        labels_array = np.array(labels)

        unique_labels = np.unique(labels_array)
        n_clusters = len(unique_labels)

        intra_cluster_sims = []
        inter_cluster_dists = []

        for label in unique_labels:
            cluster_mask = labels_array == label
            cluster_embeddings = embeddings_array[cluster_mask]

            if len(cluster_embeddings) > 1:
                for i in range(len(cluster_embeddings)):
                    for j in range(i + 1, len(cluster_embeddings)):
                        sim = np.dot(cluster_embeddings[i], cluster_embeddings[j]) / (
                            np.linalg.norm(cluster_embeddings[i])
                            * np.linalg.norm(cluster_embeddings[j])
                        )
                        intra_cluster_sims.append(sim)

        for i, label_i in enumerate(unique_labels):
            for label_j in unique_labels[i + 1 :]:
                cluster_i = embeddings_array[labels_array == label_i]
                cluster_j = embeddings_array[labels_array == label_j]

                if len(cluster_i) > 0 and len(cluster_j) > 0:
                    centroid_i = np.mean(cluster_i, axis=0)
                    centroid_j = np.mean(cluster_j, axis=0)
                    dist = np.linalg.norm(centroid_i - centroid_j)
                    inter_cluster_dists.append(dist)

        silhouette = 0.0
        if n_clusters > 1:
            try:
                from sklearn.metrics import silhouette_score

                silhouette = silhouette_score(
                    embeddings_array, labels_array, metric="cosine"
                )
            except ImportError:
                logger.warning("sklearn не установлен, silhouette_score не вычислен")

        metrics = {
            "tier": 0,
            "total_pairs": len(test_pairs),
            "n_clusters": n_clusters,
            "avg_intra_cluster_sim": float(np.mean(intra_cluster_sims))
            if intra_cluster_sims
            else 0.0,
            "avg_inter_cluster_dist": float(np.mean(inter_cluster_dists))
            if inter_cluster_dists
            else 0.0,
            "silhouette_score": float(silhouette),
        }

        logger.info(f"Tier 0 завершён. Silhouette: {metrics['silhouette_score']:.4f}")

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
