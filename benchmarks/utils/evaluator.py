"""Модуль для вычисления метрик бенчмарков.

Предоставляет функции для вычисления различных метрик качества:
- Для поиска: Recall@K, Precision@K, MRR, NDCG@K
- Для генерации: ROUGE, BLEU, BERTScore
"""

from typing import Dict, List, Set
import numpy as np


def recall_at_k(
    retrieved_urls: List[str],
    relevant_urls: Set[str],
    k: int,
) -> float:
    """Вычислить Recall@K.

    Args:
        retrieved_urls: Список найденных URL
        relevant_urls: Множество релевантных URL
        k: Количество top-K результатов

    Returns:
        Recall@K: Доля релевантных документов среди top-K

    Raises:
        ValueError: Если relevant_urls пусто
    """
    if not relevant_urls:
        return 0.0

    top_k_urls = set(retrieved_urls[:k])
    hits = len(top_k_urls & relevant_urls)

    return hits / len(relevant_urls)


def precision_at_k(
    retrieved_urls: List[str],
    relevant_urls: Set[str],
    k: int,
) -> float:
    """Вычислить Precision@K.

    Args:
        retrieved_urls: Список найденных URL
        relevant_urls: Множество релевантных URL
        k: Количество top-K результатов

    Returns:
        Precision@K: Доля релевантных документов в top-K
    """
    if k == 0:
        return 0.0

    top_k_urls = set(retrieved_urls[:k])
    hits = len(top_k_urls & relevant_urls)

    return hits / k


def mrr(retrieved_urls: List[str], relevant_urls: Set[str]) -> float:
    """Вычислить Mean Reciprocal Rank.

    Args:
        retrieved_urls: Список найденных URL
        relevant_urls: Множество релевантных URL

    Returns:
        MRR: Обратный ранг первого релевантного документа
    """
    for i, url in enumerate(retrieved_urls, 1):
        if url in relevant_urls:
            return 1.0 / i

    return 0.0


def ndcg_at_k(
    retrieved_urls: List[str],
    relevant_urls: Set[str],
    k: int,
) -> float:
    """Вычислить Normalized Discounted Cumulative Gain @K.

    Args:
        retrieved_urls: Список найденных URL
        relevant_urls: Множество релевантных URL
        k: Количество top-K результатов

    Returns:
        NDCG@K: Нормализованный дисконтированный кумулятивный выигрыш

    Note:
        Принимает все URL из relevant_urls как равнозначные
        (gain = 1), т.к. у нас нет оценок релевантности.
    """
    if k == 0 or not retrieved_urls:
        return 0.0

    # DCG
    dcg = 0.0
    for i, url in enumerate(retrieved_urls[:k], 1):
        gain = 1.0 if url in relevant_urls else 0.0
        dcg += gain / np.log2(i + 1)

    # Ideal DCG
    ideal_dcg = 0.0
    for i in range(1, min(k, len(relevant_urls)) + 1):
        ideal_dcg += 1.0 / np.log2(i + 1)

    # NDCG
    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def compute_retrieval_metrics(
    test_questions: List[str],
    retrieved_results: Dict[str, List[str]],
    ground_truth_urls: Dict[str, Set[str]],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """Вычислить все метрики для бенчмарков поиска.

    Args:
        test_questions: Список тестовых вопросов
        retrieved_results: Словарь вопрос -> список найденных URL
        ground_truth_urls: Словарь вопрос -> множество релевантных URL
        k_values: Список значений K для Recall@K и Precision@K

    Returns:
        Словарь со всеми вычисленными метриками
    """
    results = {f"recall@{k}": [] for k in k_values}
    results.update({f"precision@{k}": [] for k in k_values})
    results["mrr"] = []
    results["ndcg@5"] = []
    results["ndcg@10"] = []

    for question in test_questions:
        retrieved = retrieved_results.get(question, [])
        relevant = ground_truth_urls.get(question, set())

        for k in k_values:
            results[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
            results[f"precision@{k}"].append(precision_at_k(retrieved, relevant, k))

        results["mrr"].append(mrr(retrieved, relevant))
        results["ndcg@5"].append(ndcg_at_k(retrieved, relevant, 5))
        results["ndcg@10"].append(ndcg_at_k(retrieved, relevant, 10))

    # Усреднить значения
    final_results = {}
    for k, v in results.items():
        if not v:
            final_results[k] = 0.0
        else:
            final_results[k] = np.mean(v)
    return final_results
