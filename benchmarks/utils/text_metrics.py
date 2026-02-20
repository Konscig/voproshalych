"""Алгоритмические метрики для оценки качества текста.

Предоставляет функции для вычисления различных метрик качества генерации:
- ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- BLEU (Bilingual Evaluation Understudy)
- BERTScore (семантическое сходство через BERT)
- Лексическое перекрытие n-грамм
"""

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _get_rouge_scorer():
    """Получить ROUGE scorer с ленивым импортом."""
    try:
        from rouge_score import rouge_scorer

        return rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
    except ImportError as e:
        logger.warning("rouge-score не установлен. Установите: pip install rouge-score")
        raise e


def _get_sacrebleu():
    """Получить sacrebleu с ленивым импортом."""
    try:
        import sacrebleu

        return sacrebleu
    except ImportError as e:
        logger.warning("sacrebleu не установлен. Установите: pip install sacrebleu")
        raise e


def _get_bert_score():
    """Получить bert_score с ленивым импортом."""
    try:
        import bert_score

        return bert_score
    except ImportError as e:
        logger.warning("bert-score не установлен. Установите: pip install bert-score")
        raise e


def compute_rouge(hypothesis: str, reference: str) -> Dict[str, float]:
    """Вычислить ROUGE метрики.

    Args:
        hypothesis: Сгенерированный ответ.
        reference: Эталонный ответ.

    Returns:
        Словарь с rouge1_f, rouge2_f, rougeL_f (fmeasure).
    """
    scorer = _get_rouge_scorer()
    scores = scorer.score(reference, hypothesis)

    return {
        "rouge1_f": float(scores["rouge1"].fmeasure),
        "rouge2_f": float(scores["rouge2"].fmeasure),
        "rougeL_f": float(scores["rougeL"].fmeasure),
    }


def compute_bleu(hypothesis: str, reference: str) -> float:
    """Вычислить BLEU score.

    Args:
        hypothesis: Сгенерированный ответ.
        reference: Эталонный ответ.

    Returns:
        BLEU score (0-100).
    """
    sacrebleu = _get_sacrebleu()
    bleu = sacrebleu.sentence_bleu(hypothesis, [reference])
    return float(bleu.score)


def compute_bertscore(
    hypotheses: List[str],
    references: List[str],
    lang: str = "ru",
) -> Dict[str, float]:
    """Вычислить BERTScore.

    Args:
        hypotheses: Список сгенерированных ответов.
        references: Список эталонных ответов.
        lang: Язык (ru, en).

    Returns:
        Словарь с bertscore_precision, bertscore_recall, bertscore_f1.
    """
    bert_score = _get_bert_score()
    P, R, F1 = bert_score.score(
        hypotheses,
        references,
        lang=lang,
        verbose=False,
    )

    return {
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "bertscore_f1": float(F1.mean()),
    }


def compute_lexical_overlap(
    hypothesis: str,
    reference: str,
    n: int = 2,
) -> Dict[str, float]:
    """Вычислить лексическое перекрытие n-грамм.

    Args:
        hypothesis: Сгенерированный ответ.
        reference: Эталонный ответ.
        n: Размер n-грамм.

    Returns:
        Словарь с ngram_precision, ngram_recall, ngram_f1.
    """

    def get_ngrams(text: str, n: int) -> set:
        tokens = text.lower().split()
        return set(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

    hyp_ngrams = get_ngrams(hypothesis, n)
    ref_ngrams = get_ngrams(reference, n)

    overlap = hyp_ngrams & ref_ngrams

    precision = len(overlap) / len(hyp_ngrams) if hyp_ngrams else 0.0
    recall = len(overlap) / len(ref_ngrams) if ref_ngrams else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        f"{n}gram_precision": precision,
        f"{n}gram_recall": recall,
        f"{n}gram_f1": f1,
    }


def compute_all_text_metrics(
    hypothesis: str,
    reference: str,
    include_bertscore: bool = False,
) -> Dict[str, float]:
    """Вычислить все доступные текстовые метрики.

    Args:
        hypothesis: Сгенерированный ответ.
        reference: Эталонный ответ.
        include_bertscore: Включить BERTScore (требует модель BERT).

    Returns:
        Словарь со всеми метриками.
    """
    metrics = {}

    # ROUGE
    try:
        metrics.update(compute_rouge(hypothesis, reference))
    except Exception as e:
        logger.warning(f"ROUGE метрики недоступны: {e}")
        metrics["rouge1_f"] = 0.0
        metrics["rouge2_f"] = 0.0
        metrics["rougeL_f"] = 0.0

    # BLEU
    try:
        metrics["bleu"] = compute_bleu(hypothesis, reference)
    except Exception as e:
        logger.warning(f"BLEU метрика недоступна: {e}")
        metrics["bleu"] = 0.0

    # Lexical overlap
    try:
        metrics.update(compute_lexical_overlap(hypothesis, reference, n=1))
        metrics.update(compute_lexical_overlap(hypothesis, reference, n=2))
    except Exception as e:
        logger.warning(f"Lexical overlap метрики недоступны: {e}")
        metrics["1gram_precision"] = 0.0
        metrics["1gram_recall"] = 0.0
        metrics["1gram_f1"] = 0.0
        metrics["2gram_precision"] = 0.0
        metrics["2gram_recall"] = 0.0
        metrics["2gram_f1"] = 0.0

    # BERTScore (опционально, требует модель)
    if include_bertscore:
        try:
            bert_metrics = compute_bertscore([hypothesis], [reference])
            metrics.update(bert_metrics)
        except Exception as e:
            logger.warning(f"BERTScore метрики недоступны: {e}")
            metrics["bertscore_precision"] = 0.0
            metrics["bertscore_recall"] = 0.0
            metrics["bertscore_f1"] = 0.0

    return metrics


def aggregate_text_metrics(
    metrics_list: List[Dict[str, float]],
) -> Dict[str, float]:
    """Агрегировать список текстовых метрик (среднее).

    Args:
        metrics_list: Список словарей с метриками.

    Returns:
        Словарь со средними значениями метрик.
    """
    if not metrics_list:
        return {}

    aggregated = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list if key in m]
        if values:
            aggregated[f"avg_{key}"] = float(np.mean(values))

    return aggregated
