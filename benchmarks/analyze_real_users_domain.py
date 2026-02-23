"""Анализ предметной области и паттернов real-user вопросов."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
import re
import sys
from typing import Any

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from qa.config import Config
from qa.database import QuestionAnswer, create_engine

logger = logging.getLogger(__name__)


def _top_tokens(texts: list[str], top_n: int = 50) -> list[dict[str, Any]]:
    token_counter = Counter()
    for text in texts:
        tokens = re.findall(r"[а-яА-Яa-zA-Z0-9_-]{3,}", text.lower())
        token_counter.update(tokens)
    return [
        {"token": token, "count": count}
        for token, count in token_counter.most_common(top_n)
    ]


def analyze_real_users_domain(engine: Engine, limit: int = 5000) -> dict[str, Any]:
    """Собрать аналитические метрики предметной области по QuestionAnswer."""
    with Session(engine) as session:
        rows = session.scalars(
            select(QuestionAnswer)
            .where(QuestionAnswer.question.isnot(None), QuestionAnswer.question != "")
            .order_by(QuestionAnswer.id.asc())
            .limit(limit)
        ).all()

    total = len(rows)
    with_answers = [row for row in rows if row.answer and row.answer.strip()]
    without_answers = [row for row in rows if not row.answer or not row.answer.strip()]
    scored = [row for row in rows if row.score is not None]
    score_5 = [row for row in rows if row.score == 5]
    score_1 = [row for row in rows if row.score == 1]

    question_texts = [row.question.strip() for row in rows if row.question]
    with_answer_questions = [
        row.question.strip() for row in with_answers if row.question
    ]
    without_answer_questions = [
        row.question.strip() for row in without_answers if row.question
    ]

    avg_question_length = (
        sum(len(text) for text in question_texts) / len(question_texts)
        if question_texts
        else 0.0
    )

    slang_seed = [
        "пара",
        "сессия",
        "допса",
        "препод",
        "стипуха",
        "пересдача",
        "хвост",
        "общага",
    ]
    slang_stats = {
        word: sum(1 for text in question_texts if word in text.lower())
        for word in slang_seed
    }

    by_score = Counter(row.score for row in rows if row.score is not None)

    return {
        "total_questions": total,
        "with_answers": len(with_answers),
        "without_answers": len(without_answers),
        "with_answers_rate": (len(with_answers) / total) if total else 0.0,
        "scored_questions": len(scored),
        "score_distribution": dict(by_score),
        "score_5_questions": len(score_5),
        "score_1_questions": len(score_1),
        "avg_question_length": avg_question_length,
        "top_tokens_all": _top_tokens(question_texts, top_n=50),
        "top_tokens_with_answers": _top_tokens(with_answer_questions, top_n=30),
        "top_tokens_without_answers": _top_tokens(without_answer_questions, top_n=30),
        "slang_indicator_counts": slang_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Domain analysis real-user вопросов")
    parser.add_argument("--limit", type=int, default=5000, help="Лимит вопросов")
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/reports/real_users_domain_analysis.json",
        help="Путь для JSON отчёта",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
    result = analyze_real_users_domain(engine, limit=args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Сохранён domain analysis: %s", output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
