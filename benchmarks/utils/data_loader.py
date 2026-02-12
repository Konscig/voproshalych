"""Модуль для загрузки данных для бенчмарков.

Загружает тестовые вопросы из JSON файлов или базы данных.
"""

import json
import logging
from typing import Dict, List, Set, Optional

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from qa.database import QuestionAnswer

logger = logging.getLogger(__name__)


class DataLoader:
    """Загрузчик данных для бенчмарков.

    Attributes:
        engine: Движок базы данных SQLAlchemy
    """

    def __init__(self, engine: Engine):
        """Инициализировать загрузчик данных.

        Args:
            engine: Движок базы данных
        """
        self.engine = engine

    def load_from_json(self, file_path: str) -> List[Dict]:
        """Загрузить вопросы из JSON файла.

        Args:
            file_path: Путь к JSON файлу

        Returns:
            Список словарей с вопросами и ответами
        """
        logger.info(f"Загрузка данных из {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.info(f"Загружено {len(data)} записей")
        return data

    def load_golden_set(self) -> List[Dict]:
        """Загрузить золотой набор (score=5) из базы данных.

        Returns:
            Список словарей с вопросами и ответами
        """
        logger.info("Загрузка золотого набора (score=5)")

        with Session(self.engine) as session:
            questions = session.scalars(
                select(QuestionAnswer).where(QuestionAnswer.score == 5)
            ).all()

            data = []
            for qa in questions:
                data.append(
                    {
                        "id": qa.id,
                        "question": qa.question,
                        "answer": qa.answer,
                        "confluence_url": qa.confluence_url,
                        "score": qa.score,
                        "created_at": qa.created_at.isoformat()
                        if qa.created_at
                        else None,
                    }
                )

        logger.info(f"Загружено {len(data)} записей с score=5")
        return data

    def load_low_quality_set(self, limit: Optional[int] = None) -> List[Dict]:
        """Загрузить низкокачественные вопросы (score=1).

        Args:
            limit: Максимальное количество записей

        Returns:
            Список словарей с вопросами и ответами
        """
        logger.info("Загрузка низкокачественных вопросов (score=1)")

        with Session(self.engine) as session:
            query = select(QuestionAnswer).where(QuestionAnswer.score == 1)

            if limit is not None:
                query = query.limit(limit)

            questions = session.scalars(query).all()

            data = []
            for qa in questions:
                data.append(
                    {
                        "id": qa.id,
                        "question": qa.question,
                        "answer": qa.answer,
                        "confluence_url": qa.confluence_url,
                        "score": qa.score,
                        "created_at": qa.created_at.isoformat()
                        if qa.created_at
                        else None,
                    }
                )

        logger.info(f"Загружено {len(data)} записей с score=1")
        return data

    def load_questions_with_url(self) -> List[Dict]:
        """Загрузить вопросы с URL из базы данных.

        Returns:
            Список словарей с вопросами и URL
        """
        logger.info("Загрузка вопросов с URL")

        with Session(self.engine) as session:
            questions = session.scalars(
                select(QuestionAnswer).where(
                    QuestionAnswer.confluence_url.isnot(None),
                    QuestionAnswer.confluence_url != "",
                )
            ).all()

            data = []
            for qa in questions:
                data.append(
                    {
                        "id": qa.id,
                        "question": qa.question,
                        "answer": qa.answer,
                        "confluence_url": qa.confluence_url,
                        "score": qa.score,
                        "created_at": qa.created_at.isoformat()
                        if qa.created_at
                        else None,
                    }
                )

        logger.info(f"Загружено {len(data)} записей с URL")
        return data

    def load_recent_questions(
        self, months: int = 6, limit: Optional[int] = None
    ) -> List[Dict]:
        """Загрузить последние вопросы.

        Args:
            months: Количество месяцев для выборки
            limit: Максимальное количество записей

        Returns:
            Список словарей с вопросами
        """
        from datetime import datetime, timedelta

        logger.info(f"Загрузка последних {months} месяцев вопросов")

        cutoff_date = datetime.now() - timedelta(days=months * 30)

        with Session(self.engine) as session:
            query = select(QuestionAnswer).where(
                QuestionAnswer.created_at >= cutoff_date
            )

            if limit is not None:
                query = query.limit(limit)

            questions = session.scalars(query).all()

            data = []
            for qa in questions:
                data.append(
                    {
                        "id": qa.id,
                        "question": qa.question,
                        "answer": qa.answer,
                        "confluence_url": qa.confluence_url,
                        "score": qa.score,
                        "created_at": qa.created_at.isoformat()
                        if qa.created_at
                        else None,
                    }
                )

        logger.info(f"Загружено {len(data)} последних вопросов")
        return data

    def prepare_ground_truth_urls(self, questions: List[Dict]) -> Dict[str, Set[str]]:
        """Подготовить ground truth URL для вопросов.

        Args:
            questions: Список словарей с вопросами

        Returns:
            Словарь вопрос -> множество релевантных URL
        """
        ground_truth = {}

        for qa in questions:
            question = qa["question"]
            url = qa.get("confluence_url")

            if url:
                if question not in ground_truth:
                    ground_truth[question] = set()

                ground_truth[question].add(url)

        logger.info(f"Подготовлен ground truth для {len(ground_truth)} вопросов")

        return ground_truth
