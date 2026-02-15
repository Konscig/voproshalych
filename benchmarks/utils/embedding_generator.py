"""Модуль для генерации эмбеддингов.

Генерирует эмбеддинги для вопросов и чанков в базе данных.
Использует модель multilingual-e5-large-wikiutmn.
"""

import logging
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import Engine, select, update
from sqlalchemy.orm import Session

from qa.database import Chunk, QuestionAnswer, set_embedding

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Генератор эмбеддингов для вопросов и чанков.

    Attributes:
        engine: Движок базы данных SQLAlchemy
        encoder: Модель для генерации эмбеддингов
    """

    def __init__(self, engine: Engine, encoder: SentenceTransformer):
        """Инициализировать генератор эмбеддингов.

        Args:
            engine: Движок базы данных
            encoder: Модель для генерации эмбеддингов
        """
        self.engine = engine
        self.encoder = encoder

    def generate_for_all_questions(
        self,
        score_filter: Optional[int] = None,
        overwrite: bool = False,
    ) -> dict:
        """Генерировать эмбеддинги для всех вопросов.

        Args:
            score_filter: Фильтр по оценке (опционально)
            overwrite: Перезаписывать существующие эмбеддинги

        Returns:
            Словарь со статистикой генерации
        """
        stats = {
            "total_questions": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0,
        }

        # Получаем вопросы из БД
        with Session(self.engine) as session:
            query = select(QuestionAnswer)

            if score_filter is not None:
                query = query.where(QuestionAnswer.score == score_filter)

            questions = session.scalars(query).all()

            stats["total_questions"] = len(questions)

            logger.info(f"Начинаем генерацию эмбеддингов для {len(questions)} вопросов")

            for i, qa in enumerate(questions, 1):
                try:
                    # Пропускаем если эмбеддинг уже есть и не перезаписываем
                    if not overwrite and qa.embedding is not None:
                        stats["skipped"] += 1
                        continue

                    # Генерируем эмбеддинг
                    if qa.question:
                        embedding = self.encoder.encode(qa.question)

                        # Сохраняем в БД
                        set_embedding(self.engine, qa.id, embedding)

                        stats["processed"] += 1

                        if i % 100 == 0:
                            logger.info(f"Обработано {i}/{len(questions)} вопросов")
                    else:
                        stats["skipped"] += 1
                        logger.warning(f"Вопрос {qa.id} не содержит текста, пропускаем")

                except Exception as e:
                    stats["errors"] += 1
                    logger.error(
                        f"Ошибка при генерации эмбеддинга для вопроса {qa.id}: {e}"
                    )

        logger.info(
            f"Генерация завершена. "
            f"Обработано: {stats['processed']}, "
            f"Пропущено: {stats['skipped']}, "
            f"Ошибок: {stats['errors']}"
        )

        return stats

    def generate_for_score_5(self, overwrite: bool = False) -> dict:
        """Генерировать эмбеддинги для вопросов с оценкой 5.

        Args:
            overwrite: Перезаписывать существующие эмбеддинги

        Returns:
            Словарь со статистикой генерации
        """
        logger.info("Генерация эмбеддингов для score=5 вопросов")
        return self.generate_for_all_questions(score_filter=5, overwrite=overwrite)

    def generate_for_score_1(self, overwrite: bool = False) -> dict:
        """Генерировать эмбеддинги для вопросов с оценкой 1.

        Args:
            overwrite: Перезаписывать существующие эмбеддинги

        Returns:
            Словарь со статистикой генерации
        """
        logger.info("Генерация эмбеддингов для score=1 вопросов")
        return self.generate_for_all_questions(score_filter=1, overwrite=overwrite)

    def check_coverage(self) -> dict:
        """Проверить покрытие эмбеддингами в базе данных.

        Returns:
            Словарь со статистикой покрытия
        """
        stats = {
            "total_questions": 0,
            "with_embeddings": 0,
            "without_embeddings": 0,
            "coverage_percent": 0.0,
        }

        with Session(self.engine) as session:
            questions = session.scalars(select(QuestionAnswer)).all()
            stats["total_questions"] = len(questions)

            for qa in questions:
                if qa.embedding is not None and len(qa.embedding) > 0:
                    stats["with_embeddings"] += 1
                else:
                    stats["without_embeddings"] += 1

            if stats["total_questions"] > 0:
                stats["coverage_percent"] = (
                    stats["with_embeddings"] / stats["total_questions"] * 100
                )

        logger.info(f"Покрытие эмбеддингами: {stats['coverage_percent']:.2f}%")

        return stats

    def generate_for_chunks(self, overwrite: bool = False) -> dict:
        """Генерировать эмбеддинги для всех чанков.

        Args:
            overwrite: Перезаписывать существующие эмбеддинги

        Returns:
            Словарь со статистикой генерации
        """
        stats = {
            "total_chunks": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0,
        }

        with Session(self.engine) as session:
            chunks = session.scalars(select(Chunk)).all()
            stats["total_chunks"] = len(chunks)

            logger.info(f"Начинаем генерацию эмбеддингов для {len(chunks)} чанков")

            for i, chunk in enumerate(chunks, 1):
                try:
                    if not overwrite and chunk.embedding is not None:
                        stats["skipped"] += 1
                        continue

                    if chunk.text:
                        embedding = self.encoder.encode(chunk.text)

                        session.execute(
                            update(Chunk)
                            .where(Chunk.id == chunk.id)
                            .values(embedding=embedding)
                        )
                        session.commit()

                        stats["processed"] += 1

                        if i % 100 == 0:
                            logger.info(f"Обработано {i}/{len(chunks)} чанков")
                    else:
                        stats["skipped"] += 1
                        logger.warning(
                            f"Чанк {chunk.id} не содержит текста, пропускаем"
                        )

                except Exception as e:
                    stats["errors"] += 1
                    logger.error(
                        f"Ошибка при генерации эмбеддинга для чанка {chunk.id}: {e}"
                    )
                    session.rollback()

        logger.info(
            f"Генерация для чанков завершена. "
            f"Обработано: {stats['processed']}, "
            f"Пропущено: {stats['skipped']}, "
            f"Ошибок: {stats['errors']}"
        )

        return stats

    def check_chunks_coverage(self) -> dict:
        """Проверить покрытие эмбеддингами чанков.

        Returns:
            Словарь со статистикой покрытия чанков
        """
        stats = {
            "total_chunks": 0,
            "with_embeddings": 0,
            "without_embeddings": 0,
            "coverage_percent": 0.0,
        }

        with Session(self.engine) as session:
            chunks = session.scalars(select(Chunk)).all()
            stats["total_chunks"] = len(chunks)

            for chunk in chunks:
                if chunk.embedding is not None and len(chunk.embedding) > 0:
                    stats["with_embeddings"] += 1
                else:
                    stats["without_embeddings"] += 1

            if stats["total_chunks"] > 0:
                stats["coverage_percent"] = (
                    stats["with_embeddings"] / stats["total_chunks"] * 100
                )

        logger.info(f"Покрытие эмбеддингами чанков: {stats['coverage_percent']:.2f}%")

        return stats
