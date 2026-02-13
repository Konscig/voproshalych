"""Модуль бенчмарков поиска (Retrieval Benchmarks).

Оценивает качество векторного поиска по эмбеддингам:
- Tier 1: Поиск похожих вопросов из question_answer
- Tier 2: Поиск чанков из Confluence базы знаний
"""

from typing import Dict, List, Set
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from sentence_transformers import SentenceTransformer

from qa.database import Chunk, QuestionAnswer
from benchmarks.utils.evaluator import compute_retrieval_metrics


class RetrievalBenchmark:
    """Бенчмарк для оценки качества поиска.

    Attributes:
        engine: Движок базы данных SQLAlchemy
        encoder: Модель для генерации эмбеддингов
    """

    def __init__(self, engine: Engine, encoder: SentenceTransformer):
        """Инициализировать бенчмарк поиска.

        Args:
            engine: Движок базы данных
            encoder: Модель для генерации эмбеддингов
        """
        self.engine = engine
        self.encoder = encoder

    def run_tier_1(
        self,
        test_questions: List[str],
        ground_truth_urls: Dict[str, Set[str]],
        top_k: int = 10,
    ) -> Dict[str, float]:
        """Выполнить бенчмарк Tier 1 (поиск похожих вопросов).

        Тестирует поиск похожих вопросов из таблицы question_answer.

        Args:
            test_questions: Список тестовых вопросов
            ground_truth_urls: Соответствие вопрос -> релевантные URL
            top_k: Количество результатов для поиска

        Returns:
            Словарь с вычисленными метриками
        """
        # Получаем все вопросы с эмбеддингами
        with Session(self.engine) as session:
            questions_with_embeddings = session.scalars(
                select(QuestionAnswer).where(QuestionAnswer.embedding.is_not(None))
            ).all()

            # Формируем список вопросов с эмбеддингами
            qa_list = []
            for qa in questions_with_embeddings:
                if qa.embedding is not None and len(qa.embedding) > 0:
                    qa_list.append(
                        {
                            "id": qa.id,
                            "question": qa.question,
                            "answer": qa.answer,
                            "url": qa.confluence_url,
                            "embedding": qa.embedding,
                        }
                    )

        # Для каждого тестового вопроса ищем похожие
        retrieved_results = {}
        for question in test_questions:
            question_embedding = self.encoder.encode(question)
            similar = self._find_similar_questions_tier_1(
                question_embedding, qa_list, top_k
            )
            retrieved_urls = [item["url"] for item in similar if item.get("url")]
            retrieved_results[question] = retrieved_urls

        # Вычисляем метрики
        return compute_retrieval_metrics(
            test_questions=test_questions,
            retrieved_results=retrieved_results,
            ground_truth_urls=ground_truth_urls,
            k_values=[1, 3, 5, 10],
        )

    def run_tier_2(
        self,
        test_questions: List[str],
        ground_truth_urls: Dict[str, Set[str]],
        top_k: int = 10,
    ) -> Dict[str, float]:
        """Выполнить бенчмарк Tier 2 (поиск чанков).

        Тестирует поиск чанков из базы знаний Confluence.

        Args:
            test_questions: Список тестовых вопросов
            ground_truth_urls: Соответствие вопрос -> релевантные URL
            top_k: Количество результатов для поиска

        Returns:
            Словарь с вычисленными метриками
        """
        # Получаем все чанки с эмбеддингами
        with Session(self.engine) as session:
            chunks = session.scalars(select(Chunk)).all()

            # Формируем список чанков с эмбеддингами
            chunk_list = []
            for chunk in chunks:
                if chunk.embedding is not None and len(chunk.embedding) > 0:
                    chunk_list.append(
                        {
                            "id": chunk.id,
                            "text": chunk.text,
                            "url": chunk.confluence_url,
                            "embedding": chunk.embedding,
                        }
                    )

        # Для каждого тестового вопроса ищем похожие чанки
        retrieved_results = {}
        for question in test_questions:
            question_embedding = self.encoder.encode(question)
            similar = self._find_similar_chunks_tier_2(
                question_embedding, chunk_list, top_k
            )
            retrieved_urls = [item["url"] for item in similar]
            retrieved_results[question] = retrieved_urls

        # Вычисляем метрики
        return compute_retrieval_metrics(
            test_questions=test_questions,
            retrieved_results=retrieved_results,
            ground_truth_urls=ground_truth_urls,
            k_values=[1, 3, 5, 10],
        )

    def _find_similar_questions_tier_1(
        self,
        question_embedding,
        qa_list: List[Dict],
        top_k: int = 10,
    ) -> List[Dict]:
        """Найти похожие вопросы (Tier 1).

        Args:
            question_embedding: Эмбеддинг тестового вопроса
            qa_list: Список вопросов с эмбеддингами из БД
            top_k: Количество результатов

        Returns:
            Список похожих вопросов, отсортированных по расстоянию
        """
        similarities = []

        for qa in qa_list:
            qa_embedding = qa["embedding"]

            # Проверяем размерность
            if question_embedding.shape[0] != qa_embedding.shape[0]:
                continue

            # Вычисляем косинусное расстояние
            cosine_distance = 1.0 - self._cosine_similarity(
                question_embedding, qa_embedding
            )

            similarities.append(
                {
                    "id": qa["id"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "url": qa["url"],
                    "distance": cosine_distance,
                }
            )

        # Сортируем по расстоянию и возвращаем top-K
        similarities.sort(key=lambda x: x["distance"])
        return similarities[:top_k]

    def _find_similar_chunks_tier_2(
        self,
        question_embedding,
        chunk_list: List[Dict],
        top_k: int = 10,
    ) -> List[Dict]:
        """Найти похожие чанки (Tier 2).

        Args:
            question_embedding: Эмбеддинг тестового вопроса
            chunk_list: Список чанков с эмбеддингами из БД
            top_k: Количество результатов

        Returns:
            Список похожих чанков, отсортированных по расстоянию
        """
        similarities = []

        for chunk in chunk_list:
            chunk_embedding = chunk["embedding"]

            # Проверяем размерность
            if question_embedding.shape[0] != chunk_embedding.shape[0]:
                continue

            # Вычисляем косинусное расстояние
            cosine_distance = 1.0 - self._cosine_similarity(
                question_embedding, chunk_embedding
            )

            similarities.append(
                {
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "url": chunk["url"],
                    "distance": cosine_distance,
                }
            )

        # Сортируем по расстоянию и возвращаем top-K
        similarities.sort(key=lambda x: x["distance"])
        return similarities[:top_k]

    def _cosine_similarity(self, vec1, vec2) -> float:
        """Вычислить косинусное сходство между векторами.

        Args:
            vec1: Первый вектор
            vec2: Второй вектор

        Returns:
            Косинусное сходство (0-1)
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm_a = sum(a * a for a in vec1) ** 0.5
        norm_b = sum(b * b for b in vec2) ** 0.5

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
