"""Модуль для загрузки статичных датасетов из текстовых файлов.

Загружает вопросы из текстовых файлов в формате:
id | question | answer | confluence_url | score | created_at | user_id

Генерирует эмбеддинги для вопросов.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class StaticDataLoader:
    """Загрузчик статичных датасетов.

    Загружает вопросы из текстовых файлов в benchmarks/data/static/
    и генерирует эмбеддинги для вопросов.
    """

    def __init__(self, data_dir: Optional[str] = None, encoder=None):
        """Инициализировать загрузчик.

        Args:
            data_dir: Директория со статичными датасетами (по умолчанию benchmarks/data/static/)
            encoder: Модель для генерации эмбеддингов (опционально)
        """
        if data_dir is None:
            project_root = Path(__file__).parent.parent
            data_dir = project_root / "data" / "static"
        else:
            data_dir = Path(data_dir)

        self.data_dir = data_dir
        self.encoder = encoder
        logger.info(f"Директория статичных датасетов: {data_dir}")

    def load_golden_set(self) -> List[Dict]:
        """Загрузить золотой набор (score=5).

        Returns:
            Список словарей с вопросами и эмбеддингами
        """
        file_path = self.data_dir / "03_golden_set_score_5.json.txt"
        return self._load_text_file(file_path)

    def load_low_quality_set(self, limit: int = 200) -> List[Dict]:
        """Загрузить низкое качество (score=1).

        Args:
            limit: Максимум записей для загрузки

        Returns:
            Список словарей с вопросами и эмбеддингами
        """
        file_path = self.data_dir / "04_low_quality_score_1.json.txt"
        return self._load_text_file(file_path, limit=limit)

    def load_questions_with_url(self) -> List[Dict]:
        """Загрузить вопросы с URL.

        Returns:
            Список словарей с вопросами и эмбеддингами
        """
        file_path = self.data_dir / "05_questions_with_url.json.txt"
        return self._load_text_file(file_path)

    def load_recent_questions(self, months: int = 6, limit: int = 1000) -> List[Dict]:
        """Загрузить последние вопросы.

        Args:
            months: Количество месяцев
            limit: Максимум записей для загрузки

        Returns:
            Список словарей с вопросами и эмбеддингами
        """
        file_path = (
            self.data_dir / "10_recent_questions_with_url_last_6_months.json.txt"
        )
        return self._load_text_file(file_path, limit=limit)

    def _load_text_file(
        self, file_path: Path, limit: Optional[int] = None
    ) -> List[Dict]:
        """Загрузить текстовый файл с вопросами.

        Args:
            file_path: Путь к файлу
            limit: Максимум записей для загрузки

        Returns:
            Список словарей с вопросами
        """
        if not file_path.exists():
            logger.error(f"Файл не найден: {file_path}")
            return []

        questions = []
        in_data_section = False

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    if not line:
                        continue

                    if line.startswith("### OUTPUT:"):
                        in_data_section = True
                        continue

                    if not in_data_section:
                        continue

                    if line.startswith("### "):
                        continue

                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) < 2:
                        continue

                    question_text = parts[1]
                    answer_text = parts[2] if len(parts) > 2 else None
                    url = parts[3] if len(parts) > 3 else None
                    score = (
                        int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else None
                    )

                    question_dict = {
                        "question": question_text,
                        "answer": answer_text,
                        "url": url,
                    }

                    questions.append(question_dict)

                    if limit and len(questions) >= limit:
                        break

        except Exception as e:
            logger.error(f"Ошибка чтения файла {file_path}: {e}")
            return []

        logger.info(f"Загружено {len(questions)} вопросов из {file_path.name}")

        questions = self._prepare_questions_with_embeddings(questions)

        return questions

    def _prepare_questions_with_embeddings(self, questions: List[Dict]) -> List[Dict]:
        """Подготовить вопросы с эмбеддингами для бенчмарков.

        Генерирует эмбеддинги для вопросов.

        Args:
            questions: Список вопросов

        Returns:
            Список вопросов с эмбеддингами
        """
        if not self.encoder:
            logger.warning("Энкодер не указан, эмбеддинги не будут генерироваться")
            return questions

        prepared_questions = []

        for qa in questions:
            question_embedding = self.encoder.encode(qa["question"])

            prepared_qa = {
                "id": qa.get("id"),
                "question": qa["question"],
                "answer": qa.get("answer"),
                "url": qa.get("url"),
                "embedding": question_embedding,
            }

            prepared_questions.append(prepared_qa)

        logger.info(f"Сгенерировано {len(prepared_questions)} эмбеддингов")
        return prepared_questions

    def prepare_ground_truth_urls(self, questions: List[Dict]) -> Dict[str, set]:
        """Подготовить ground truth URL.

        Args:
            questions: Список вопросов

        Returns:
            Словарь вопрос -> множество URL
        """
        ground_truth = {}
        for qa in questions:
            question = qa["question"]
            url = qa.get("url")
            if url:
                ground_truth[question] = {url}
            else:
                ground_truth[question] = set()

        logger.info(f"Подготовлен ground truth для {len(ground_truth)} вопросов")
        return ground_truth
