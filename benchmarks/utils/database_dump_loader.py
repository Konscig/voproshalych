"""Модуль для автоматической загрузки дампа БД.

Подгружает дамп базы данных PostgreSQL и инициализирует
таблицы для работы системы бенчарков.
"""

import io
import logging
import gzip
import tarfile
from typing import Optional

import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logger = logging.getLogger(__name__)


class DatabaseDumpLoader:
    """Загрузчик дампа БД.

    Загружает дамп PostgreSQL и инициализирует таблицы.

    Attributes:
        engine: Движок SQLAlchemy
        Session: Фабрика сессий
    """

    def __init__(self, database_url: str, dump_path: str):
        """Инициализировать загрузчик дампа.

        Args:
            database_url: URL подключения к PostgreSQL
            dump_path: Путь к файлу дампа или директории
        """
        self.database_url = database_url
        self.dump_path = dump_path
        self.engine = None
        self.Session = None

    def connect(self):
        """Подключиться к базе данных."""
        try:
            self.engine = create_engine(self.database_url)
            self.Session = sessionmaker(bind=self.engine)
            logger.info(f"Подключено к базе данных: {self.database_url}")
        except Exception as e:
            logger.error(f"Ошибка подключения к БД: {e}")
            raise

    def load_dump(self) -> bool:
        """Загрузить дамп БД.

        Returns:
            True если дамп загружен успешно, иначе False
        """
        try:
            if os.path.isfile(self.dump_path):
                return self._load_dump_file()
            elif os.path.isdir(self.dump_path):
                return self._load_dump_directory()
            else:
                logger.error(f"Дамп не найден: {self.dump_path}")
                return False
        except Exception as e:
            logger.error(f"Ошибка загрузки дампа: {e}")
            return False

    def _load_dump_file(self) -> bool:
        """Загрузить дамп из файла.

        Returns:
            True если дамп загружен успешно
        """
        if self.dump_path.endswith(".gz"):
            return self._load_gz_dump()
        elif self.dump_path.endswith(".sql"):
            return self._load_sql_dump()
        elif self.dump_path.endswith(".tar") or self.dump_path.endswith(".tar.gz"):
            return self._load_tar_dump()
        else:
            logger.error(f"Неизвестный формат дампа: {self.dump_path}")
            return False

    def _load_gz_dump(self) -> bool:
        """Загрузить gzip дамп."""
        try:
            logger.info(f"Загрузка gzip дампа: {self.dump_path}")

            with gzip.open(self.dump_path, "rt", encoding="utf-8") as f:
                sql_content = f.read()

            with self.Session() as session:
                connection = session.connection()
                try:
                    connection.connection.execute(text(sql_content))
                    connection.commit()
                    logger.info("Gzip дамп загружен успешно")
                    return True
                finally:
                    connection.close()
        except Exception as e:
            logger.error(f"Ошибка загрузки gzip дампа: {e}")
            return False

    def _load_sql_dump(self) -> bool:
        """Загрузить SQL дамп."""
        try:
            logger.info(f"Загрузка SQL дампа: {self.dump_path}")

            with open(self.dump_path, "r", encoding="utf-8") as f:
                sql_content = f.read()

            with self.Session() as session:
                connection = session.connection()
                try:
                    connection.connection.execute(text(sql_content))
                    connection.commit()
                    logger.info("SQL дамп загружен успешно")
                    return True
                finally:
                    connection.close()
        except Exception as e:
            logger.error(f"Ошибка загрузки SQL дампа: {e}")
            return False

    def _load_tar_dump(self) -> bool:
        """Загрузить tar дамп.

        Returns:
            True если дамп загружен успешно
        """
        try:
            logger.info(f"Загрузка tar дампа: {self.dump_path}")

            if self.dump_path.endswith(".tar.gz"):
                tar = tarfile.open(self.dump_path, "r:gz")
            else:
                logger.error(f"Неизвестный формат дампа: {self.dump_path}")
                return False

            with self.Session() as session:
                connection = session.connection()
                try:
                    # Получаем список файлов в архиве
                    members = tar.getmembers()

                    # Фильтруем только SQL файлы
                    sql_files = [m for m in members if m.name.endswith(".sql")]
                    logger.info(f"Найдено {len(sql_files)} SQL файлов в архиве")

                    # Для каждого SQL файла
                    for member in sql_files:
                        logger.info(f"Загрузка файла: {member.name}")

                        if member.name.endswith(".gz"):
                            # Файл сжатый gzip
                            with gzip.open(member, "rt", encoding="utf-8") as f:
                                sql_content = f.read()

                            connection.connection.execute(text(sql_content))

                            logger.info("Gzip дамп загружен успешно")
                            return True
                        else:
                            # Обычный SQL файл
                            with open(member, "r", encoding="utf-8") as f:
                                sql_content = f.read()

                                connection.connection.execute(text(sql_content))

                                logger.info("SQL дамп загружен успешно")
                                return True
                except Exception as e:
                    logger.error(f"Ошибка загрузки дампа из архива: {e}")
                    return False
        except Exception as e:
            logger.error(f"Ошибка загрузки дампа: {e}")
            return False

    def _load_dump_directory(self) -> bool:
        """Загрузить дамп из директории.

        Returns:
            True если дамп загружен успешно, иначе False
        """
        try:
            logger.info(f"Загрузка дампа из директории: {self.dump_path}")

            if not os.path.isdir(self.dump_path):
                logger.error(f"Дамп не найден: {self.dump_path}")
                return False

            # Проверяем, что это директория, а не обычный файл
            if os.path.isfile(self.dump_path):
                return self._load_dump_file()

            # Это директория с архивом
            sql_files = []

            # Рекурсивно ищем SQL файлы с расширением .sql
            for filename in os.listdir(self.dump_path):
                if filename.endswith(".sql"):
                    sql_files.append(os.path.join(self.dump_path, filename))
                elif filename.endswith(".gz") and filename.endswith(".sql"):
                    sql_files.append(os.path.join(self.dump_path, filename))

            if not sql_files:
                logger.error("SQL файлы не найдены в директории дампа")
                return False

            logger.info(f"Найдено {len(sql_files)} SQL файлов в директории дампа")

            with self.Session() as session:
                connection = session.connection()
                try:
                    connection.connection.execute(text("SET client_encoding to 'UTF8'"))

                    # Для каждого SQL файла
                    for sql_file in sql_files:
                        logger.info(f"Загрузка файла: {sql_file}")

                        if sql_file.endswith(".gz"):
                            with gzip.open(sql_file, "rt", encoding="utf-8") as f:
                                sql_content = f.read()
                        else:
                            with open(sql_file, "r", encoding="utf-8") as f:
                                sql_content = f.read()

                        connection.connection.execute(text(sql_content))

                    connection.commit()
                    logger.info(
                        f"Загружено {len(sql_files)} файлов из директории дампа"
                    )
                    return True
                finally:
                    connection.close()
        except Exception as e:
            logger.error(f"Ошибка загрузки дампа из директории: {e}")
            return False

            with self.Session() as session:
                connection = session.connection()
                try:
                    for sql_file in sql_files:
                        logger.info(f"Загрузка файла: {sql_file}")
                        if sql_file.endswith(".gz"):
                            with gzip.open(sql_file, "rt", encoding="utf-8") as f:
                                sql_content = f.read()
                        else:
                            with open(sql_file, "r", encoding="utf-8") as f:
                                sql_content = f.read()

                        connection.connection.execute(text(sql_content))

                    connection.commit()
                    logger.info(f"Загружено {len(sql_files)} файлов из директории")
                    return True
                finally:
                    connection.close()
        except Exception as e:
            logger.error(f"Ошибка загрузки дампа из директории: {e}")
            return False

    def check_tables(self) -> dict:
        """Проверить наличие таблиц в БД.

        Returns:
            Словарь с статистикой по таблицам
        """
        try:
            with self.Session() as session:
                # Проверяем question_answer
                qa_count = session.execute(
                    text("SELECT COUNT(*) FROM question_answer")
                ).scalar()

                # Проверяем chunk
                chunk_count = session.execute(
                    text("SELECT COUNT(*) FROM chunk")
                ).scalar()

                # Проверяем наличие эмбеддингов
                qa_embeddings_count = session.execute(
                    text(
                        "SELECT COUNT(*) FROM question_answer WHERE embedding IS NOT NULL"
                    )
                ).scalar()

                chunk_embeddings_count = session.execute(
                    text("SELECT COUNT(*) FROM chunk WHERE embedding IS NOT NULL")
                ).scalar()

                stats = {
                    "question_answer_total": qa_count,
                    "chunk_total": chunk_count,
                    "qa_embeddings": qa_embeddings_count,
                    "chunk_embeddings": chunk_embeddings_count,
                    "qa_coverage_percent": qa_embeddings_count / qa_count * 100
                    if qa_count > 0
                    else 0,
                    "chunk_coverage_percent": chunk_embeddings_count / chunk_count * 100
                    if chunk_count > 0
                    else 0,
                }

                logger.info(f"Статистика таблиц: {stats}")
                return stats
        except Exception as e:
            logger.error(f"Ошибка проверки таблиц: {e}")
            return {"error": str(e)}

    def close(self):
        """Закрыть подключение к БД."""
        if self.engine:
            self.engine.dispose()
            logger.info("Подключение к БД закрыто")
