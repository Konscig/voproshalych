"""Модуль для автоматической загрузки дампа БД.

Подгружает дамп базы данных PostgreSQL и инициализирует
таблицы для работы системы бенчарков.
"""

import os
import subprocess
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
        elif self.dump_path.endswith(".dump"):
            return self._load_plain_dump()
        elif self.dump_path.endswith(".tar") or self.dump_path.endswith(".tar.gz"):
            return self._load_tar_dump()
        else:
            logger.error(f"Неизвестный формат дампа: {self.dump_path}")
            return False

    def _load_plain_dump(self) -> bool:
        """Загрузить обычный дамп (без сжатия) через psql.

        Returns:
            True если дамп загружен успешно
        """
        try:
            logger.info(f"Загрузка дампа: {self.dump_path}")

            from sqlalchemy.engine.url import make_url
            
            db_url = make_url(self.database_url)
            host = db_url.host
            port = db_url.port
            user = db_url.username
            password = db_url.password
            database = db_url.database

            env = os.environ.copy()
            env['PGPASSWORD'] = password if password else ''

            command = [
                'psql',
                f'--host={host}',
                f'--port={port}' if port else '',
                f'--username={user}',
                f'--dbname={database}',
                f'--file={self.dump_path}'
            ]
            
            command = [c for c in command if c]
            
            result = subprocess.run(
                command,
                env=env,
                capture_output=True,
                text=True
            )

            if result.returncode == 0:
                logger.info("Дамп загружен успешно")
                return True
            else:
                logger.error(f"Ошибка загрузки дампа: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Ошибка загрузки дампа: {e}")
            return False

    def _load_sql_dump(self) -> bool:
        """Загрузить SQL дамп через psql."""
        try:
            logger.info(f"Загрузка SQL дампа: {self.dump_path}")
            return self._load_plain_dump()
        except Exception as e:
            logger.error(f"Ошибка загрузки SQL дампа: {e}")
            return False

    def _load_gz_dump(self) -> bool:
        """Загрузить gzip дамп."""
        try:
            logger.info(f"Загрузка gzip дампа: {self.dump_path}")

            from sqlalchemy.engine.url import make_url
            
            db_url = make_url(self.database_url)
            host = db_url.host
            port = db_url.port
            user = db_url.username
            password = db_url.password
            database = db_url.database

            env = os.environ.copy()
            env['PGPASSWORD'] = password if password else ''

            command = [
                'psql',
                f'--host={host}',
                f'--port={port}' if port else '',
                f'--username={user}',
                f'--dbname={database}'
            ]
            
            command = [c for c in command if c]
            
            with gzip.open(self.dump_path, 'rt', encoding='utf-8') as f:
                result = subprocess.run(
                    command,
                    env=env,
                    stdin=f,
                    capture_output=True,
                    text=True
                )

            if result.returncode == 0:
                logger.info("Gzip дамп загружен успешно")
                return True
            else:
                logger.error(f"Ошибка загрузки gzip дампа: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Ошибка загрузки gzip дампа: {e}")
            return False

    def _load_sql_dump(self) -> bool:
        """Загрузить SQL дамп через psql."""
        try:
            logger.info(f"Загрузка SQL дампа: {self.dump_path}")
            return self._load_plain_dump()
        except Exception as e:
            logger.error(f"Ошибка загрузки SQL дампа: {e}")
            return False

    def _load_gz_dump(self) -> bool:
        """Загрузить gzip дамп."""
        try:
            logger.info(f"Загрузка gzip дампа: {self.dump_path}")

            from sqlalchemy.engine.url import make_url
            
            db_url = make_url(self.database_url)
            host = db_url.host
            port = db_url.port
            user = db_url.username
            password = db_url.password
            database = db_url.database

            env = os.environ.copy()
            env['PGPASSWORD'] = password if password else ''

            command = [
                'psql',
                f'--host={host}',
                f'--port={port}' if port else '',
                f'--username={user}',
                f'--dbname={database}'
            ]
            
            command = [c for c in command if c]
            
            with gzip.open(self.dump_path, 'rt', encoding='utf-8') as f:
                result = subprocess.run(
                    command,
                    env=env,
                    stdin=f,
                    capture_output=True,
                    text=True
                )

            if result.returncode == 0:
                logger.info("Gzip дамп загружен успешно")
                return True
            else:
                logger.error(f"Ошибка загрузки gzip дампа: {result.stderr}")
                return False
        except Exception as e:
            logger.error(f"Ошибка загрузки gzip дампа: {e}")
            return False

        except Exception as e:
            logger.error(f"Ошибка загрузки дампа: {e}")
            return False

    def _load_sql_dump(self) -> bool:
        """Загрузить SQL дамп."""
        try:
            logger.info(f"Загрузка SQL дампа: {self.dump_path}")

            with open(self.dump_path, "r", encoding="utf-8") as f:
                sql_content = f.read()

            with self.Session() as session:

                try:
                    session.execute(text(sql_content))
                    session.commit()
                    logger.info("SQL дамп загружен успешно")
                    return True
                except Exception as e:
                    session.rollback()
                    logger.error(f"Ошибка: {e}")
                    return False

        except Exception as e:
            logger.error(f"Ошибка загрузки SQL дампа: {e}")
            return False

    def _load_gz_dump(self) -> bool:
        """Загрузить gzip дамп."""
        try:
            logger.info(f"Загрузка gzip дампа: {self.dump_path}")

            with gzip.open(self.dump_path, "rt", encoding="utf-8") as f:
                sql_content = f.read()

            with self.Session() as session:

                try:
                    session.execute(text(sql_content))
                    session.commit()
                    logger.info("Gzip дамп загружен успешно")
                    return True
                except Exception as e:
                    session.rollback()
                    logger.error(f"Ошибка: {e}")
                    return False

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

                try:
                    session.execute(text(sql_content))
                    session.commit()
                    logger.info("SQL дамп загружен успешно")
                    return True
                except Exception as e:
                    session.rollback()
                    logger.error(f"Ошибка: {e}")
                    return False

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

                            session.execute(text(sql_content))

                            logger.info("Gzip дамп загружен успешно")
                            return True
                        else:
                            # Обычный SQL файл
                            with open(member, "r", encoding="utf-8") as f:
                                sql_content = f.read()

                                session.execute(text(sql_content))

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

                try:
                    session.execute(text("SET client_encoding to 'UTF8'"))

                    # Для каждого SQL файла
                    for sql_file in sql_files:
                        logger.info(f"Загрузка файла: {sql_file}")

                        if sql_file.endswith(".gz"):
                            with gzip.open(sql_file, "rt", encoding="utf-8") as f:
                                sql_content = f.read()
                        else:
                            with open(sql_file, "r", encoding="utf-8") as f:
                                sql_content = f.read()

                        session.execute(text(sql_content))

                    session.commit()
                    logger.info(
                        f"Загружено {len(sql_files)} файлов из директории дампа"
                    )
                    return True
                except Exception as e:
                    session.rollback()
                    logger.error(f"Ошибка: {e}")
                    return False

        except Exception as e:
            logger.error(f"Ошибка загрузки дампа из директории: {e}")
            return False

            with self.Session() as session:

                try:
                    for sql_file in sql_files:
                        logger.info(f"Загрузка файла: {sql_file}")
                        if sql_file.endswith(".gz"):
                            with gzip.open(sql_file, "rt", encoding="utf-8") as f:
                                sql_content = f.read()
                        else:
                            with open(sql_file, "r", encoding="utf-8") as f:
                                sql_content = f.read()

                        session.execute(text(sql_content))

                    session.commit()
                    logger.info(f"Загружено {len(sql_files)} файлов из директории")
                    return True
                except Exception as e:
                    session.rollback()
                    logger.error(f"Ошибка: {e}")
                    return False

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


def check_tables_from_loader(database_url: str) -> dict:
    """Проверить состояние таблиц напрямую (без создания экземпляра класса).

    Args:
        database_url: URL подключения к PostgreSQL

    Returns:
        Словарь с статистикой по таблицам
    """
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    try:
        engine = create_engine(database_url)
        with Session(engine) as session:
            stats = {
                "question_answer_total": 0,
                "chunk_total": 0,
                "qa_embeddings": 0,
                "chunk_embeddings": 0,
                "qa_coverage_percent": 0,
                "chunk_coverage_percent": 0,
            }

            try:
                qa_count = session.execute(
                    text("SELECT COUNT(*) FROM question_answer")
                ).scalar()
                stats["question_answer_total"] = qa_count
            except Exception:
                pass

            try:
                chunk_count = session.execute(
                    text("SELECT COUNT(*) FROM chunk")
                ).scalar()
                stats["chunk_total"] = chunk_count
            except Exception:
                pass

            try:
                qa_embeddings_count = session.execute(
                    text(
                        "SELECT COUNT(*) FROM question_answer WHERE embedding IS NOT NULL"
                    )
                ).scalar()
                stats["qa_embeddings"] = qa_embeddings_count
                if qa_count > 0:
                    stats["qa_coverage_percent"] = qa_embeddings_count / qa_count * 100
            except Exception:
                pass

            try:
                chunk_embeddings_count = session.execute(
                    text("SELECT COUNT(*) FROM chunk WHERE embedding IS NOT NULL")
                ).scalar()
                stats["chunk_embeddings"] = chunk_embeddings_count
                if chunk_count > 0:
                    stats["chunk_coverage_percent"] = (
                        chunk_embeddings_count / chunk_count * 100
                    )
            except Exception:
                pass

            logger.info(f"Статистика таблиц: {stats}")
            return stats
    except Exception as e:
        logger.error(f"Ошибка проверки таблиц: {e}")
        return {"error": str(e)}


def load_dump_main(database_url: str, dump_path: str) -> bool:
    """Главная функция для загрузки дампа.

    Args:
        database_url: URL подключения к PostgreSQL
        dump_path: Путь к дампу

    Returns:
        True если дамп загружен успешно
    """
    loader = DatabaseDumpLoader(database_url, dump_path)

    loader.connect()

    if loader.load_dump():
        loader.check_tables()
        loader.close()
        return True
    else:
        loader.close()
        return False


def load_dump_main(database_url: str, dump_path: str) -> bool:
    """Главная функция для загрузки дампа.

    Args:
        database_url: URL подключения к PostgreSQL
        dump_path: Путь к дампу

    Returns:
        True если дамп загружен успешно
    """
    loader = DatabaseDumpLoader(database_url, dump_path)

    loader.connect()

    if loader.load_dump():
        loader.check_tables()
        loader.close()
        return True
    else:
        loader.close()
        return False
