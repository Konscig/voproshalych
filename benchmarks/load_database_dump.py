"""CLI скрипт для автоматической загрузки дампа БД.

Использование:
    python load_database_dump.py --dump /path/to/dump.sql.gz
    python load_database_dump.py --dump-dir /path/to/dump/dir/
"""

import argparse
import logging
import sys
import os
import subprocess
from pathlib import Path
from sys import path as sys_path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys_path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env.docker")


def drop_tables(database_url: str) -> bool:
    """Удалить существующие таблицы.

    Args:
        database_url: URL подключения к PostgreSQL

    Returns:
        True если успешно
    """
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    try:
        engine = create_engine(database_url)
        with Session(engine) as session:
            session.execute(text("DROP TABLE IF EXISTS question_answer CASCADE"))
            session.execute(text("DROP TABLE IF EXISTS chunk CASCADE"))
            session.execute(text("DROP TABLE IF EXISTS holiday CASCADE"))
            session.execute(text("DROP TABLE IF EXISTS admin CASCADE"))
            session.commit()
            logger.info("Таблицы удалены")
        return True
    except Exception as e:
        logger.error(f"Ошибка удаления таблиц: {e}")
        return False


def load_dump_main(database_url: str, dump_path: str) -> bool:
    """Главная функция для загрузки дампа.

    Args:
        database_url: URL подключения к PostgreSQL
        dump_path: Путь к дампу

    Returns:
        True если дамп загружен успешно
    """
    try:
        logger.info(f"Загрузка дампа: {dump_path}")

        cmd = ["psql", "-d", database_url, "-f", dump_path]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Дамп загружен успешно")
            return True
        else:
            logger.error(f"Ошибка загрузки дампа: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Ошибка загрузки дампа: {e}")
        return False


def main():
    """Главная функция CLI скрипта."""
    parser = argparse.ArgumentParser(
        description="Автоматическая загрузка дампа БД для бенчарков"
    )

    parser.add_argument(
        "--dump",
        type=str,
        help="Путь к файлу дампа (.sql, .sql.gz, .tar, .tar.gz, .dump)",
    )

    parser.add_argument(
        "--dump-dir",
        type=str,
        help="Путь к директории с дампами",
    )

    parser.add_argument(
        "--drop-tables",
        action="store_true",
        help="Удалить существующие таблицы перед загрузкой дампа",
    )

    args = parser.parse_args()

    # Проверяем аргументы
    if not args.dump and not args.dump_dir:
        parser.print_help()
        sys.exit(1)

    # Получаем URL БД из окружения
    database_url = (
        f"postgresql://{os.environ.get('POSTGRES_USER', 'voproshalych')}:"
        f"{os.environ.get('POSTGRES_PASSWORD', 'voproshalych_password')}@"
        f"{os.environ.get('POSTGRES_HOST', 'localhost:5432')}/"
        f"{os.environ.get('POSTGRES_DB', 'virtassist')}"
    )

    logger.info(f"URL БД: {database_url}")

    if args.drop_tables:
        logger.info("Удаление существующих таблиц...")
        drop_tables(database_url)
        sys.exit(0)

    success = load_dump_main(database_url, args.dump if args.dump else args.dump_dir)

    if success:
        logger.info("✅ Дамп загружен успешно!")
        logger.info("Система бенчарков готова к работе.")
    else:
        logger.error("❌ Ошибка загрузки дампа")
        sys.exit(1)


if __name__ == "__main__":
    main()
