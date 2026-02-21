"""Скрипт для загрузки дампа базы данных PostgreSQL.

Подгружает дамп и инициализирует таблицы для работы системы бенчмарков.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.utils.database_dump_loader import DatabaseDumpLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

load_dotenv(dotenv_path=".env.docker")


def get_database_url() -> str:
    """Получить URL базы данных из переменных окружения.

    Returns:
        URL подключения к PostgreSQL
    """
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port_str = os.environ.get("POSTGRES_PORT", "5432")
    user = os.environ.get("POSTGRES_USER", "postgres")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    database = os.environ.get("POSTGRES_DB", "virtassist")

    host = host.split(":")[0] if ":" in host else host
    port = port_str.split(":")[-1] if ":" in port_str else port_str

    # Если хост 'db' не резолвится (локальный запуск вне Docker), используем localhost
    if host == "db":
        import socket

        try:
            socket.gethostbyname("db")
        except socket.gaierror:
            logger.info(
                "Хост 'db' не доступен, используем 'localhost' для локального подключения"
            )
            host = "localhost"

    return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"


def drop_tables_via_docker(database_url: str) -> bool:
    """Удалить существующие таблицы через прямое подключение к БД.

    Args:
        database_url: URL подключения к PostgreSQL

    Returns:
        True если успешно
    """
    try:
        from sqlalchemy import create_engine, text

        logger.info("Удаление таблиц...")

        engine = create_engine(database_url)

        with engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS question_answer CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS chunk CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS holiday CASCADE"))
            conn.execute(text("DROP TABLE IF EXISTS admin CASCADE"))

        engine.dispose()

        logger.info("Таблицы успешно удалены")
        return True
    except Exception as e:
        logger.error(f"Ошибка удаления таблиц: {e}")
        return False


def main():
    """Главная функция."""
    parser = argparse.ArgumentParser(
        description="Загрузить дамп базы данных PostgreSQL"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Подробный вывод",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Тихий режим (только ошибки)",
    )
    parser.add_argument(
        "--dump",
        type=str,
        help="Путь к файлу дампа",
    )
    parser.add_argument(
        "--dump-dir",
        type=str,
        help="Путь к директории с дампами",
    )
    parser.add_argument(
        "--drop-tables-only",
        action="store_true",
        help="Только удалить таблицы без загрузки дампа",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    database_url = get_database_url()
    logger.info(f"Подключение к базе данных: {database_url}")

    if args.drop_tables_only:
        logger.info("Режим: удаление таблиц")
        if drop_tables_via_docker(database_url):
            print("✅ Таблицы успешно удалены")
            sys.exit(0)
        else:
            print("❌ Ошибка удаления таблиц")
            sys.exit(1)

    dump_path = args.dump or args.dump_dir

    if not dump_path:
        logger.error("Укажите путь к дампа через --dump или --dump-dir")
        parser.print_help()
        sys.exit(1)

    if not os.path.exists(dump_path):
        logger.error(f"Дамп не найден: {dump_path}")
        sys.exit(1)

    logger.info(f"Загрузка дампа: {dump_path}")

    loader = DatabaseDumpLoader(database_url, dump_path)

    try:
        loader.connect()

        if loader.load_dump():
            stats = loader.check_tables()
            loader.close()

            logger.info("Статистика таблиц:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")

            print("✅ Дамп базы данных успешно загружен")
            sys.exit(0)
        else:
            loader.close()
            print("❌ Ошибка загрузки дампа базы данных")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Ошибка: {e}")
        print(f"❌ Ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
