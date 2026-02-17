"""CLI скрипт для автоматической загрузки дампа БД через docker compose.

Использование:
    python load_database_dump.py --dump /path/to/dump.sql
    python load_database_dump.py --drop-tables
"""

import argparse
import logging
import os
import subprocess
import sys
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


def drop_tables_via_docker() -> bool:
    """Удалить существующие таблицы через docker compose.

    Returns:
        True если успешно
    """
    try:
        cmd = [
            "docker",
            "compose",
            "exec",
            "-T",
            "db",
            "psql",
            "-U",
            os.environ.get("POSTGRES_USER", "postgres"),
            "-d",
            os.environ.get("POSTGRES_DB", "virtassist"),
            "-c",
            "DROP TABLE IF EXISTS question_answer CASCADE; DROP TABLE IF EXISTS chunk CASCADE; DROP TABLE IF EXISTS holiday CASCADE; DROP TABLE IF EXISTS admin CASCADE;",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info("Таблицы удалены")
            return True
        else:
            logger.error(f"Ошибка удаления таблиц: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Ошибка удаления таблиц: {e}")
        return False


def load_dump_main(dump_path: str) -> bool:
    """Главная функция для загрузки дампа через docker compose.

    Args:
        dump_path: Путь к дампу

    Returns:
        True если дамп загружен успешно
    """
    dump_abs_path = os.path.abspath(dump_path)

    if not os.path.exists(dump_abs_path):
        logger.error(f"Файл дампа не найден: {dump_abs_path}")
        return False

    try:
        logger.info("Очистка таблиц перед загрузкой дампа...")
        if not drop_tables_via_docker():
            logger.error("Не удалось очистить таблицы")
            return False

        logger.info(f"Загрузка дампа: {dump_abs_path}")

        db_container_name = "virtassist-db"
        remote_path = "/tmp/dump.sql"

        logger.info("Копирование дампа в контейнер db...")
        copy_cmd = [
            "docker",
            "cp",
            dump_abs_path,
            f"{db_container_name}:{remote_path}",
        ]
        copy_result = subprocess.run(copy_cmd, capture_output=True, text=True)

        if copy_result.returncode != 0:
            logger.error(f"Ошибка копирования дампа: {copy_result.stderr}")
            return False

        logger.info("Загрузка дампа через psql...")
        psql_cmd = [
            "docker",
            "compose",
            "exec",
            "-T",
            "db",
            "psql",
            "-U",
            os.environ.get("POSTGRES_USER", "postgres"),
            "-d",
            os.environ.get("POSTGRES_DB", "virtassist"),
            "-f",
            remote_path,
        ]
        psql_result = subprocess.run(psql_cmd, capture_output=True, text=True)

        if psql_result.returncode == 0:
            logger.info("Дамп загружен успешно")
            return True
        else:
            logger.error(f"Ошибка загрузки дампа: {psql_result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Ошибка загрузки дампа: {e}")
        return False


def main():
    """Главная функция CLI скрипта."""
    parser = argparse.ArgumentParser(
        description="Автоматическая загрузка дампа БД для бенчарков через docker compose"
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
        "--drop-tables-only",
        action="store_true",
        help="Только удалить таблицы (без загрузки дампа)",
    )

    args = parser.parse_args()

    # Проверяем аргументы
    if not args.dump and not args.dump_dir:
        parser.print_help()
        sys.exit(1)

    if args.drop_tables_only:
        logger.info("Удаление существующих таблиц...")
        if drop_tables_via_docker():
            logger.info("✅ Таблицы удалены!")
            sys.exit(0)
        else:
            logger.error("❌ Ошибка удаления таблиц")
            sys.exit(1)

    dump_path = args.dump if args.dump else args.dump_dir
    success = load_dump_main(dump_path)

    if success:
        logger.info("✅ Дамп загружен успешно!")
        logger.info("Система бенчарков готова к работе.")
    else:
        logger.error("❌ Ошибка загрузки дампа")
        sys.exit(1)


if __name__ == "__main__":
    main()
