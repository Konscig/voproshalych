"""CLI скрипт для автоматической загрузки дампа БД.

Использование:
    python load_database_dump.py --dump /path/to/dump.sql.gz
    python load_database_dump.py --dump-dir /path/to/dump/dir/
"""

import argparse
import logging
import sys

from benchmarks.utils.database_dump_loader import DatabaseDumpLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main():
    """Главная функция CLI скрипта."""
    parser = argparse.ArgumentParser(
        description="Автоматическая загрузка дампа БД для бенчмарков"
    )

    parser.add_argument(
        "--dump",
        type=str,
        help="Путь к файлу дампа (.sql, .sql.gz, .tar, .tar.gz)",
    )

    parser.add_argument(
        "--dump-dir",
        type=str,
        help="Путь к директории с дампами",
    )

    args = parser.parse_args()

    # Проверяем аргументы
    if not args.dump and not args.dump_dir:
        parser.print_help()
        sys.exit(1)

    # Получаем URL БД из окружения
    import os
    from os import environ

    database_url = environ.get(
        "BENCHMARK_DATABASE_URL", environ.get("POSTGRES_DB_URL", "")
    )

    if not database_url:
        # Формируем URL по умолчанию
        database_url = (
            f"postgresql://{environ.get('POSTGRES_USER', 'user')}:"
            f"{environ.get('POSTGRES_PASSWORD', 'password')}@"
            f"{environ.get('POSTGRES_HOST', 'localhost')}/"
            f"{environ.get('POSTGRES_DB', 'voproshalych')}"
        )

    logger.info(f"URL БД: {database_url}")

    # Создаём загрузчик
    dump_path = args.dump if args.dump else args.dump_dir
    loader = DatabaseDumpLoader(database_url, dump_path)

    # Подключаемся к БД
    loader.connect()

    # Проверяем текущее состояние таблиц
    logger.info("Проверяем текущее состояние таблиц...")
    stats_before = loader.check_tables()
    logger.info(f"Статистика до загрузки:")
    logger.info(f"  question_answer: {stats_before.get('question_answer_total', 0)}")
    logger.info(f"  chunk: {stats_before.get('chunk_total', 0)}")
    logger.info(f"  QA с эмбеддингами: {stats_before.get('qa_embeddings', 0)}")
    logger.info(f"  Чанки с эмбеддингами: {stats_before.get('chunk_embeddings', 0)}")

    # Загружаем дамп
    logger.info(f"Загрузка дампа: {dump_path}")
    success = loader.load_dump()

    if success:
        # Проверяем состояние после загрузки
        logger.info("Проверяем состояние после загрузки...")
        stats_after = loader.check_tables()
        logger.info(f"Статистика после загрузки:")
        logger.info(f"  question_answer: {stats_after.get('question_answer_total', 0)}")
        logger.info(f"  chunk: {stats_after.get('chunk_total', 0)}")
        logger.info(f"  QA с эмбеддингами: {stats_after.get('qa_embeddings', 0)}")
        logger.info(f"  Чанки с эмбеддингами: {stats_after.get('chunk_embeddings', 0)}")

        # Сравниваем
        qa_added = stats_after.get("question_answer_total", 0) - stats_before.get(
            "question_answer_total", 0
        )
        chunk_added = stats_after.get("chunk_total", 0) - stats_before.get(
            "chunk_total", 0
        )
        logger.info(f"\nДобавлено записей:")
        logger.info(f"  question_answer: {qa_added}")
        logger.info(f"  chunk: {chunk_added}")

        logger.info("\n✅ Дамп загружен успешно!")
        logger.info("Система бенчарков готова к работе.")
    else:
        logger.error("\n❌ Ошибка загрузки дампа")
        sys.exit(1)

    # Закрываем подключение
    loader.close()


if __name__ == "__main__":
    main()
