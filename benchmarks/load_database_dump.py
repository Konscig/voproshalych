def drop_tables_via_docker() -> bool:
    """Удалить существующие таблицы через docker compose.

    Returns:
        True если успешно
    """
    try:
        db_container_name = "virtassist-db"

        result = subprocess.run(
            f'docker exec {db_container_name} psql -U {os.environ.get("POSTGRES_USER", "postgres")} -d {os.environ.get("POSTGRES_DB", "virtassist")} -c "DROP TABLE IF EXISTS question_answer CASCADE; DROP TABLE IF EXISTS chunk CASCADE; DROP TABLE IF EXISTS holiday CASCADE; DROP TABLE IF EXISTS admin CASCADE;"',
            capture_output=True,
            text=True,
            shell=True,
        )

        if result.returncode == 0:
            logger.info("Таблицы удалены")
            return True
        else:
            logger.error(f"Ошибка удаления таблиц: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Ошибка удаления таблиц: {e}")
        return False
