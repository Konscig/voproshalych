#!/usr/bin/env python3
"""Скрипт для запуска дашборда бенчарков.

Использование:
    python run_dashboard.py
"""

import subprocess
import sys
from pathlib import Path

# Определяем директории
project_root = Path(__file__).parent
benchmarks_dir = project_root / "Submodules" / "voproshalych" / "benchmarks"

# Запускаем дашборд из директории benchmarks
dashboard_path = benchmarks_dir / "dashboard.py"

print(f"Запуск дашборда: {dashboard_path}")

if not dashboard_path.exists():
    print(f"❌ Файл не найден: {dashboard_path}")
    sys.exit(1)

try:
    # Запускаем streamlit
    result = subprocess.run(
        ["streamlit", "run", str(dashboard_path)],
        check=True,
        capture_output=True,
        text=True,
    )

    print(result.stdout)

    if result.returncode != 0:
        print(f"❌ Ошибка запуска: {result.stderr}")
        sys.exit(result.returncode)

except Exception as e:
    print(f"❌ Ошибка: {e}")
    sys.exit(1)
