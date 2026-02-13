#!/usr/bin/env python3
"""Скрипт для запуска дашборда бенчарков.

Использование:
    python run_dashboard.py
"""

import subprocess
import sys
from pathlib import Path

# Определяем директорию с дашбордом (текущая директория benchmarks/)
dashboard_path = Path(__file__).parent / "dashboard.py"

print(f"Запуск дашборда: {dashboard_path}")

if not dashboard_path.exists():
    print(f"❌ Файл не найден: {dashboard_path}")
    sys.exit(1)

try:
    # Запускаем Python-модуль напрямую
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.dashboard"],
        check=True,
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"❌ Ошибка запуска")
        sys.exit(result.returncode)

except Exception as e:
    print(f"❌ Ошибка: {e}")
    sys.exit(1)
