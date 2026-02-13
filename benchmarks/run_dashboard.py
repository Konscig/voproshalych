#!/usr/bin/env python3
"""Скрипт для запуска дашборда бенчарков.

Использование:
    python run_dashboard.py
"""

import sys

try:
    from benchmarks.dashboard import main

    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print(
        "Убедитесь, что вы запускаете этот скрипт из директории Submodules/voproshalych"
    )
    sys.exit(1)
