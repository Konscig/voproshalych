#!/usr/bin/env python3
"""Скрипт для запуска дашборда бенчарков.

Использование:
    python benchmarks/run_dashboard.py
"""

import sys
from pathlib import Path

# Добавляем родительскую директорию в sys.path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    from benchmarks.dashboard import main

    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"❌ Ошибка импорта: {e}")
    print(
        "Убедитесь, что вы запускаете этот скрипт из директории Submodules/voproshalych"
    )
    print("Или запустите напрямую: python benchmarks/dashboard.py")
    sys.exit(1)
