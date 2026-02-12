"""Модуль для генерации отчётов бенчмарков.

Генерирует отчёты в формате Markdown с результатами бенчмарков.
"""

import json
from datetime import datetime
from typing import Dict


class ReportGenerator:
    """Генератор отчётов бенчмарков.

    Генерирует отчёты в формате Markdown.
    """

    def generate_retrieval_report(
        self,
        metrics: Dict[str, float],
        dataset_name: str,
        tier: str = "all",
    ) -> str:
        """Сгенерировать отчёт по бенчмаркам поиска.

        Args:
            metrics: Словарь с вычисленными метриками
            dataset_name: Название датасета
            tier: Tier бенчмарка (tier1, tier2, all)

        Returns:
            Отчёт в формате Markdown
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        report = f"""# Бенчмарк поиска (Tier {tier})

**Дата выполнения:** {timestamp}
**Датасет:** {dataset_name}

---

## Результаты

### Метрики качества поиска

| Метрика | Значение |
|---------|----------|
"""

        # Добавляем метрики в таблицу
        for metric, value in sorted(metrics.items()):
            report += f"| {metric} | {value:.4f} |\n"

        report += """
---

## Выводы

"""

        # Добавляем аналитические выводы
        report += self._analyze_retrieval_metrics(metrics)

        report += f"\n---\n\n*Сгенерировано автоматически: {timestamp}*\n"

        return report

    def _analyze_retrieval_metrics(self, metrics: Dict[str, float]) -> str:
        """Проанализировать метрики поиска и сформировать выводы.

        Args:
            metrics: Словарь с вычисленными метриками

        Returns:
            Текст с анализом и выводами
        """
        analysis = []

        # Recall@1
        recall_1 = metrics.get("recall@1", 0.0)
        if recall_1 >= 0.8:
            analysis.append(
                "- ✅ Отличный результат: Recall@1 = "
                f"{recall_1:.2%} (система находит релевантные "
                "документы на первой позиции в 80% случаев)"
            )
        elif recall_1 >= 0.5:
            analysis.append(
                "- ⚠️ Хороший результат: Recall@1 = "
                f"{recall_1:.2%} (система находит релевантные "
                "документы на первой позиции в 50% случаев)"
            )
        else:
            analysis.append(
                "- ❌ Плохой результат: Recall@1 = "
                f"{recall_1:.2%} (системе сложно находить "
                "релевантные документы на первой позиции)"
            )

        # MRR
        mrr = metrics.get("mrr", 0.0)
        if mrr >= 0.8:
            analysis.append(
                "- ✅ Отличный MRR: система быстро находит "
                f"релевантные документы (MRR = {mrr:.4f})"
            )
        elif mrr >= 0.5:
            analysis.append(
                "- ⚠️ Средний MRR: системе требуется больше "
                f"времени для поиска (MRR = {mrr:.4f})"
            )
        else:
            analysis.append(
                "- ❌ Низкий MRR: системе сложно находить "
                f"релевантные документы (MRR = {mrr:.4f})"
            )

        # Recall@10
        recall_10 = metrics.get("recall@10", 0.0)
        if recall_10 >= 0.9:
            analysis.append(
                "- ✅ Отличное покрытие: Recall@10 = "
                f"{recall_10:.2%} (почти все релевантные "
                "документы находятся в топ-10)"
            )
        elif recall_10 >= 0.7:
            analysis.append(
                "- ⚠️ Хорошее покрытие: Recall@10 = "
                f"{recall_10:.2%} (большинство релевантных "
                "документов находятся в топ-10)"
            )
        else:
            analysis.append(
                "- ❌ Низкое покрытие: Recall@10 = "
                f"{recall_10:.2%} (многие релевантные "
                "документы не находятся в топ-10)"
            )

        return "\n".join(analysis)

    def save_report(self, report: str, output_path: str) -> None:
        """Сохранить отчёт в файл.

        Args:
            report: Текст отчёта
            output_path: Путь для сохранения файла
        """
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)

        print(f"Отчёт сохранён: {output_path}")

    def save_metrics_json(self, metrics: Dict, output_path: str) -> None:
        """Сохранить метрики в формате JSON.

        Args:
            metrics: Словарь с метриками
            output_path: Путь для сохранения файла
        """
        # Преобразуем numpy типы в нативные Python типы
        import numpy as np

        def convert_value(value):
            if isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return float(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            return value

        converted_metrics = {k: convert_value(v) for k, v in metrics.items()}

        # Добавляем timestamp
        converted_metrics["timestamp"] = datetime.now().isoformat()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(converted_metrics, f, indent=2, ensure_ascii=False)

        print(f"Метрики сохранены: {output_path}")
