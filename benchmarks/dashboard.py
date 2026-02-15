"""Интерактивный дашборд для просмотра метрик бенчмарков RAG-системы.

Использует Gradio для визуализации метрик качества поиска и генерации.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple

try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class RAGBenchmarkDashboard:
    """Дашборд для просмотра метрик RAG-бенчмарков.

    Attributes:
        reports_dir: Директория с отчётами
        metrics_data: Загруженные метрики
    """

    def __init__(self, reports_dir: str = "benchmarks/reports"):
        """Инициализировать дашборд.

        Args:
            reports_dir: Директория с отчётами
        """
        self.reports_dir = reports_dir
        self.metrics_data = self._load_metrics()

    def _load_metrics(self) -> Dict:
        """Загрузить все JSON отчёты с метриками.

        Returns:
            Словарь с метриками
        """
        metrics = {}

        if not os.path.exists(self.reports_dir):
            logger.warning(f"Директория с отчётами не найдена: {self.reports_dir}")
            return metrics

        for filename in os.listdir(self.reports_dir):
            if filename.startswith("rag_benchmark_") and filename.endswith(".json"):
                filepath = os.path.join(self.reports_dir, filename)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)

                        timestamp_str = self._extract_timestamp(filename)
                        data["timestamp"] = timestamp_str
                        data["filename"] = filename

                        metrics[timestamp_str] = data
                except Exception as e:
                    logger.error(f"Ошибка загрузки {filename}: {e}")

        logger.info(f"Загружено {len(metrics)} файлов с метриками")
        return metrics

    def _extract_timestamp(self, filename: str) -> str:
        """Извлечь timestamp из имени файла.

        Args:
            filename: Имя файла

        Returns:
            Строка timestamp
        """
        match = re.search(r"rag_benchmark_(\d{8}_\d{6})", filename)
        if match:
            return match.group(1)
        return filename

    def get_latest_run(self) -> Optional[Dict]:
        """Получить последний запуск бенчмарка.

        Returns:
            Данные последнего запуска
        """
        if not self.metrics_data:
            return None

        latest_timestamp = max(self.metrics_data.keys())
        return self.metrics_data[latest_timestamp]

    def get_metric_history(
        self, tier: str, metric: str
    ) -> Tuple[List[str], List[float]]:
        """Получить историю изменения метрики.

        Args:
            tier: Уровень бенчмарка (tier_1, tier_2, tier_3)
            metric: Название метрики

        Returns:
            Кортеж (даты, значения)
        """
        sorted_timestamps = sorted(self.metrics_data.keys())

        dates = []
        values = []

        for timestamp in sorted_timestamps:
            data = self.metrics_data[timestamp]

            if tier in data and metric in data[tier]:
                try:
                    value = float(data[tier][metric])
                    dates.append(timestamp)
                    values.append(value)
                except (ValueError, TypeError):
                    continue

        return dates, values

    def get_all_tier_metrics(self, tier: str) -> Dict[str, List]:
        """Получить все метрики для уровня.

        Args:
            tier: Уровень бенчмарка

        Returns:
            Словарь {метрика: [(дата, значение)]}
        """
        sorted_timestamps = sorted(self.metrics_data.keys())
        metrics = {}

        for timestamp in sorted_timestamps:
            data = self.metrics_data[timestamp]

            if tier in data:
                for metric_name, value in data[tier].items():
                    if metric_name == "tier":
                        continue

                    try:
                        float_value = float(value)
                        if metric_name not in metrics:
                            metrics[metric_name] = []
                        metrics[metric_name].append((timestamp, float_value))
                    except (ValueError, TypeError):
                        continue

        return metrics

    def create_interface(self):
        """Создать интерфейс Gradio.

        Returns:
            Объект интерфейса Gradio

        Raises:
            ImportError: Если gradio не установлен
        """
        if not GRADIO_AVAILABLE:
            raise ImportError(
                "Gradio не установлен. Установите его с помощью: pip install gradio"
            )

        with gr.Blocks(title="Дашборд метрик RAG-бенчмарков") as demo:
            gr.Markdown("# 📊 Дашборд метрик RAG-бенчмарков Вопрошалыча")

            with gr.Tab("Последний запуск"):
                self._create_latest_run_tab()

            with gr.Tab("История"):
                self._create_history_tab()

            with gr.Tab("Сравнение уровней"):
                self._create_comparison_tab()

            with gr.Tab("Все запуски"):
                self._create_all_runs_tab()

        return demo

    def _create_latest_run_tab(self):
        """Создать вкладку с последним запуском."""
        latest_run = self.get_latest_run()

        if not latest_run:
            gr.Markdown("❌ Нет данных о запусках бенчмарков")
            return

        gr.Markdown(f"### Последний запуск: {latest_run['timestamp']}")

        with gr.Row():
            for tier_name in ["tier_1", "tier_2", "tier_3"]:
                if tier_name in latest_run:
                    with gr.Column():
                        gr.Markdown(f"#### {tier_name.upper()}")

                        metrics_data = []
                        for metric, value in latest_run[tier_name].items():
                            if metric != "tier" and isinstance(value, (int, float)):
                                metrics_data.append([metric, f"{value:.4f}"])

                        gr.Dataframe(
                            value=metrics_data,
                            headers=["Метрика", "Значение"],
                            label=f"Метрики {tier_name.upper()}",
                            interactive=False,
                        )

    def _create_history_tab(self):
        """Создать вкладку с историей изменений."""
        gr.Markdown("### 📈 История изменений метрик")

        with gr.Row():
            tier_dropdown = gr.Dropdown(
                choices=["tier_1", "tier_2", "tier_3"],
                value="tier_1",
                label="Уровень бенчмарка",
            )

            metric_dropdown = gr.Dropdown(
                choices=[
                    "hit_rate@1",
                    "hit_rate@5",
                    "hit_rate@10",
                    "mrr",
                    "avg_faithfulness",
                    "avg_answer_relevance",
                    "avg_e2e_score",
                    "avg_semantic_similarity",
                ],
                value="mrr",
                label="Метрика",
            )

        plot = gr.LinePlot(
            label="История метрики",
            x="Дата",
            y="Значение",
        )

        def update_plot(tier, metric):
            dates, values = self.get_metric_history(tier, metric)

            if not dates:
                return None

            return {
                "Дата": dates,
                metric: values,
            }

        tier_dropdown.change(
            fn=update_plot,
            inputs=[tier_dropdown, metric_dropdown],
            outputs=[plot],
        )

        metric_dropdown.change(
            fn=update_plot,
            inputs=[tier_dropdown, metric_dropdown],
            outputs=[plot],
        )

    def _create_comparison_tab(self):
        """Создать вкладку сравнения уровней."""
        gr.Markdown("### 🔍 Сравнение уровней бенчмарков")

        metric_dropdown = gr.Dropdown(
            choices=[
                "hit_rate@1",
                "hit_rate@5",
                "hit_rate@10",
                "mrr",
                "avg_faithfulness",
                "avg_answer_relevance",
                "avg_e2e_score",
                "avg_semantic_similarity",
            ],
            value="mrr",
            label="Метрика",
        )

        plot = gr.LinePlot(
            label="Сравнение уровней",
            x="Дата",
            y="Значение",
        )

        def update_comparison_plot(metric):
            tier_1_dates, tier_1_values = self.get_metric_history("tier_1", metric)
            tier_2_dates, tier_2_values = self.get_metric_history("tier_2", metric)
            tier_3_dates, tier_3_values = self.get_metric_history("tier_3", metric)

            data = {"Дата": []}

            if tier_1_dates:
                data["Дата"] = tier_1_dates
                data["Tier 1"] = tier_1_values

            if tier_2_dates:
                if not data["Дата"]:
                    data["Дата"] = tier_2_dates
                data["Tier 2"] = tier_2_values

            if tier_3_dates:
                if not data["Дата"]:
                    data["Дата"] = tier_3_dates
                data["Tier 3"] = tier_3_values

            if not data["Дата"]:
                return None

            return data

        metric_dropdown.change(
            fn=update_comparison_plot,
            inputs=[metric_dropdown],
            outputs=[plot],
        )

    def _create_all_runs_tab(self):
        """Создать вкладку со всеми запусками."""
        gr.Markdown("### 📋 Все запуски бенчмарков")

        all_runs = []
        for timestamp, data in sorted(self.metrics_data.items()):
            run_info = {
                "Timestamp": timestamp,
                "Tier 1 MRR": data.get("tier_1", {}).get("mrr", "N/A"),
                "Tier 2 Faithfulness": data.get("tier_2", {}).get(
                    "avg_faithfulness", "N/A"
                ),
                "Tier 3 E2E": data.get("tier_3", {}).get("avg_e2e_score", "N/A"),
            }
            all_runs.append(run_info)

        gr.Dataframe(
            value=all_runs,
            headers=["Timestamp", "Tier 1 MRR", "Tier 2 Faithfulness", "Tier 3 E2E"],
            label="Все запуски",
            interactive=False,
        )


def main():
    """Главная функция для запуска дашборда."""
    if not GRADIO_AVAILABLE:
        print("❌ Gradio не установлен!")
        print("Установите его с помощью: pip install gradio")
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    dashboard = RAGBenchmarkDashboard()
    interface = dashboard.create_interface()

    interface.launch(
        server_name="0.0.0.0",
        share=False,
        debug=True,
    )


if __name__ == "__main__":
    main()
