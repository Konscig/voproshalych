"""Интерактивный дашборд для просмотра метрик бенчмарков.

Использует Gradio для визуализации метрик качества поиска и генерации.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

try:
    import gradio as gr

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

logger = logging.getLogger(__name__)


class BenchmarkDashboard:
    """Дашборд для просмотра метрик бенчмарков.

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

        # Загружаем JSON файлы
        for filename in os.listdir(self.reports_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.reports_dir, filename)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        metrics[filename] = data
                except Exception as e:
                    logger.error(f"Ошибка загрузки {filename}: {e}")

        logger.info(f"Загружено {len(metrics)} файлов с метриками")
        return metrics

    def get_metric_value(self, metric_name: str, metric_file: str = None) -> float:
        """Получить значение метрики.

        Args:
            metric_name: Название метрики (recall@1, precision@5 и т.д.)
            metric_file: Файл с метриками

        Returns:
            Значение метрики
        """
        if metric_file and metric_file in self.metrics_data:
            return self.metrics_data[metric_file].get(metric_name, 0.0)
        elif metric_file is None:
            for data in self.metrics_data.values():
                if metric_name in data:
                    return data[metric_name]
        return 0.0

    def get_all_metric_values(self, metric_name: str) -> List[float]:
        """Получить все значения метрики по всем отчётам.

        Args:
            metric_name: Название метрики

        Returns:
            Список значений метрики
        """
        values = []

        for data in self.metrics_data.values():
            if metric_name in data:
                values.append(data[metric_name])

        return values

    def get_benchmark_list(self) -> List[str]:
        """Получить список всех бенчмарков.

        Returns:
            Список названий бенчмарков
        """
        benchmarks = set()

        for filename in self.metrics_data.keys():
            if filename.startswith("retrieval_tier"):
                benchmark = filename.split("_")[2]
                benchmarks.add(benchmark)

        return sorted(list(benchmarks))

    def get_metric_history(self, metric_name: str) -> Dict[str, List]:
        """Получить историю изменения метрики по датам.

        Args:
            metric_name: Название метрики

        Returns:
            Словарь {дата: значение}
        """
        history = {}

        for filename, data in self.metrics_data.items():
            if metric_name in data:
                timestamp = data.get("timestamp", "")
                date_str = timestamp.split("T")[0] if timestamp else "unknown"
                history[date_str] = data[metric_name]

        return history

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

        with gr.Blocks(title="Дашборд метрик бенчарков") as demo:
            gr.Markdown("# 📊 Дашборд метрик бенчарков Вопрошалыча")

            with gr.Tab("Обзор"):
                gr.Markdown("### Сводка по всем бенчмаркам")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 📈 Текущие метрики")

                        benchmark_list = self.get_benchmark_list()
                        benchmark_dropdown = gr.Dropdown(
                            choices=benchmark_list,
                            label="Бенчмарк",
                            value=benchmark_list[0] if benchmark_list else None,
                        )

                        metric_names = [
                            "recall@1",
                            "recall@3",
                            "recall@5",
                            "recall@10",
                            "precision@1",
                            "precision@3",
                            "precision@5",
                            "precision@10",
                            "mrr",
                            "ndcg@5",
                            "ndcg@10",
                        ]

                        metric_dropdown = gr.Dropdown(
                            choices=metric_names,
                            label="Метрика",
                            value="recall@1",
                        )

                        metric_value = gr.Number(
                            label="Значение",
                            precision=4,
                            value=0.0,
                            interactive=False,
                        )

                    with gr.Column():
                        gr.Markdown("#### 📋 Список метрик")

                        metric_table = gr.Dataframe(
                            label="Все метрики",
                            headers=["Метрика", "Значение"],
                            interactive=False,
                        )

                    def update_metrics(benchmark, metric):
                        metric_file = f"retrieval_tier1_{benchmark}_*.json"
                        value = self.get_metric_value(metric, metric_file)
                        metric_value.value = value

                        # Обновляем таблицу
                        metrics_data = []
                        for m in metric_names:
                            m_value = self.get_metric_value(m, metric_file)
                            metrics_data.append([m, f"{m_value:.4f}"])

                        metric_table.value = metrics_data

                        return metric_value

                    benchmark_dropdown.change(
                        fn=lambda b, m: update_metrics(b, "recall@1"),
                        inputs=[benchmark_dropdown],
                        outputs=[metric_value],
                    )

                    metric_dropdown.change(
                        fn=update_metrics,
                        inputs=[benchmark_dropdown, metric_dropdown],
                        outputs=[metric_value],
                    )

            with gr.Tab("История"):
                gr.Markdown("### 📈 История изменений метрик")

                metric_history_dropdown = gr.Dropdown(
                    choices=["recall@1", "mrr", "ndcg@10"],
                    label="Метрика",
                    value="recall@1",
                )

                history_plot = gr.LinePlot(
                    label="История",
                    x_label="Дата",
                    y_label="Значение",
                )

                def update_history_plot(metric_name):
                    history = self.get_metric_history(metric_name)

                    if not history:
                        return None

                    dates = sorted(history.keys())
                    values = [history[d] for d in dates]

                    data = {"Дата": dates, metric_name: values}

                    return data

                metric_history_dropdown.change(
                    fn=update_history_plot,
                    inputs=[metric_history_dropdown],
                    outputs=[history_plot],
                )

            with gr.Tab("Сравнение"):
                gr.Markdown("### 🔍 Сравнение бенчмарков")

                with gr.Row():
                    b1 = gr.Checkbox(
                        label="Tier 1 (golden_set)",
                        value=True,
                    )
                    b2 = gr.Checkbox(
                        label="Tier 1 (questions_with_url)",
                        value=False,
                    )
                    b3 = gr.Checkbox(
                        label="Tier 2 (golden_set)",
                        value=False,
                    )

                comparison_plot = gr.LinePlot(
                    label="Сравнение метрик",
                    x_label="Бенчмарк",
                    y_label="Значение",
                )

                def update_comparison_plot(tier1_gs, tier1_qw, tier2_gs):
                    selected = []
                    if tier1_gs:
                        selected.append("golden_set")
                    if tier1_qw:
                        selected.append("questions_with_url")
                    if tier2_gs:
                        selected.append("golden_set_tier2")

                    if not selected:
                        return None

                    metric_name = "recall@1"
                    data = {"Бенчмарк": [], metric_name: []}

                    for benchmark in selected:
                        value = self.get_metric_value(metric_name, f"*{benchmark}*")
                        data["Бенчмарк"].append(benchmark)
                        data[metric_name].append(value)

                    return data

                b1.change(
                    fn=update_comparison_plot,
                    inputs=[b1, b2, b3],
                    outputs=[comparison_plot],
                )
                b2.change(
                    fn=update_comparison_plot,
                    inputs=[b1, b2, b3],
                    outputs=[comparison_plot],
                )
                b3.change(
                    fn=update_comparison_plot,
                    inputs=[b1, b2, b3],
                    outputs=[comparison_plot],
                )

            with gr.Tab("Автозагрузка дампа"):
                gr.Markdown("### 🔄 Автоматическая загрузка дампа БД")

                gr.Markdown("""
                **Инструкция по автозагрузке:**

                1. Получите дамп базы данных с продакшн-сервера
                2. Сохраните в директорию `benchmarks/data/dump/`
                3. Запустите QA-сервис (он автоматически загрузит дамп)
                4. Проверьте статистику таблиц после загрузки

                **Формат дампа:**
                - `.sql` - SQL файл
                - `.sql.gz` - Сжатый SQL файл
                - `.tar` - Архив с SQL файлами
                - `.tar.gz` - Сжатый архив

                **Пример команды загрузки:**
                ```bash
                python benchmarks/load_database_dump.py \\
                    --dump benchmarks/data/dump/virtassist-main-YYYYMMDD-HHMMSS.sql.gz
                ```
                """)

                dump_path = gr.Textbox(
                    label="Путь к файлу дампа",
                    placeholder="benchmarks/data/dump/dump.sql.gz",
                )

                check_stats_btn = gr.Button("📊 Проверить статистику")

                def check_database_stats():
                    try:
                        from benchmarks.utils.database_dump_loader import (
                            DatabaseDumpLoader,
                        )
                        from os import environ

                        database_url = (
                            f"postgresql://{environ.get('POSTGRES_USER', 'user')}:"
                            f"{environ.get('POSTGRES_PASSWORD', 'password')}@"
                            f"{environ.get('POSTGRES_HOST', 'localhost')}/"
                            f"{environ.get('POSTGRES_DB', 'voproshalych')}"
                        )

                        loader = DatabaseDumpLoader(database_url, "")
                        loader.connect()
                        stats = loader.check_tables()
                        loader.close()

                        return (
                            f"✅ Статистика загружена:\n\n"
                            f"**question_answer:** {stats.get('question_answer_total', 0)} записей\n"
                            f"**chunk:** {stats.get('chunk_total', 0)} записей\n"
                            f"**QA с эмбеддингами:** {stats.get('qa_embeddings', 0)} "
                            f"({stats.get('qa_coverage_percent', 0):.1f}%)\n"
                            f"**Чанки с эмбеддингами:** {stats.get('chunk_embeddings', 0)} "
                            f"({stats.get('chunk_coverage_percent', 0):.1f}%)"
                        )
                    except Exception as e:
                        return f"❌ Ошибка: {str(e)}"

                check_stats_btn.click(
                    fn=check_database_stats,
                    outputs=[gr.Textbox(label="Результат", visible=True)],
                )

        return demo


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

    dashboard = BenchmarkDashboard()
    interface = dashboard.create_interface()

    interface.launch(
        server_name="0.0.0.0",
        share=False,
        debug=True,
    )


if __name__ == "__main__":
    main()
