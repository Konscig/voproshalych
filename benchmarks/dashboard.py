"""Интерактивный аналитический дашборд для метрик RAG-бенчмарков."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    import gradio as gr
    import pandas as pd

    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

from sqlalchemy import select
from sqlalchemy.orm import Session

from qa.config import Config
from qa.database import BenchmarkRun, create_engine, ensure_benchmark_runs_schema

logger = logging.getLogger(__name__)

APP_TITLE = "RAG Quality Assurance System"

METRICS_BY_TIER = {
    "tier_1": ["mrr", "hit_rate@1", "hit_rate@5", "hit_rate@10"],
    "tier_2": ["avg_faithfulness", "avg_answer_relevance"],
    "tier_3": ["avg_e2e_score", "avg_semantic_similarity"],
}

QUALITY_BASELINES = {
    "mrr": 0.8,
    "hit_rate@1": 0.7,
    "hit_rate@5": 0.9,
    "hit_rate@10": 0.95,
    "avg_faithfulness": 4.5,
    "avg_answer_relevance": 4.2,
    "avg_e2e_score": 4.2,
    "avg_semantic_similarity": 0.85,
}


class RAGBenchmarkDashboard:
    """Дашборд для просмотра качества Retrieval и Generation."""

    def __init__(self):
        self.engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
        ensure_benchmark_runs_schema(self.engine)
        self.runs = self._load_runs()

    def _load_runs(self) -> List[Dict]:
        """Загрузить запуски бенчмарков из таблицы benchmark_runs."""

        def normalize_metrics(raw_metrics):
            if isinstance(raw_metrics, dict):
                return raw_metrics
            if isinstance(raw_metrics, str):
                try:
                    parsed = json.loads(raw_metrics)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    return {}
            return {}

        with Session(self.engine) as session:
            records = session.scalars(
                select(BenchmarkRun).order_by(BenchmarkRun.timestamp.asc())
            ).all()

        runs = []
        for record in records:
            timestamp_str = record.timestamp.strftime("%Y%m%d_%H%M%S")
            runs.append(
                {
                    "id": record.id,
                    "timestamp": timestamp_str,
                    "timestamp_readable": record.timestamp.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "git_branch": record.git_branch or "unknown",
                    "git_commit_hash": record.git_commit_hash or "unknown",
                    "run_author": record.run_author or "unknown",
                    "dataset_file": record.dataset_file or "unknown",
                    "overall_status": record.overall_status,
                    "tier_1": normalize_metrics(record.tier_1_metrics),
                    "tier_2": normalize_metrics(record.tier_2_metrics),
                    "tier_3": normalize_metrics(record.tier_3_metrics),
                }
            )

        logger.info("Загружено %s запусков из benchmark_runs", len(runs))
        return runs

    def get_latest_run(self) -> Optional[Dict]:
        """Получить последний запуск бенчмарка."""
        if not self.runs:
            return None
        return self.runs[-1]

    def get_metric_history(
        self, tier: str, metric: str
    ) -> tuple[List[str], List[float]]:
        """Получить историю изменения метрики по всем запускам."""
        dates: List[str] = []
        values: List[float] = []

        for run in self.runs:
            tier_metrics = run.get(tier, {})
            value = tier_metrics.get(metric)
            if value is None:
                continue
            try:
                values.append(float(value))
                dates.append(run["timestamp"])
            except (TypeError, ValueError):
                continue

        return dates, values

    def _metric_options_for_tier(self, tier: str) -> List[str]:
        return METRICS_BY_TIER.get(tier, [])

    def _build_series_rows(
        self,
        dates: List[str],
        values: List[float],
        metric: str,
        series_name: str,
    ) -> List[Dict[str, str | float]]:
        """Построить строки для gr.LinePlot с baseline-линией."""
        rows: List[Dict[str, str | float]] = []

        rendered_dates = dates[:]
        rendered_values = values[:]
        if len(rendered_dates) == 1:
            rendered_dates.append(f"{rendered_dates[0]}_point")
            rendered_values.append(rendered_values[0])

        for date, value in zip(rendered_dates, rendered_values):
            rows.append({"timestamp": date, "value": value, "series": series_name})

        baseline = QUALITY_BASELINES.get(metric)
        if baseline is not None:
            for date in rendered_dates:
                rows.append(
                    {"timestamp": date, "value": baseline, "series": "Baseline"}
                )

        return rows

    def create_interface(self):
        """Создать интерфейс Gradio."""
        if not GRADIO_AVAILABLE:
            raise ImportError(
                "Gradio не установлен. Выполните: uv sync --extra dashboard"
            )

        with gr.Blocks(title=APP_TITLE) as demo:
            gr.Markdown(f"# {APP_TITLE}")
            gr.Markdown(
                "Панель мониторинга качества Retrieval, Generation и end-to-end "
                "ответов RAG-системы."
            )

            with gr.Accordion("System Info", open=False):
                gr.Markdown(
                    "\n".join(
                        [
                            f"- Judge model: `{Config.JUDGE_MODEL or 'unknown'}`",
                            f"- Generation model: `{os.getenv('GENERATION_MODEL') or Config.MISTRAL_MODEL or 'unknown'}`",
                            f"- Embedding model: `{Config.EMBEDDING_MODEL_PATH}`",
                        ]
                    )
                )

            with gr.Tab("Latest Run"):
                self._create_latest_run_tab()

            with gr.Tab("Metric History"):
                self._create_history_tab()

            with gr.Tab("Tier Comparison"):
                self._create_comparison_tab()

            with gr.Tab("Runs Registry"):
                self._create_all_runs_tab()

            with gr.Tab("Run Dataset"):
                self._create_run_dataset_tab()

            with gr.Tab("Справка"):
                self._create_reference_tab()

        return demo

    def _create_latest_run_tab(self):
        latest_run = self.get_latest_run()
        if not latest_run:
            gr.Markdown("Нет данных о запусках бенчмарков в таблице benchmark_runs.")
            return

        gr.Markdown(
            "\n".join(
                [
                    "### Latest benchmark run",
                    f"- Timestamp: `{latest_run['timestamp_readable']}`",
                    f"- Branch: `{latest_run['git_branch']}`",
                    f"- Commit: `{latest_run['git_commit_hash']}`",
                    f"- Author: `{latest_run['run_author']}`",
                    f"- Dataset: `{latest_run['dataset_file']}`",
                    f"- Overall status: `{latest_run['overall_status']}`",
                ]
            )
        )

        summary_rows = []
        summary_rows.append(
            {
                "Tier": "Tier 1 (Retrieval)",
                "HitRate@5": f"{float(latest_run.get('tier_1', {}).get('hit_rate@5', 0.0)):.4f}",
                "MRR": f"{float(latest_run.get('tier_1', {}).get('mrr', 0.0)):.4f}",
                "Faithfulness": "-",
                "Relevance": "-",
                "E2E Score": "-",
            }
        )
        summary_rows.append(
            {
                "Tier": "Tier 2 (Generation)",
                "HitRate@5": "-",
                "MRR": "-",
                "Faithfulness": f"{float(latest_run.get('tier_2', {}).get('avg_faithfulness', 0.0)):.4f}",
                "Relevance": f"{float(latest_run.get('tier_2', {}).get('avg_answer_relevance', 0.0)):.4f}",
                "E2E Score": "-",
            }
        )
        summary_rows.append(
            {
                "Tier": "Tier 3 (E2E)",
                "HitRate@5": "-",
                "MRR": "-",
                "Faithfulness": "-",
                "Relevance": "-",
                "E2E Score": f"{float(latest_run.get('tier_3', {}).get('avg_e2e_score', 0.0)):.4f}",
            }
        )

        gr.Dataframe(
            value=summary_rows,
            headers=[
                "Tier",
                "HitRate@5",
                "MRR",
                "Faithfulness",
                "Relevance",
                "E2E Score",
            ],
            label="Latest Metrics Summary",
            interactive=False,
            wrap=True,
        )

    def _create_history_tab(self):
        gr.Markdown("### Historical trend for selected metric")

        with gr.Row():
            tier_dropdown = gr.Dropdown(
                choices=["tier_1", "tier_2", "tier_3"],
                value="tier_1",
                label="Tier",
            )
            metric_dropdown = gr.Dropdown(
                choices=self._metric_options_for_tier("tier_1"),
                value="mrr",
                label="Metric",
            )

        initial_history = pd.DataFrame(
            self._build_series_rows(
                *self.get_metric_history("tier_1", "mrr"), "mrr", "mrr"
            )
        )

        plot = gr.LinePlot(
            value=initial_history,
            x="timestamp",
            y="value",
            color="series",
            title="Metric vs Baseline",
            x_title="Run",
            y_title="Value",
            tooltip="all",
            height=420,
            color_map={"Baseline": "#808080"},
            sort="x",
        )

        def update_metric_choices(tier: str):
            options = self._metric_options_for_tier(tier)
            value = options[0] if options else None
            return gr.Dropdown(choices=options, value=value)

        def update_plot(tier: str, metric: str):
            dates, values = self.get_metric_history(tier, metric)
            if not dates:
                return pd.DataFrame(columns=["timestamp", "value", "series"])
            rows = self._build_series_rows(dates, values, metric, metric)
            return pd.DataFrame(rows)

        tier_dropdown.change(
            fn=update_metric_choices,
            inputs=[tier_dropdown],
            outputs=[metric_dropdown],
        ).then(
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
        gr.Markdown("### Compare tiers by metric")

        metric_dropdown = gr.Dropdown(
            choices=sorted(QUALITY_BASELINES.keys()),
            value="mrr",
            label="Metric",
        )

        initial_comparison_rows: List[Dict[str, str | float]] = []
        for tier_name, label in {
            "tier_1": "Tier 1",
            "tier_2": "Tier 2",
            "tier_3": "Tier 3",
        }.items():
            dates, values = self.get_metric_history(tier_name, "mrr")
            if dates:
                initial_comparison_rows.extend(
                    self._build_series_rows(dates, values, "mrr", label)
                )

        plot = gr.LinePlot(
            value=pd.DataFrame(initial_comparison_rows),
            x="timestamp",
            y="value",
            color="series",
            title="Tier comparison",
            x_title="Run",
            y_title="Value",
            tooltip="all",
            height=420,
            color_map={"Baseline": "#808080"},
            sort="x",
        )

        def update_comparison_plot(metric: str):
            all_rows: List[Dict[str, str | float]] = []
            series_by_tier = {
                "tier_1": "Tier 1",
                "tier_2": "Tier 2",
                "tier_3": "Tier 3",
            }

            longest_dates: List[str] = []
            for tier_name, label in series_by_tier.items():
                dates, values = self.get_metric_history(tier_name, metric)
                if not dates:
                    continue
                if len(dates) > len(longest_dates):
                    longest_dates = dates[:]
                all_rows.extend(
                    self._build_series_rows(dates, values, metric, series_name=label)
                )

            if not all_rows:
                return pd.DataFrame(columns=["timestamp", "value", "series"])

            baseline = QUALITY_BASELINES.get(metric)
            if baseline is not None and longest_dates:
                baseline_dates = longest_dates
                if len(baseline_dates) == 1:
                    baseline_dates = [
                        baseline_dates[0],
                        f"{baseline_dates[0]}_point",
                    ]
                all_rows.extend(
                    {
                        "timestamp": date,
                        "value": baseline,
                        "series": "Baseline",
                    }
                    for date in baseline_dates
                )

            deduped = []
            seen = set()
            for row in all_rows:
                key = (row["timestamp"], row["value"], row["series"])
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(row)
            return pd.DataFrame(deduped)

        metric_dropdown.change(
            fn=update_comparison_plot,
            inputs=[metric_dropdown],
            outputs=[plot],
        )

    def _create_all_runs_tab(self):
        all_runs = []
        for run in reversed(self.runs):
            tier_1 = run.get("tier_1", {})
            tier_2 = run.get("tier_2", {})
            tier_3 = run.get("tier_3", {})
            all_runs.append(
                {
                    "Timestamp": run["timestamp_readable"],
                    "Branch / Commit": f"{run['git_branch']} / {run['git_commit_hash']}",
                    "T1: HitRate@5": f"{float(tier_1.get('hit_rate@5', 0.0)):.4f}",
                    "T1: MRR": f"{float(tier_1.get('mrr', 0.0)):.4f}",
                    "T2: Faithfulness": f"{float(tier_2.get('avg_faithfulness', 0.0)):.4f}",
                    "T2: Relevance": f"{float(tier_2.get('avg_answer_relevance', 0.0)):.4f}",
                    "T3: E2E Score": f"{float(tier_3.get('avg_e2e_score', 0.0)):.4f}",
                }
            )

        gr.Dataframe(
            value=all_runs,
            headers=[
                "Timestamp",
                "Branch / Commit",
                "T1: HitRate@5",
                "T1: MRR",
                "T2: Faithfulness",
                "T2: Relevance",
                "T3: E2E Score",
            ],
            interactive=False,
            wrap=True,
        )

    def _create_run_dataset_tab(self):
        run_choices = [
            f"{run['timestamp_readable']} | {run['git_commit_hash']} | {run['dataset_file']}"
            for run in reversed(self.runs)
        ]

        if not run_choices:
            gr.Markdown("Нет запусков для просмотра привязанного датасета.")
            return

        run_selector = gr.Dropdown(
            choices=run_choices,
            value=run_choices[0],
            label="Выберите запуск",
        )
        dataset_meta = gr.Markdown()
        dataset_preview = gr.Dataframe(
            headers=["chunk_id", "question", "ground_truth_answer", "confluence_url"],
            interactive=False,
            wrap=True,
        )

        ordered_runs = list(reversed(self.runs))

        def update_dataset_preview(selected: str):
            if not selected:
                return "", []

            run_index = run_choices.index(selected)
            run = ordered_runs[run_index]
            dataset_file = run.get("dataset_file") or ""
            benchmark_dir = Path(__file__).resolve().parent
            project_root = benchmark_dir.parent
            candidate_paths = [
                benchmark_dir / "data" / dataset_file,
                project_root / "benchmarks" / "data" / dataset_file,
                Path.cwd() / "data" / dataset_file,
                Path.cwd() / "benchmarks" / "data" / dataset_file,
            ]
            dataset_path = next(
                (path for path in candidate_paths if path.exists()),
                candidate_paths[0],
            )

            if not dataset_path.exists():
                return (
                    f"**Dataset:** `{dataset_file}`\n\nФайл не найден по пути `{dataset_path}`",
                    [],
                )

            try:
                import json

                with open(dataset_path, "r", encoding="utf-8") as f:
                    dataset = json.load(f)

                preview = []
                for row in dataset[:20]:
                    preview.append(
                        [
                            row.get("chunk_id"),
                            row.get("question", ""),
                            row.get("ground_truth_answer", ""),
                            row.get("confluence_url", ""),
                        ]
                    )

                meta = (
                    f"**Dataset:** `{dataset_file}`\n"
                    f"\n**Rows:** {len(dataset)}\n"
                    f"\n**Linked run:** `{run['timestamp_readable']}`"
                )
                return meta, preview
            except Exception as error:
                return (
                    f"**Dataset:** `{dataset_file}`\n\nОшибка чтения файла: `{error}`",
                    [],
                )

        run_selector.change(
            fn=update_dataset_preview,
            inputs=[run_selector],
            outputs=[dataset_meta, dataset_preview],
        )

        initial_meta, initial_preview = update_dataset_preview(run_choices[0])
        dataset_meta.value = initial_meta
        dataset_preview.value = initial_preview

    def _create_reference_tab(self):
        gr.Markdown(
            """
### Формальная постановка задачи

Система оценивает качество RAG в трёх уровнях:
- **Tier 1 (Retrieval):** качество ранжирования документов-кандидатов.
- **Tier 2 (Generation):** качество генерации ответа при фиксированном контексте.
- **Tier 3 (E2E):** итоговое качество ответа в полном pipeline retrieval+generation.

### Определения метрик

- **Precision@k**: доля релевантных документов в top-k.
- **Recall@k**: доля найденных релевантных документов среди всех релевантных.
- **MRR (Mean Reciprocal Rank):**
  `MRR = (1 / |Q|) * Σ_q (1 / rank_q)` , где `rank_q` — позиция первого релевантного
  документа для запроса `q`.
- **Faithfulness (1..5):** аппроксимация условной вероятности того, что факты в ответе
  выводимы из предоставленного контекста без галлюцинаций.

### Методология LLM-as-a-Judge

Для Tier 2 и Tier 3 используется схема **LLM-as-a-Judge**:
1. Формируется вход: `question`, `context`, `system_answer`, `ground_truth`.
2. Судья выставляет числовую оценку по фиксированным рубрикам.
3. На уровне отчёта агрегируются средние значения и сравниваются с baseline.

### Интерпретация baseline

- Tier 1: `MRR >= 0.80`, `Hit@5 >= 0.90`, `Hit@10 >= 0.95`
- Tier 2: `Faithfulness >= 4.5`, `Relevance >= 4.2`
- Tier 3: `E2E >= 4.2`, `Semantic Similarity >= 0.85`

Значения ниже baseline указывают на деградацию качества retrieval, prompt-политики,
или согласованности генерации с контекстом.
            """
        )


def main():
    """Главная функция запуска дашборда."""
    if not GRADIO_AVAILABLE:
        print("Gradio не установлен.")
        print("Установите зависимости: uv sync --extra dashboard")
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    dashboard = RAGBenchmarkDashboard()
    interface = dashboard.create_interface()
    interface.launch(server_name="0.0.0.0", share=False, debug=True)


if __name__ == "__main__":
    main()
