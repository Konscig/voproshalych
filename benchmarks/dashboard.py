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
    "tier_0": ["avg_intra_cluster_sim", "avg_inter_cluster_dist", "silhouette_score"],
    "tier_1": ["mrr", "hit_rate@1", "hit_rate@5", "hit_rate@10"],
    "tier_2": [
        "avg_faithfulness",
        "avg_answer_relevance",
        "avg_rouge1_f",
        "avg_rougeL_f",
        "avg_bleu",
    ],
    "tier_3": [
        "avg_e2e_score",
        "avg_semantic_similarity",
        "avg_rouge1_f",
        "avg_bleu",
    ],
    "tier_judge": [
        "consistency_score",
        "error_rate",
        "avg_latency_ms",
        "avg_faithfulness",
    ],
    "tier_judge_pipeline": [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "avg_latency_ms",
    ],
    "tier_ux": ["cache_hit_rate", "context_preservation", "multi_turn_consistency"],
    "tier_real_users": [
        "mrr",
        "recall@1",
        "recall@5",
        "precision@1",
        "precision@5",
        "ndcg@5",
    ],
}

QUALITY_BASELINES = {
    "avg_intra_cluster_sim": 0.85,
    "avg_inter_cluster_dist": 0.30,
    "silhouette_score": 0.50,
    "mrr": 0.8,
    "hit_rate@1": 0.7,
    "hit_rate@5": 0.9,
    "hit_rate@10": 0.95,
    "recall@1": 0.7,
    "recall@5": 0.9,
    "recall@10": 0.95,
    "precision@1": 0.7,
    "precision@5": 0.18,
    "ndcg@5": 0.8,
    "avg_faithfulness": 4.5,
    "avg_answer_relevance": 4.2,
    "avg_e2e_score": 4.2,
    "avg_semantic_similarity": 0.85,
    "avg_rouge1_f": 0.50,
    "avg_rouge2_f": 0.30,
    "avg_rougeL_f": 0.45,
    "avg_bleu": 30.0,
    "consistency_score": 0.90,
    "error_rate": 0.05,
    "avg_latency_ms": 3000.0,
    "accuracy": 0.85,
    "precision": 0.85,
    "recall": 0.85,
    "f1_score": 0.85,
    "cache_hit_rate": 0.40,
    "context_preservation": 0.70,
    "multi_turn_consistency": 0.70,
}


def _safe_float(value, default: float = 0.0) -> float:
    """Безопасно привести значение к float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
                    "dataset_type": record.dataset_type or "synthetic",
                    "judge_model": record.judge_model or "unknown",
                    "generation_model": record.generation_model or "unknown",
                    "overall_status": record.overall_status,
                    "tier_0": normalize_metrics(record.tier_0_metrics),
                    "tier_1": normalize_metrics(record.tier_1_metrics),
                    "tier_2": normalize_metrics(record.tier_2_metrics),
                    "tier_3": normalize_metrics(record.tier_3_metrics),
                    "tier_judge": normalize_metrics(record.tier_judge_metrics),
                    "tier_judge_pipeline": normalize_metrics(
                        record.tier_judge_pipeline_metrics
                    ),
                    "tier_ux": normalize_metrics(record.tier_ux_metrics),
                    "tier_real_users": normalize_metrics(record.real_user_metrics),
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

        rendered_dates = []
        for d in dates:
            if "_" in d:
                date_part, time_part = d.split("_", 1)
                rendered_dates.append(f"{date_part}-{time_part[:2]}")
            else:
                rendered_dates.append(d[:8])

        rendered_values = values[:]
        if len(rendered_dates) == 1:
            rendered_dates.append(f"{rendered_dates[0]}_p")
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
                            f"- Judge model: `{os.getenv('BENCHMARKS_JUDGE_MODEL') or os.getenv('JUDGE_MODEL') or Config.JUDGE_MODEL or 'unknown'}`",
                            f"- Generation model: `{os.getenv('GENERATION_MODEL') or Config.MISTRAL_MODEL or 'unknown'}`",
                            f"- Embedding model: `{Config.EMBEDDING_MODEL_PATH}`",
                        ]
                    )
                )

            with gr.Tab("Run Details"):
                self._create_run_details_tab()

            with gr.Tab("Runs Registry"):
                self._create_all_runs_tab()

            with gr.Tab("Metric History"):
                self._create_history_tab()

            with gr.Tab("Tier Comparison"):
                self._create_comparison_tab()

            with gr.Tab("Run Dataset"):
                self._create_run_dataset_tab()

            with gr.Tab("Справка"):
                self._create_reference_tab()

        return demo

    def _create_run_details_tab(self):
        if not self.runs:
            gr.Markdown("Нет данных о запусках бенчмарков в таблице benchmark_runs.")
            return

        ordered_runs = list(reversed(self.runs))

        def format_run_choice(run: Dict) -> str:
            return (
                f"{run['timestamp_readable']} | {run.get('dataset_type', 'synthetic')} | "
                f"{run['git_commit_hash'][:7]} | {run.get('dataset_file', 'N/A')}"
            )

        run_choices = [format_run_choice(r) for r in ordered_runs]

        gr.Markdown("### Выберите запуск для просмотра метрик")

        def get_run_metrics(selected: str) -> tuple:
            run = next(
                (r for r in ordered_runs if format_run_choice(r) == selected),
                None,
            )
            if not run:
                return "Запуск не найден", []

            info = "\n".join(
                [
                    f"### Запуск: {run['timestamp_readable']}",
                    f"- **Branch:** `{run['git_branch']}`",
                    f"- **Commit:** `{run['git_commit_hash']}`",
                    f"- **Author:** `{run['run_author']}`",
                    f"- **Dataset:** `{run.get('dataset_file', 'N/A')}`",
                    f"- **Dataset type:** `{run.get('dataset_type', 'synthetic')}`",
                    f"- **Judge model:** `{run.get('judge_model', 'N/A')}`",
                    f"- **Generation model:** `{run.get('generation_model', 'N/A')}`",
                    f"- **Overall status:** `{run['overall_status']}`",
                ]
            )

            rows = []

            tier_0 = run.get("tier_0", {})
            for m in [
                "avg_intra_cluster_sim",
                "avg_inter_cluster_dist",
                "silhouette_score",
            ]:
                rows.append(
                    [
                        "Tier 0 (Embedding)",
                        m.replace("_", " ").title(),
                        round(_safe_float(tier_0.get(m)), 4),
                    ]
                )

            tier_1 = run.get("tier_1", {})
            for m in ["mrr", "hit_rate@1", "hit_rate@5", "hit_rate@10"]:
                rows.append(
                    [
                        "Tier 1 (Retrieval)",
                        m.replace("_", " ").title(),
                        round(_safe_float(tier_1.get(m)), 4),
                    ]
                )

            tier_2 = run.get("tier_2", {})
            for m in [
                "avg_faithfulness",
                "avg_answer_relevance",
                "avg_rouge1_f",
                "avg_rougeL_f",
                "avg_bleu",
            ]:
                rows.append(
                    [
                        "Tier 2 (Generation)",
                        m.replace("_", " ").title(),
                        round(_safe_float(tier_2.get(m)), 4),
                    ]
                )

            tier_3 = run.get("tier_3", {})
            for m in [
                "avg_e2e_score",
                "avg_semantic_similarity",
                "avg_rouge1_f",
                "avg_bleu",
            ]:
                rows.append(
                    [
                        "Tier 3 (End-to-End)",
                        m.replace("_", " ").title(),
                        round(_safe_float(tier_3.get(m)), 4),
                    ]
                )

            tier_judge = run.get("tier_judge", {})
            for m in [
                "consistency_score",
                "error_rate",
                "avg_latency_ms",
                "avg_faithfulness",
            ]:
                rows.append(
                    [
                        "Tier Judge (Qwen)",
                        m.replace("_", " ").title(),
                        round(_safe_float(tier_judge.get(m)), 4),
                    ]
                )

            tier_judge_pipeline = run.get("tier_judge_pipeline", {})
            for m in ["accuracy", "precision", "recall", "f1_score", "avg_latency_ms"]:
                rows.append(
                    [
                        "Tier Judge Pipeline (Mistral)",
                        m.replace("_", " ").title(),
                        round(_safe_float(tier_judge_pipeline.get(m)), 4),
                    ]
                )

            tier_ux = run.get("tier_ux", {})
            for m in [
                "cache_hit_rate",
                "context_preservation",
                "multi_turn_consistency",
            ]:
                rows.append(
                    [
                        "Tier UX",
                        m.replace("_", " ").title(),
                        round(_safe_float(tier_ux.get(m)), 4),
                    ]
                )

            tier_real = run.get("tier_real_users", {})
            for m in [
                "mrr",
                "recall@1",
                "recall@5",
                "precision@1",
                "precision@5",
                "ndcg@5",
            ]:
                rows.append(
                    [
                        "Real Users (Retrieval)",
                        m.replace("_", " ").title(),
                        round(_safe_float(tier_real.get(m)), 4),
                    ]
                )

            return info, rows

        initial_info, initial_rows = get_run_metrics(run_choices[0])

        run_selector = gr.Dropdown(
            choices=run_choices,
            value=run_choices[0],
            label="Выберите запуск",
        )
        run_info = gr.Markdown(value=initial_info)
        metrics_table = gr.Dataframe(
            value=initial_rows,
            headers=["Tier", "Metric", "Value"],
            interactive=False,
            wrap=True,
        )

        def update_run(selected: str):
            info, rows = get_run_metrics(selected)
            return info, rows
            return info, rows

        run_selector.change(
            fn=update_run,
            inputs=[run_selector],
            outputs=[run_info, metrics_table],
        )

        def load_markdown_report(selected: str) -> str:
            run = next(
                (r for r in ordered_runs if format_run_choice(r) == selected),
                None,
            )
            if not run:
                return "Запуск не найден"

            dataset_file = run.get("dataset_file", "")
            if not dataset_file:
                return "Имя файла датасета не найдено в запуске"

            base_name = dataset_file.replace(".json", "")
            possible_paths = [
                os.path.join("benchmarks/reports", f"{base_name}.md"),
                os.path.join(
                    "benchmarks/reports", f"rag_benchmark_{base_name.split('_')[-1]}.md"
                ),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            return f.read()
                    except Exception as e:
                        return f"Ошибка чтения файла: {e}"

            return f"Markdown отчёт не найден. Искали: {possible_paths}"

        gr.Markdown("---")
        gr.Markdown("### 📄 Markdown Report")

        initial_markdown = load_markdown_report(run_choices[0])

        markdown_display = gr.Markdown(value=initial_markdown)

        def update_markdown(selected: str):
            return load_markdown_report(selected)

        run_selector.change(
            fn=update_markdown,
            inputs=[run_selector],
            outputs=[markdown_display],
        )

    def _create_history_tab(self):
        gr.Markdown("### Historical trend for selected metric")

        all_tiers = list(METRICS_BY_TIER.keys())

        default_tier = "tier_1"
        default_metric = "mrr"
        if not self.get_metric_history(default_tier, default_metric)[0]:
            for candidate_tier in all_tiers:
                for candidate_metric in self._metric_options_for_tier(candidate_tier):
                    if self.get_metric_history(candidate_tier, candidate_metric)[0]:
                        default_tier = candidate_tier
                        default_metric = candidate_metric
                        break
                if default_metric != "mrr" or default_tier != "tier_1":
                    break

        with gr.Row():
            tier_dropdown = gr.Dropdown(
                choices=all_tiers,
                value=default_tier,
                label="Tier",
            )
            metric_dropdown = gr.Dropdown(
                choices=self._metric_options_for_tier(default_tier),
                value=default_metric,
                label="Metric",
            )

        initial_history = pd.DataFrame(
            self._build_series_rows(
                *self.get_metric_history(default_tier, default_metric),
                default_metric,
                default_metric,
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
            if not metric:
                return pd.DataFrame(columns=["timestamp", "value", "series"])
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

        comparison_metrics = sorted(
            {metric for metrics in METRICS_BY_TIER.values() for metric in metrics}
        )

        tier_labels = {
            "tier_0": "Tier 0 (Embedding)",
            "tier_1": "Tier 1 (Retrieval)",
            "tier_2": "Tier 2 (Generation)",
            "tier_3": "Tier 3 (E2E)",
            "tier_judge": "Tier Judge (Qwen)",
            "tier_judge_pipeline": "Tier Judge Pipeline",
            "tier_ux": "Tier UX",
            "tier_real_users": "Real Users",
        }

        metric_dropdown = gr.Dropdown(
            choices=comparison_metrics,
            value="mrr",
            label="Metric",
        )

        initial_comparison_rows: List[Dict[str, str | float]] = []
        for tier_name, label in tier_labels.items():
            if tier_name in METRICS_BY_TIER:
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
                "tier_0": "Tier 0 (Embed)",
                "tier_1": "Tier 1 (Retrieval)",
                "tier_2": "Tier 2 (Generation)",
                "tier_3": "Tier 3 (E2E)",
                "tier_judge": "Tier Judge (Qwen)",
                "tier_judge_pipeline": "Tier Judge Pipeline",
                "tier_ux": "Tier UX",
                "tier_real_users": "Real Users",
            }

            longest_dates: List[str] = []
            for tier_name, label in series_by_tier.items():
                if tier_name not in METRICS_BY_TIER:
                    continue
                if metric not in METRICS_BY_TIER[tier_name]:
                    continue
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
        mode_filter = gr.Dropdown(
            choices=["all", "synthetic", "manual", "real-users"],
            value="all",
            label="Фильтр по типу датасета",
        )

        def build_registry_rows(filter_mode: str):
            rows = []
            for run in reversed(self.runs):
                run_mode = run.get("dataset_type", "synthetic")
                if filter_mode != "all" and run_mode != filter_mode:
                    continue
                t1 = run.get("tier_1", {})
                t2 = run.get("tier_2", {})
                t3 = run.get("tier_3", {})
                tjp = run.get("tier_judge_pipeline", {})
                rows.append(
                    [
                        run["timestamp_readable"][:16],
                        run["git_branch"][:12] if run.get("git_branch") else "N/A",
                        run.get("dataset_type", "synthetic")[:8],
                        round(_safe_float(t1.get("mrr")), 2),
                        round(_safe_float(t1.get("hit_rate@5")), 2),
                        round(_safe_float(t2.get("avg_faithfulness")), 2),
                        round(_safe_float(t2.get("avg_answer_relevance")), 2),
                        round(_safe_float(t3.get("avg_e2e_score")), 2),
                        round(_safe_float(tjp.get("accuracy")), 2) if tjp else "-",
                        run["overall_status"][:8],
                    ]
                )
            return rows

        headers = [
            "Time",
            "Branch",
            "Type",
            "MRR",
            "Hit@5",
            "Faith",
            "Relev",
            "E2E",
            "TJP Acc",
            "Status",
        ]

        initial_rows = build_registry_rows("all")
        table = gr.Dataframe(
            value=initial_rows,
            headers=headers,
            interactive=False,
            wrap=False,
        )

        def update_table(filter_mode: str):
            return gr.Dataframe(value=build_registry_rows(filter_mode), headers=headers)

        mode_filter.change(
            fn=update_table,
            inputs=[mode_filter],
            outputs=[table],
        )

    def _create_run_dataset_tab(self):
        ordered_runs = list(reversed(self.runs))

        if not ordered_runs:
            gr.Markdown("Нет запусков для просмотра привязанного датасета.")
            return

        def format_run_choice(run: Dict) -> str:
            return (
                f"{run['timestamp_readable']} | {run['dataset_type']} | "
                f"{run['git_commit_hash']} | {run['dataset_file']}"
            )

        def get_run_choices(filter_mode: str) -> List[str]:
            if filter_mode == "all":
                return [format_run_choice(run) for run in ordered_runs]
            return [
                format_run_choice(run)
                for run in ordered_runs
                if run.get("dataset_type") == filter_mode
            ]

        def load_dataset_preview(selected: str) -> tuple:
            if not selected:
                return "", []

            run = next(
                (item for item in ordered_runs if format_run_choice(item) == selected),
                None,
            )
            if run is None:
                return "Не найден выбранный запуск", []

            if run.get("dataset_type") == "real-users":
                real_metrics = run.get("tier_real_users", {})
                info = "\n".join(
                    [
                        f"**Dataset:** `{run.get('dataset_file', 'unknown')}`",
                        f"**Mode:** `{run.get('dataset_type', 'real-users')}`",
                        f"**MRR:** `{_safe_float(real_metrics.get('mrr')):.4f}`",
                        f"**Recall@5:** `{_safe_float(real_metrics.get('recall@5')):.4f}`",
                        f"**NDCG@5:** `{_safe_float(real_metrics.get('ndcg@5')):.4f}`",
                    ]
                )
                return info, []

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
                for row in dataset[:30]:
                    preview.append(
                        [
                            row.get("chunk_id"),
                            row.get("question", ""),
                            row.get("ground_truth_answer", ""),
                            row.get("confluence_url", ""),
                            ", ".join(
                                map(str, row.get("relevant_chunk_ids", []) or [])
                            ),
                            ", ".join(row.get("relevant_urls", []) or []),
                        ]
                    )

                meta = (
                    f"**Dataset:** `{dataset_file}`\n"
                    f"\n**Mode:** `{run.get('dataset_type', 'synthetic')}`\n"
                    f"\n**Rows:** {len(dataset)}\n"
                    f"\n**Linked run:** `{run['timestamp_readable']}`"
                )

                if dataset and isinstance(dataset[0], dict):
                    if (
                        "relevant_chunk_ids" in dataset[0]
                        or "relevant_urls" in dataset[0]
                    ):
                        meta += (
                            "\n\n`Manual dataset detected`: доступны поля "
                            "`relevant_chunk_ids`, `relevant_urls`, `notes`."
                        )

                return meta, preview
            except Exception as error:
                return (
                    f"**Dataset:** `{dataset_file}`\n\nОшибка чтения файла: `{error}`",
                    [],
                )

        mode_filter = gr.Dropdown(
            choices=["all", "synthetic", "manual", "real-users"],
            value="all",
            label="Тип запуска",
        )

        run_choices = get_run_choices("all")

        initial_meta, initial_preview = load_dataset_preview(run_choices[0])

        run_selector = gr.Dropdown(
            choices=run_choices,
            value=run_choices[0],
            label="Выберите запуск",
        )
        dataset_meta = gr.Markdown(value=initial_meta)
        dataset_preview = gr.Dataframe(
            value=initial_preview,
            headers=[
                "chunk_id",
                "question",
                "ground_truth_answer",
                "confluence_url",
                "relevant_chunk_ids",
                "relevant_urls",
            ],
            interactive=False,
            wrap=True,
        )

        def update_on_filter_change(filter_mode: str):
            options = get_run_choices(filter_mode)
            if not options:
                return (
                    gr.Dropdown(choices=[], value=None),
                    "",
                    [[]],
                )
            meta, preview = load_dataset_preview(options[0])
            return (
                gr.Dropdown(choices=options, value=options[0]),
                meta,
                preview,
            )

        def update_on_run_select(selected: str):
            meta, preview = load_dataset_preview(selected)
            return meta, preview

        mode_filter.change(
            fn=update_on_filter_change,
            inputs=[mode_filter],
            outputs=[run_selector, dataset_meta, dataset_preview],
        )

        run_selector.change(
            fn=update_on_run_select,
            inputs=[run_selector],
            outputs=[dataset_meta, dataset_preview],
        )

    def _create_reference_tab(self):
        gr.Markdown(
            """
# Справка по системе оценки RAG

## Назначение

Дашборд предназначен для мониторинга качества Retrieval-Augmented Generation системы. 
Он позволяет отслеживать метрики на различных этапах пайплайна и анализировать динамику изменений между запусками.

---

## Уровни оценки (Tiers)

### Tier 0: Embedding Quality

Оценивает внутреннее качество эмбеддингов без использования LLM.

- **avg_intra_cluster_sim**: средняя косинусная близость внутри кластера. Чем выше, тем лучше embeddings группируются по смыслу.
- **avg_inter_cluster_dist**: среднее расстояние между кластерами. Чем выше, тем лучше разделены тематические группы.
- **silhouette_score**: силуэт-коэффициент от -1 до 1. Значения ближе к 1 указывают на хорошую кластеризацию.

### Tier 1: Retrieval

Оценивает качество векторного поиска релевантных документов.

- **HitRate@K**: доля запросов, для которых хотя бы один релевантный документ найден в топ-K.
- **MRR (Mean Reciprocal Rank)**: среднее значение обратного ранга первого релевантного документа.
- **Recall@K**: полнота - доля найденных релевантных документов от общего числа.
- **Precision@K**: точность - доля релевантных документов среди топ-K.
- **NDCG@K**: нормализованная дисконтированная накопленная выгода.

Интерпретация: высокие значения MRR и Recall означают, что система находит релевантные документы.

### Tier 2: Generation

Оценивает качество сгенерированного ответа при известном релевантном контексте.

- **avg_faithfulness**: насколько ответ соответствует предоставленному контексту (шкала 1-5).
- **avg_answer_relevance**: насколько ответ релевантен вопросу (шкала 1-5).

Эти метрики выставляются LLM-судьей. Значения выше 4.0 считаются хорошими.

### Tier 3: End-to-End

Оценивает полный пайплайн retrieval + generation.

- **avg_e2e_score**: общая оценка качества ответа LLM-судьей (шкала 1-5).
- **avg_semantic_similarity**: косинусная близость между сгенерированным и эталонным ответами.

### Tier Judge (Qwen)

Оценивает согласованность judge-модели при повторных запусках.

- **consistency_score**: доля запросов, где оценка не изменилась при повторном запуске.
- **error_rate**: доля запросов, где произошла ошибка API.
- **avg_latency_ms**: среднее время отклика API в миллисекундах.

### Tier Judge Pipeline (Mistral)

Тестирует production judge, который решает "показывать ответ пользователю или нет".

- **accuracy**: точность решений судьи по сравнению с размеченными данными.
- **precision/recall/f1_score**: для класса "показывать ответ".

### Tier UX

Анализирует пользовательский опыт взаимодействия.

- **cache_hit_rate**: доля запросов, найденных в кэше похожих вопросов.
- **context_preserve**: сохранение контекста в многоturn диалогах.
- **multi_turn_consistency**: согласованность ответов в рамках одной сессии.

### Real Users

Метрики на реальных вопросах пользователей из таблицы QuestionAnswer.

- Используются те же метрики retrieval: MRR, Recall@K, Precision@K, NDCG@K.
- Позволяют оценить качество на реальном трафике.

---

## Типы запусков

### Synthetic

Автоматически сгенерированный датасет из чанков через LLM. Используется для быстрой проверки и регрессионного тестирования.

### Manual / Аннотация

Датасет, подготовленный для ручной разметки. Позволяет оценить качество с участием человека-эксперта.

### Real Users

Использование реальных вопросов из таблицы QuestionAnswer. Наиболее репрезентативный сценарий.

---

## Рекомендуемые пороги (Baseline)

| Tier | Метрика | Порог |
|------|---------|-------|
| Tier 0 | avg_intra_cluster_sim | >= 0.85 |
| Tier 0 | silhouette_score | >= 0.50 |
| Tier 1 | MRR | >= 0.80 |
| Tier 1 | HitRate@5 | >= 0.90 |
| Tier 1 | HitRate@10 | >= 0.95 |
| Tier 2 | Faithfulness | >= 4.5 |
| Tier 2 | Answer Relevance | >= 4.2 |
| Tier 3 | E2E Score | >= 4.2 |
| Tier 3 | Semantic Similarity | >= 0.85 |
| Tier Judge | consistency_score | >= 0.90 |
| Tier Judge | error_rate | <= 0.05 |
| Tier Judge Pipeline | accuracy | >= 0.85 |
| Tier Judge Pipeline | f1_score | >= 0.85 |

---

## Работа с историей

- **Metric History**: отслеживание динамики конкретной метрики во времени.
- **Tier Comparison**: сравнение производительности разных уровней на одной метрике.
- **Runs Registry**: список всех запусков с основными метриками для быстрого обзора.

При анализе регрессий обращайте внимание на метрики ниже baseline - они указывают на потенциальные проблемы в системе.
            """
        )


def find_free_port(start_port=7860, max_port=7870):
    """Найти свободный порт в диапазоне."""
    import socket

    for port in range(start_port, max_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return None


def main():
    """Главная функция запуска дашборда."""
    if not GRADIO_AVAILABLE:
        print("Gradio не установлен.")
        print("Установите зависимости: uv sync --extra dashboard")
        return

    import argparse

    parser = argparse.ArgumentParser(description="Запуск дашборда бенчмарков")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Порт для запуска дашборда (по умолчанию: первый свободный от 7860)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Определяем порт
    if args.port:
        port = args.port
        logger.info(f"Используем указанный порт: {port}")
    else:
        port = find_free_port()
        if port:
            logger.info(f"Найден свободный порт: {port}")
        else:
            logger.error("Не удалось найти свободный порт в диапазоне 7860-7870")
            return

    dashboard = RAGBenchmarkDashboard()
    interface = dashboard.create_interface()
    interface.launch(server_name="0.0.0.0", server_port=port, share=False, debug=True)


if __name__ == "__main__":
    main()
