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

logger = logging.getLogger(__name__)

APP_TITLE = "RAG Quality Assurance System"

METRICS_BY_TIER = {
    "tier_0": [
        "avg_nn_distance",
        "std_nn_distance",
        "density_score",
        "avg_spread",
        "max_spread",
        "spread_std",
        "effective_dimensionality",
        "avg_pairwise_distance",
        "std_pairwise_distance",
        "min_pairwise_distance",
        "max_pairwise_distance",
    ],
    "tier_1": ["mrr", "hit_rate@1", "hit_rate@5", "hit_rate@10"],
    "tier_2": [
        "avg_faithfulness",
        "avg_answer_relevance",
        "avg_answer_correctness",
        "response_exact_match_rate",
        "response_semantic_consistency",
        "avg_rouge1_f",
        "avg_rougeL_f",
        "avg_bleu",
    ],
    "tier_3": [
        "avg_e2e_score",
        "avg_semantic_similarity",
        "avg_dot_similarity",
        "avg_euclidean_distance",
        "response_exact_match_rate",
        "response_semantic_consistency",
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
    "domain_analysis_metrics": [
        "total_questions",
        "with_answers",
        "without_answers",
        "with_answers_rate",
        "scored_questions",
        "score_5_questions",
        "score_1_questions",
        "avg_question_length",
    ],
    "utilization_metrics": [
        "total_chunks",
        "used_chunks",
        "unused_chunks",
        "utilization_rate",
        "question_count",
        "top_k",
    ],
    "topic_coverage_metrics": [
        "n_topics",
        "total_questions",
        "avg_chunks_per_topic",
        "top_k",
    ],
}

QUALITY_BASELINES = {
    "avg_nn_distance": 0.30,
    "density_score": 3.00,
    "avg_pairwise_distance": 0.45,
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
    "avg_answer_correctness": 4.2,
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
        self.reports_dir = Path("benchmarks/reports")
        self.runs = self._load_runs()

    def _load_runs(self) -> List[Dict]:
        """Загрузить запуски бенчмарков из JSON файла."""
        benchmark_runs_path = self.reports_dir / "benchmark_runs.json"

        if not benchmark_runs_path.exists():
            logger.info("Файл benchmark_runs.json не найден, загрузка 0 запусков")
            return []

        with open(benchmark_runs_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        runs = []
        for record in records:
            timestamp_str = record.get("timestamp", "")[:15]
            runs.append(
                {
                    "id": record.get("id"),
                    "timestamp": timestamp_str,
                    "timestamp_readable": timestamp_str.replace("T", " ")[:19],
                    "git_branch": record.get("git_branch") or "unknown",
                    "git_commit_hash": record.get("git_commit_hash") or "unknown",
                    "run_author": record.get("run_author") or "unknown",
                    "dataset_file": record.get("dataset_file") or "unknown",
                    "dataset_type": record.get("dataset_type") or "synthetic",
                    "judge_model": record.get("judge_model") or "unknown",
                    "generation_model": record.get("generation_model") or "unknown",
                    "embedding_model": record.get("embedding_model") or "unknown",
                    "judge_eval_mode": record.get("judge_eval_mode") or "direct",
                    "consistency_runs": record.get("consistency_runs") or 1,
                    "overall_status": record.get("overall_status"),
                    "tier_0": record.get("tier_0_metrics") or {},
                    "tier_1": record.get("tier_1_metrics") or {},
                    "tier_2": record.get("tier_2_metrics") or {},
                    "tier_3": record.get("tier_3_metrics") or {},
                    "tier_judge": record.get("tier_judge_metrics") or {},
                    "tier_judge_pipeline": record.get("tier_judge_pipeline_metrics")
                    or {},
                    "tier_ux": record.get("tier_ux_metrics") or {},
                    "tier_real_users": record.get("real_user_metrics") or {},
                    "utilization_metrics": record.get("utilization_metrics") or {},
                    "topic_coverage_metrics": record.get("topic_coverage_metrics")
                    or {},
                    "domain_analysis_metrics": record.get("domain_analysis_metrics")
                    or {},
                    "executive_summary": record.get("executive_summary", ""),
                }
            )

        logger.info("Загружено %s запусков из benchmark_runs.json", len(runs))
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

            with gr.Tab("Vector Space"):
                self._create_vector_space_tab()

            with gr.Tab("Chunk Utilization"):
                self._create_chunk_utilization_tab()

            with gr.Tab("Topic Coverage"):
                self._create_topic_coverage_tab()

            with gr.Tab("Domain Insights"):
                self._create_domain_insights_tab()

            with gr.Tab("Справка"):
                self._create_reference_tab()

        return demo

    def _create_run_details_tab(self):
        if not self.runs:
            gr.Markdown(
                "Нет данных о запусках бенчмарков. Запустите бенчмарк для создания записей."
            )
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
                    f"- **Embedding model:** `{run.get('embedding_model', 'N/A')}`",
                    f"- **Judge eval mode:** `{run.get('judge_eval_mode', 'N/A')}`",
                    f"- **Consistency runs:** `{run.get('consistency_runs', 'N/A')}`",
                    f"- **Overall status:** `{run['overall_status']}`",
                ]
            )

            rows = []

            tier_labels = {
                "tier_0": "Tier 0 (Embedding)",
                "tier_1": "Tier 1 (Retrieval)",
                "tier_2": "Tier 2 (Generation)",
                "tier_3": "Tier 3 (End-to-End)",
                "tier_judge": "Tier Judge (Qwen)",
                "tier_judge_pipeline": "Tier Judge Pipeline (Mistral)",
                "tier_ux": "Tier UX",
                "tier_real_users": "Real Users (Retrieval)",
                "utilization_metrics": "Chunk Utilization",
                "topic_coverage_metrics": "Topic Coverage",
                "domain_analysis_metrics": "Domain Analysis",
            }

            for tier_key, tier_label in tier_labels.items():
                tier_metrics = run.get(tier_key, {})
                if not isinstance(tier_metrics, dict) or not tier_metrics:
                    continue
                for metric_name, metric_value in sorted(tier_metrics.items()):
                    baseline = QUALITY_BASELINES.get(metric_name)
                    if isinstance(metric_value, (float, int)):
                        rendered_value = round(float(metric_value), 4)
                        if baseline is None:
                            status = "n/a"
                        else:
                            lower_is_better = metric_name in {
                                "avg_nn_distance",
                                "avg_euclidean_distance",
                                "error_rate",
                                "avg_latency_ms",
                            }
                            if lower_is_better:
                                status = (
                                    "ok" if float(metric_value) <= baseline else "below"
                                )
                            else:
                                status = (
                                    "ok" if float(metric_value) >= baseline else "below"
                                )
                    else:
                        rendered_value = json.dumps(metric_value, ensure_ascii=False)
                        status = "n/a"
                    rows.append(
                        [
                            tier_label,
                            metric_name,
                            rendered_value,
                            "-" if baseline is None else baseline,
                            status,
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
            headers=["Tier", "Metric", "Value", "Baseline", "Status"],
            interactive=False,
            wrap=True,
        )

        executive_summary = gr.Markdown(
            value=f"### Executive Summary\n\n{ordered_runs[0].get('executive_summary', 'Нет данных')}"
        )

        def update_run(selected: str):
            info, rows = get_run_metrics(selected)
            run = next(
                (r for r in ordered_runs if format_run_choice(r) == selected),
                None,
            )
            summary = "Нет данных"
            if run:
                summary = run.get("executive_summary", "Нет данных")
            return info, rows, f"### Executive Summary\n\n{summary}"

        run_selector.change(
            fn=update_run,
            inputs=[run_selector],
            outputs=[run_info, metrics_table, executive_summary],
        )

    def _create_history_tab(self):
        gr.Markdown("### Historical trend for selected metric")
        import plotly.graph_objects as go

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

        def build_history_figure(tier: str, metric: str):
            dates, values = self.get_metric_history(tier, metric)
            fig = go.Figure()
            if not dates or not values:
                fig.update_layout(
                    title="Metric vs Baseline",
                    xaxis_title="Run",
                    yaxis_title="Value",
                    template="plotly_white",
                )
                fig.update_xaxes(showgrid=True)
                fig.update_yaxes(showgrid=True)
                return fig

            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=values,
                    mode="lines+markers",
                    marker={"size": 11},
                    line={"width": 2},
                    name=metric,
                )
            )

            baseline = QUALITY_BASELINES.get(metric)
            if baseline is not None:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=[baseline] * len(dates),
                        mode="lines+markers",
                        marker={"size": 9},
                        line={"width": 1.5, "dash": "dot"},
                        name="Baseline",
                    )
                )

            fig.update_layout(
                title="Metric vs Baseline",
                xaxis_title="Run",
                yaxis_title="Value",
                template="plotly_white",
                legend={"orientation": "h"},
            )
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)
            return fig

        plot = gr.Plot(value=build_history_figure(default_tier, default_metric))

        def update_metric_choices(tier: str):
            options = self._metric_options_for_tier(tier)
            value = options[0] if options else None
            return gr.Dropdown(choices=options, value=value)

        def update_plot(tier: str, metric: str):
            if not metric:
                return go.Figure()
            return build_history_figure(tier, metric)

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
        import plotly.graph_objects as go

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
            "utilization_metrics": "Chunk Utilization",
            "topic_coverage_metrics": "Topic Coverage",
            "domain_analysis_metrics": "Domain Analysis",
        }

        metric_dropdown = gr.Dropdown(
            choices=comparison_metrics,
            value="mrr",
            label="Metric",
        )

        def build_comparison_figure(metric: str):
            fig = go.Figure()
            longest_dates: List[str] = []
            for tier_name, label in tier_labels.items():
                if tier_name not in METRICS_BY_TIER:
                    continue
                if metric not in METRICS_BY_TIER[tier_name]:
                    continue
                dates, values = self.get_metric_history(tier_name, metric)
                if not dates:
                    continue
                if len(dates) > len(longest_dates):
                    longest_dates = dates
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=values,
                        mode="lines+markers",
                        marker={"size": 10},
                        line={"width": 2},
                        name=label,
                    )
                )

            baseline = QUALITY_BASELINES.get(metric)
            if baseline is not None and longest_dates:
                fig.add_trace(
                    go.Scatter(
                        x=longest_dates,
                        y=[baseline] * len(longest_dates),
                        mode="lines+markers",
                        marker={"size": 8},
                        line={"dash": "dot", "width": 1.5},
                        name="Baseline",
                    )
                )

            fig.update_layout(
                title="Tier comparison",
                xaxis_title="Run",
                yaxis_title="Value",
                template="plotly_white",
                legend={"orientation": "h"},
            )
            fig.update_xaxes(showgrid=True)
            fig.update_yaxes(showgrid=True)
            return fig

        plot = gr.Plot(value=build_comparison_figure("mrr"))

        def update_comparison_plot(metric: str):
            return build_comparison_figure(metric)

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
                        run["timestamp_readable"],
                        run["git_branch"] if run.get("git_branch") else "N/A",
                        run.get("dataset_type", "synthetic"),
                        round(_safe_float(t1.get("mrr")), 2),
                        round(_safe_float(t1.get("hit_rate@5")), 2),
                        round(_safe_float(t2.get("avg_faithfulness")), 2),
                        round(_safe_float(t2.get("avg_answer_relevance")), 2),
                        round(_safe_float(t3.get("avg_e2e_score")), 2),
                        round(_safe_float(tjp.get("accuracy")), 2) if tjp else "-",
                        run["overall_status"],
                    ]
                )
            return rows

        headers = [
            "Time",
            "Git Branch",
            "Dataset Type",
            "MRR",
            "Hit Rate@5",
            "Faithfulness",
            "Relevance",
            "E2E",
            "Judge Pipeline Accuracy",
            "Overall Status",
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
                return "", pd.DataFrame()

            run = next(
                (item for item in ordered_runs if format_run_choice(item) == selected),
                None,
            )
            if run is None:
                return "Не найден выбранный запуск", pd.DataFrame()

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
                return info, pd.DataFrame()

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
                    pd.DataFrame(),
                )

            try:
                import json

                with open(dataset_path, "r", encoding="utf-8") as f:
                    dataset = json.load(f)

                all_keys = set()
                for item in dataset:
                    if isinstance(item, dict):
                        all_keys.update(item.keys())

                text_priority = [
                    "question",
                    "ground_truth_answer",
                    "chunk_text",
                    "confluence_url",
                    "relevant_urls",
                    "notes",
                    "source",
                    "question_source",
                ]
                numeric_priority = [
                    "chunk_id",
                    "question_answer_id",
                    "user_score",
                    "is_relevant_chunk_matched",
                    "is_question_ok",
                    "is_answer_ok",
                    "is_chunk_ok",
                ]

                ordered_keys = [key for key in text_priority if key in all_keys]
                ordered_keys.extend(
                    key
                    for key in sorted(all_keys)
                    if key not in ordered_keys and key not in numeric_priority
                )
                ordered_keys.extend(
                    key
                    for key in numeric_priority
                    if key in all_keys and key not in ordered_keys
                )

                preview_rows = []
                for row in dataset[:30]:
                    if not isinstance(row, dict):
                        continue
                    rendered_row = {}
                    for key in ordered_keys:
                        value = row.get(key)
                        if isinstance(value, (dict, list)):
                            rendered_row[key] = json.dumps(value, ensure_ascii=False)
                        else:
                            rendered_row[key] = value
                    preview_rows.append(rendered_row)

                preview = pd.DataFrame(preview_rows)
                meta = (
                    f"**Dataset:** `{dataset_file}`\n"
                    f"\n**Mode:** `{run.get('dataset_type', 'synthetic')}`\n"
                    f"\n**Rows:** {len(dataset)}\n"
                    f"\n**Linked run:** `{run['timestamp_readable']}`\n"
                    f"\n**Поля ({len(ordered_keys)}):** `{', '.join(ordered_keys)}`"
                )

                return meta, preview
            except Exception as error:
                return (
                    f"**Dataset:** `{dataset_file}`\n\nОшибка чтения файла: `{error}`",
                    pd.DataFrame(),
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
            interactive=False,
            wrap=True,
        )

        def update_on_filter_change(filter_mode: str):
            options = get_run_choices(filter_mode)
            if not options:
                return (
                    gr.Dropdown(choices=[], value=None),
                    "",
                    pd.DataFrame(),
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

    def _create_vector_space_tab(self):
        gr.Markdown("### Визуализация векторного пространства чанков")

        with gr.Row():
            limit_slider = gr.Slider(
                minimum=100,
                maximum=10000,
                value=3000,
                step=100,
                label="Количество чанков",
            )
            dim_radio = gr.Radio(["2D", "3D"], value="2D", label="Размерность")
            color_by = gr.Dropdown(
                choices=["section", "cluster", "chunk_id"],
                value="section",
                label="Раскраска",
            )

        output_html = gr.HTML(label="Векторное пространство")
        visualize_btn = gr.Button("Визуализировать")

        def run_visualization(limit: int, dimension: str, color: str):
            try:
                from benchmarks.visualize_vector_space import (
                    _load_chunk_embeddings,
                    visualize_embeddings,
                )

                dim = 3 if dimension == "3D" else 2
                output_path = (
                    self.reports_dir
                    / f"vector_space_dashboard_{dim}d_{int(limit)}_{color}.html"
                )
                embeddings, metadata = _load_chunk_embeddings(int(limit))
                html_path = visualize_embeddings(
                    embeddings=embeddings,
                    metadata=metadata,
                    output_path=str(output_path),
                    dim=dim,
                    color_by=color,
                )
                with open(html_path, "r", encoding="utf-8") as file:
                    return file.read()
            except Exception as error:
                return (
                    "<div style='padding:12px;border:1px solid #ddd;'>"
                    f"Ошибка визуализации: {error}</div>"
                )

        visualize_btn.click(
            fn=run_visualization,
            inputs=[limit_slider, dim_radio, color_by],
            outputs=[output_html],
        )

    def _create_chunk_utilization_tab(self):
        gr.Markdown("### Анализ использования чанков")
        if not self.runs:
            gr.Markdown("Нет запусков для анализа utilization.")
            return

        ordered_runs = list(reversed(self.runs))

        def format_run_choice(run: Dict) -> str:
            return (
                f"{run['timestamp_readable']} | {run['dataset_type']} | "
                f"{run['git_commit_hash'][:7]}"
            )

        run_choices = [format_run_choice(run) for run in ordered_runs]
        run_selector = gr.Dropdown(
            choices=run_choices,
            value=run_choices[0],
            label="Выберите запуск",
        )

        def build_utilization_view(selected: str):
            import plotly.express as px
            import plotly.graph_objects as go

            run = next(
                (item for item in ordered_runs if format_run_choice(item) == selected),
                None,
            )
            if run is None:
                return "Запуск не найден", go.Figure(), go.Figure()

            metrics = run.get("utilization_metrics") or {}
            if not metrics:
                return "В запуске нет utilization_metrics", go.Figure(), go.Figure()

            used = int(metrics.get("used_chunks", 0))
            total = int(metrics.get("total_chunks", 0))
            unused = int(metrics.get("unused_chunks", max(total - used, 0)))
            rate = _safe_float(metrics.get("utilization_rate"))

            pie = px.pie(
                names=["used", "unused"],
                values=[used, unused],
                title="Utilization",
            )

            used_ids = metrics.get("used_chunk_ids") or []
            top_ids = used_ids[:20]
            top = px.bar(
                x=list(range(len(top_ids))),
                y=top_ids,
                labels={"x": "Позиция", "y": "chunk_id"},
                title="Первые 20 использованных chunk_id",
            )

            text = (
                f"**Utilization rate:** `{rate:.4f}`\n"
                f"\n**Used chunks:** `{used}` из `{total}`"
            )
            return text, pie, top

        initial_text, initial_pie, initial_top = build_utilization_view(run_choices[0])
        summary = gr.Markdown(value=initial_text)
        pie_plot = gr.Plot(
            value=initial_pie,
            label="Использованные vs неиспользованные",
        )
        top_plot = gr.Plot(
            value=initial_top,
            label="Топ использованных chunk_id",
        )

        run_selector.change(
            fn=build_utilization_view,
            inputs=[run_selector],
            outputs=[summary, pie_plot, top_plot],
        )

    def _create_topic_coverage_tab(self):
        gr.Markdown("### Анализ покрытия тем")
        if not self.runs:
            gr.Markdown("Нет запусков для анализа topic coverage.")
            return

        ordered_runs = list(reversed(self.runs))

        def format_run_choice(run: Dict) -> str:
            return (
                f"{run['timestamp_readable']} | {run['dataset_type']} | "
                f"{run['git_commit_hash'][:7]}"
            )

        run_choices = [format_run_choice(run) for run in ordered_runs]
        run_selector = gr.Dropdown(
            choices=run_choices,
            value=run_choices[0],
            label="Выберите запуск",
        )

        def build_topic_view(selected: str):
            import plotly.express as px
            import plotly.graph_objects as go

            run = next(
                (item for item in ordered_runs if format_run_choice(item) == selected),
                None,
            )
            if run is None:
                return "Запуск не найден", go.Figure(), pd.DataFrame()

            metrics = run.get("topic_coverage_metrics") or {}
            topics = metrics.get("topic_coverage") or []
            if not metrics or not topics:
                return (
                    "В запуске нет topic_coverage_metrics",
                    go.Figure(),
                    pd.DataFrame(),
                )

            table = pd.DataFrame(topics)
            plot = px.bar(
                table,
                x="topic_id",
                y="unique_chunks",
                hover_data=["question_count", "unique_urls"],
                title="Уникальные чанки по темам",
            )
            text = (
                f"**Тем:** `{metrics.get('n_topics', 0)}`\n"
                f"\n**Вопросов:** `{metrics.get('total_questions', 0)}`\n"
                f"\n**Среднее число чанков на тему:** "
                f"`{_safe_float(metrics.get('avg_chunks_per_topic')):.2f}`"
            )
            return text, plot, table

        initial_text, initial_plot, initial_table = build_topic_view(run_choices[0])
        summary = gr.Markdown(value=initial_text)
        coverage_plot = gr.Plot(value=initial_plot, label="Покрытие по темам")
        coverage_table = gr.Dataframe(
            value=initial_table,
            interactive=False,
            wrap=True,
        )

        run_selector.change(
            fn=build_topic_view,
            inputs=[run_selector],
            outputs=[summary, coverage_plot, coverage_table],
        )

    def _create_domain_insights_tab(self):
        gr.Markdown("### Анализ предметной области на real-user данных")
        if not self.runs:
            gr.Markdown("Нет запусков для domain analysis.")
            return

        ordered_runs = list(reversed(self.runs))

        def format_run_choice(run: Dict) -> str:
            return (
                f"{run['timestamp_readable']} | {run['dataset_type']} | "
                f"{run['git_commit_hash'][:7]}"
            )

        run_choices = [format_run_choice(run) for run in ordered_runs]
        run_selector = gr.Dropdown(
            choices=run_choices,
            value=run_choices[0],
            label="Выберите запуск",
        )

        summary = gr.Markdown()
        score_table = gr.Dataframe(interactive=False, wrap=True)
        token_table = gr.Dataframe(interactive=False, wrap=True)

        def build_domain_view(selected: str):
            run = next(
                (item for item in ordered_runs if format_run_choice(item) == selected),
                None,
            )
            if run is None:
                return "Запуск не найден", pd.DataFrame(), pd.DataFrame()

            metrics = run.get("domain_analysis_metrics") or {}
            if not metrics:
                return (
                    "В запуске нет domain_analysis_metrics",
                    pd.DataFrame(),
                    pd.DataFrame(),
                )

            summary_text = (
                f"**Всего вопросов:** `{metrics.get('total_questions', 0)}`\n"
                f"\n**С ответами:** `{metrics.get('with_answers', 0)}`\n"
                f"\n**Без ответов:** `{metrics.get('without_answers', 0)}`\n"
                f"\n**Средняя длина вопроса:** `{_safe_float(metrics.get('avg_question_length')):.2f}`"
            )

            score_distribution = metrics.get("score_distribution", {})
            score_df = pd.DataFrame(
                [
                    {"score": str(score), "count": count}
                    for score, count in score_distribution.items()
                ]
            )

            top_tokens = metrics.get("top_tokens_all", [])
            token_df = pd.DataFrame(top_tokens)

            return summary_text, score_df, token_df

        initial_summary, initial_scores, initial_tokens = build_domain_view(
            run_choices[0]
        )
        summary = gr.Markdown(value=initial_summary)
        score_table = gr.Dataframe(value=initial_scores, interactive=False, wrap=True)
        token_table = gr.Dataframe(value=initial_tokens, interactive=False, wrap=True)

        run_selector.change(
            fn=build_domain_view,
            inputs=[run_selector],
            outputs=[summary, score_table, token_table],
        )

    def _create_reference_tab(self):
        gr.Markdown("# Справка по системе оценки RAG")
        gr.Markdown("Содержит полный список метрик, используемых в дашборде и отчётах.")

        rows = []
        for tier, metrics in METRICS_BY_TIER.items():
            for metric in metrics:
                baseline = QUALITY_BASELINES.get(metric)
                rows.append(
                    {
                        "Tier": tier,
                        "Metric": metric,
                        "Baseline": "-" if baseline is None else baseline,
                    }
                )

        table_df = pd.DataFrame(rows)
        gr.HTML("<div style='text-align:center'><h3>Все метрики</h3></div>")
        gr.Dataframe(value=table_df, interactive=False, wrap=False)


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
