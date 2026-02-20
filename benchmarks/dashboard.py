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
    "avg_faithfulness": 4.5,
    "avg_answer_relevance": 4.2,
    "avg_e2e_score": 4.2,
    "avg_semantic_similarity": 0.85,
    "avg_rouge1_f": 0.50,
    "avg_rougeL_f": 0.45,
    "avg_bleu": 30.0,
    "consistency_score": 0.90,
    "error_rate": 0.05,
    "cache_hit_rate": 0.40,
    "context_preservation": 0.70,
}

QUALITY_BASELINES = {
    "avg_intra_cluster_sim": 0.85,
    "avg_inter_cluster_dist": 0.30,
    "silhouette_score": 0.50,
    "mrr": 0.8,
    "hit_rate@1": 0.7,
    "hit_rate@5": 0.9,
    "hit_rate@10": 0.95,
    "avg_faithfulness": 4.5,
    "avg_answer_relevance": 4.2,
    "avg_e2e_score": 4.2,
    "avg_semantic_similarity": 0.85,
    "consistency_score": 0.90,
    "error_rate": 0.05,
    "cache_hit_rate": 0.40,
    "context_preservation": 0.70,
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
                    "overall_status": record.overall_status,
                    "tier_0": normalize_metrics(record.tier_0_metrics),
                    "tier_1": normalize_metrics(record.tier_1_metrics),
                    "tier_2": normalize_metrics(record.tier_2_metrics),
                    "tier_3": normalize_metrics(record.tier_3_metrics),
                    "tier_judge": normalize_metrics(record.tier_judge_metrics),
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
                            f"- Judge model: `{os.getenv('BENCHMARKS_JUDGE_MODEL') or os.getenv('JUDGE_MODEL') or Config.JUDGE_MODEL or 'unknown'}`",
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
                    f"- Dataset type: `{latest_run.get('dataset_type', 'synthetic')}`",
                    f"- Judge model: `{os.getenv('BENCHMARKS_JUDGE_MODEL') or os.getenv('JUDGE_MODEL') or Config.JUDGE_MODEL or 'unknown'}`",
                    f"- Overall status: `{latest_run['overall_status']}`",
                ]
            )
        )

        summary_rows = []
        summary_rows.append(
            {
                "Tier": "Tier 1 (Retrieval)",
                "HitRate@5": round(
                    _safe_float(latest_run.get("tier_1", {}).get("hit_rate@5")), 4
                ),
                "MRR": round(_safe_float(latest_run.get("tier_1", {}).get("mrr")), 4),
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
                "Faithfulness": round(
                    _safe_float(latest_run.get("tier_2", {}).get("avg_faithfulness")),
                    4,
                ),
                "Relevance": round(
                    _safe_float(
                        latest_run.get("tier_2", {}).get("avg_answer_relevance")
                    ),
                    4,
                ),
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
                "E2E Score": round(
                    _safe_float(latest_run.get("tier_3", {}).get("avg_e2e_score")), 4
                ),
            }
        )

        if latest_run.get("tier_real_users"):
            summary_rows.append(
                {
                    "Tier": "Real Users (Retrieval)",
                    "HitRate@5": "-",
                    "MRR": round(
                        _safe_float(latest_run.get("tier_real_users", {}).get("mrr")),
                        4,
                    ),
                    "Faithfulness": "-",
                    "Relevance": "-",
                    "E2E Score": "-",
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

        default_tier = "tier_1"
        default_metric = "mrr"
        if not self.get_metric_history(default_tier, default_metric)[0]:
            for candidate_tier in ["tier_1", "tier_2", "tier_3", "tier_real_users"]:
                for candidate_metric in self._metric_options_for_tier(candidate_tier):
                    if self.get_metric_history(candidate_tier, candidate_metric)[0]:
                        default_tier = candidate_tier
                        default_metric = candidate_metric
                        break
                if default_metric != "mrr" or default_tier != "tier_1":
                    break

        with gr.Row():
            tier_dropdown = gr.Dropdown(
                choices=["tier_1", "tier_2", "tier_3", "tier_real_users"],
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

        metric_dropdown = gr.Dropdown(
            choices=comparison_metrics,
            value="mrr",
            label="Metric",
        )

        initial_comparison_rows: List[Dict[str, str | float]] = []
        for tier_name, label in {
            "tier_1": "Tier 1",
            "tier_2": "Tier 2",
            "tier_3": "Tier 3",
            "tier_real_users": "Real Users",
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
                "tier_real_users": "Real Users",
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
        mode_filter = gr.Dropdown(
            choices=["all", "synthetic", "manual", "real-users"],
            value="all",
            label="Фильтр по типу датасета",
        )
        table = gr.Dataframe(
            value=[],
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

        def build_registry_rows(filter_mode: str):
            rows = []
            for run in reversed(self.runs):
                run_mode = run.get("dataset_type", "synthetic")
                if filter_mode != "all" and run_mode != filter_mode:
                    continue
                tier_1 = run.get("tier_1", {})
                tier_2 = run.get("tier_2", {})
                tier_3 = run.get("tier_3", {})
                rows.append(
                    {
                        "Timestamp": str(run["timestamp_readable"]),
                        "Branch / Commit": (
                            f"{run['git_branch']} / {run['git_commit_hash']}"
                        ),
                        "T1: HitRate@5": round(
                            _safe_float(tier_1.get("hit_rate@5")), 4
                        ),
                        "T1: MRR": round(_safe_float(tier_1.get("mrr")), 4),
                        "T2: Faithfulness": round(
                            _safe_float(tier_2.get("avg_faithfulness")), 4
                        ),
                        "T2: Relevance": round(
                            _safe_float(tier_2.get("avg_answer_relevance")), 4
                        ),
                        "T3: E2E Score": round(
                            _safe_float(tier_3.get("avg_e2e_score")), 4
                        ),
                    }
                )
            return rows

        mode_filter.change(
            fn=build_registry_rows, inputs=[mode_filter], outputs=[table]
        )
        table.value = build_registry_rows("all")

    def _create_run_dataset_tab(self):
        mode_filter = gr.Dropdown(
            choices=["all", "synthetic", "manual", "real-users"],
            value="all",
            label="Тип запуска",
        )

        ordered_runs = list(reversed(self.runs))

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

        run_choices = get_run_choices("all")

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

        def update_dataset_preview(selected: str):
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
            # TODO: вынести датасеты в отдельную таблицу benchmark_datasets,
            # чтобы хранить snapshot записи вместе с запуском в БД.
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

        def update_run_choices(filter_mode: str):
            options = get_run_choices(filter_mode)
            if not options:
                return gr.Dropdown(choices=[], value=None), "", []
            meta, preview = update_dataset_preview(options[0])
            return gr.Dropdown(choices=options, value=options[0]), meta, preview

        mode_filter.change(
            fn=update_run_choices,
            inputs=[mode_filter],
            outputs=[run_selector, dataset_meta, dataset_preview],
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
            r"""
### Методология оценки RAG

- **Tier 1 (Retrieval):** оценка ранжирования чанков в векторном поиске.
- **Tier 2 (Generation):** оценка качества ответа при релевантном контексте.
- **Tier 3 (End-to-End):** оценка полного пайплайна retrieval + generation.
- **Real Users:** retrieval-метрики на реальных вопросах пользователей.

### Retrieval-метрики

Для множества запросов $Q$:

$$
\mathrm{HitRate@K} = \frac{1}{|Q|} \sum_{q \in Q} \mathbf{1}[\exists d \in \mathrm{TopK}(q): d \in G_q]
$$

$$
\mathrm{Recall@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\mathrm{TopK}(q) \cap G_q|}{|G_q|}
$$

$$
\mathrm{Precision@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\mathrm{TopK}(q) \cap G_q|}{K}
$$

$$
\mathrm{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\mathrm{rank}_q}
$$

$$
\mathrm{NDCG@K} = \frac{1}{|Q|} \sum_{q \in Q} \frac{\mathrm{DCG@K}(q)}{\mathrm{IDCG@K}(q)}
$$

### Generation и End-to-End (LLM-as-a-Judge)

Судья выставляет оценки $s_i \in \{1,2,3,4,5\}$.

$$
\mathrm{avg\_faithfulness} = \frac{1}{N} \sum_{i=1}^{N} s_i
$$

$$
\mathrm{avg\_answer\_relevance} = \frac{1}{N} \sum_{i=1}^{N} s_i
$$

$$
\mathrm{avg\_e2e\_score} = \frac{1}{N} \sum_{i=1}^{N} s_i
$$

Шкала 1-5 трактуется как **ordinal scale** (упорядоченная, не строго линейная).

Для semantic similarity используется косинусная близость эмбеддингов:

$$
\mathrm{cos\_sim}(u, v) = \frac{u \cdot v}{\|u\|\,\|v\|}
$$

### Baseline-пороги

- Tier 1: `MRR >= 0.80`, `HitRate@5 >= 0.90`, `HitRate@10 >= 0.95`
- Tier 2: `Faithfulness >= 4.5`, `Answer Relevance >= 4.2`
- Tier 3: `E2E Score >= 4.2`, `Semantic Similarity >= 0.85`
- Real Users: `MRR`, `Recall@K`, `NDCG@K` анализируются в динамике по релизам
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
