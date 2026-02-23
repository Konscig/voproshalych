"""Генератор аналитических отчётов для бенчмарков RAG-системы."""

from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, Optional


QUALITY_BASELINES = {
    "mrr": 0.8,
    "hit_rate@1": 0.7,
    "hit_rate@5": 0.9,
    "hit_rate@10": 0.95,
    "avg_faithfulness": 4.5,
    "avg_answer_relevance": 4.2,
    "avg_answer_correctness": 4.2,
    "avg_e2e_score": 4.2,
    "avg_semantic_similarity": 0.85,
    "response_exact_match_rate": 0.7,
    "response_semantic_consistency": 0.85,
    "retrieval_consistency": 0.95,
}


class ReportGenerator:
    """Генератор отчётов бенчмарков в формате Markdown."""

    @staticmethod
    def _run_git_command(args: list[str]) -> Optional[str]:
        """Выполнить git-команду и вернуть trimmed stdout."""
        try:
            result = subprocess.run(
                ["git", *args],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip() or None
        except Exception:
            return None

    @staticmethod
    def _resolve_git_dir() -> Optional[Path]:
        """Найти путь к git metadata каталогу."""
        git_path = Path(".git")
        if git_path.is_dir():
            return git_path
        if git_path.is_file():
            content = git_path.read_text(encoding="utf-8").strip()
            prefix = "gitdir:"
            if content.startswith(prefix):
                rel_path = content[len(prefix) :].strip()
                return (git_path.parent / rel_path).resolve()
        return None

    @classmethod
    def _read_branch_from_head(cls) -> Optional[str]:
        """Прочитать имя ветки из .git/HEAD без git CLI."""
        git_dir = cls._resolve_git_dir()
        if git_dir is None:
            return None
        head_file = git_dir / "HEAD"
        if not head_file.exists():
            return None
        head_value = head_file.read_text(encoding="utf-8").strip()
        if head_value.startswith("ref:"):
            ref_path = head_value.split(" ", maxsplit=1)[1]
            return ref_path.split("/")[-1]
        return None

    @classmethod
    def _read_commit_from_head(cls) -> Optional[str]:
        """Прочитать short hash коммита из .git без git CLI."""
        git_dir = cls._resolve_git_dir()
        if git_dir is None:
            return None
        head_file = git_dir / "HEAD"
        if not head_file.exists():
            return None

        head_value = head_file.read_text(encoding="utf-8").strip()
        if head_value.startswith("ref:"):
            ref_path = head_value.split(" ", maxsplit=1)[1]
            ref_file = git_dir / ref_path
            if ref_file.exists():
                return ref_file.read_text(encoding="utf-8").strip()[:8]
            return None
        if len(head_value) >= 8:
            return head_value[:8]
        return None

    @classmethod
    def get_run_metadata(cls) -> Dict[str, str]:
        """Собрать метаданные запуска для отчёта и БД."""
        branch = os.getenv("BENCHMARK_GIT_BRANCH")
        if not branch:
            branch = cls._run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])
        if not branch:
            branch = os.getenv("CI_COMMIT_REF_NAME") or cls._read_branch_from_head()

        commit = os.getenv("BENCHMARK_GIT_COMMIT_HASH")
        if not commit:
            commit = cls._run_git_command(["rev-parse", "--short", "HEAD"])
        if not commit:
            commit = os.getenv("CI_COMMIT_SHORT_SHA") or cls._read_commit_from_head()

        author = (
            os.getenv("BENCHMARK_RUN_AUTHOR")
            or os.getenv("GIT_AUTHOR_NAME")
            or cls._run_git_command(["config", "user.name"])
            or os.getenv("USER")
        )

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "git_branch": branch or "unknown",
            "git_commit_hash": commit or "unknown",
            "run_author": author or "unknown",
        }

    def evaluate_overall_status(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Оценить общий статус качества на основе baseline-порогов."""
        failed = 0
        checked = 0

        for tier_data in results.values():
            if not isinstance(tier_data, dict):
                continue
            for metric, baseline in QUALITY_BASELINES.items():
                value = tier_data.get(metric)
                if value is None:
                    continue
                checked += 1
                if float(value) < baseline:
                    failed += 1

        if checked == 0:
            return "insufficient_data"
        if failed == 0:
            return "good"
        if failed <= 2:
            return "attention"
        return "critical"

    def _tier_summary(self, tier_name: str, tier_metrics: Dict[str, Any]) -> str:
        """Сформировать интерпретацию конкретного Tier."""
        if tier_name == "tier_1":
            mrr = float(tier_metrics.get("mrr", 0.0))
            hit5 = float(tier_metrics.get("hit_rate@5", 0.0))
            if mrr >= 0.8 and hit5 >= 0.9:
                return "Поиск стабилен: релевантные документы находятся быстро."
            if mrr >= 0.6:
                return "Поиск приемлемый, но есть потенциал улучшить ранжирование."
            return "Поиск нестабилен: релевантные документы часто ранжируются низко."

        if tier_name == "tier_2":
            faithfulness = float(tier_metrics.get("avg_faithfulness", 0.0))
            relevance = float(tier_metrics.get("avg_answer_relevance", 0.0))
            if faithfulness >= 4.5 and relevance >= 4.2:
                return "Генерация точная и хорошо отвечает на пользовательский запрос."
            if faithfulness < 4.0:
                return (
                    "Генерация страдает от галлюцинаций: требуется ревизия "
                    "промпта и/или контекста."
                )
            return "Генерация рабочая, но релевантность можно усилить инструкциями."

        if tier_name == "tier_3":
            e2e = float(tier_metrics.get("avg_e2e_score", 0.0))
            sim = float(tier_metrics.get("avg_semantic_similarity", 0.0))
            if e2e >= 4.2 and sim >= 0.85:
                return "End-to-end качество высокое и стабильно для продового сценария."
            if e2e >= 3.5:
                return (
                    "End-to-end качество умеренное: пользователю полезно, но неровно."
                )
            return (
                "End-to-end качество низкое: нужен полный аудит retrieval+generation."
            )

        return "Недостаточно данных для интерпретации."

    def generate_executive_summary(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Сформировать бизнес-ориентированное резюме по всем Tier."""
        blocks = ["## Executive Summary", ""]

        for tier_name in ("tier_1", "tier_2", "tier_3"):
            tier_metrics = results.get(tier_name)
            if not isinstance(tier_metrics, dict):
                continue
            blocks.append(
                f"- **{tier_name.upper()}**: {self._tier_summary(tier_name, tier_metrics)}"
            )

        overall = self.evaluate_overall_status(results)
        blocks.append(
            f"- **Общий статус**: `{overall}` (оценка относительно baseline-порогов)."
        )

        return "\n".join(blocks)

    def generate_benchmark_report(
        self,
        results: Dict[str, Dict[str, Any]],
        dataset_name: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> str:
        """Сгенерировать подробный Markdown-отчёт по всем уровням."""
        metadata = metadata or self.get_run_metadata()

        lines = [
            "# RAG Benchmark Report",
            "",
            f"**Время запуска:** {metadata['timestamp']}",
            f"**Датасет:** {dataset_name}",
            f"**Git ветка:** `{metadata['git_branch']}`",
            f"**Коммит:** `{metadata['git_commit_hash']}`",
            f"**Автор запуска:** {metadata['run_author']}",
            "",
            self.generate_executive_summary(results),
            "",
            "## Метрики по уровням",
            "",
        ]

        for tier_name, tier_results in results.items():
            if not isinstance(tier_results, dict):
                continue
            lines.append(f"### {tier_name.upper()}")
            lines.append("")
            lines.append("| Метрика | Значение | Baseline | Статус |")
            lines.append("|---------|----------|----------|--------|")

            for key, value in tier_results.items():
                if key == "tier":
                    continue

                if isinstance(value, float):
                    rendered_value = f"{value:.4f}"
                    baseline = QUALITY_BASELINES.get(key)
                    if baseline is None:
                        lines.append(f"| {key} | {rendered_value} | - | n/a |")
                    else:
                        status = "ok" if value >= baseline else "below_target"
                        lines.append(
                            f"| {key} | {rendered_value} | {baseline:.4f} | {status} |"
                        )
                else:
                    lines.append(f"| {key} | {value} | - | n/a |")

            lines.append("")

        lines.extend(
            [
                "## Как читать метрики",
                "",
                "- **Tier 1 (Retrieval):** оценивает, насколько хорошо поиск находит релевантные документы.",
                "- **Tier 2 (Generation):** оценивает качество и правдивость ответа LLM на найденном контексте.",
                "- **Tier 3 (E2E):** оценивает итоговый пользовательский опыт от вопроса до ответа.",
                "",
                "- **MRR:** средняя позиция первого релевантного результата; чем ближе к 1.0, тем лучше ранжирование.",
                "- **Faithfulness 4.5/5:** ответ в среднем почти не искажает факты из контекста.",
                "- **Answer Relevance:** насколько ответ действительно решает вопрос пользователя.",
                "- **E2E Score:** интегральная оценка полезности финального ответа.",
            ]
        )

        return "\n".join(lines) + "\n"
