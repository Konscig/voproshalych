"""CLI скрипт для запуска комплексных бенчмарков RAG-системы.

Использование:
    python run_comprehensive_benchmark.py --tier all --limit 50
    python run_comprehensive_benchmark.py --tier 1 --dataset benchmarks/data/dataset_20260216_143000.json
"""

import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sqlalchemy import func

sys.path.insert(0, str(Path(__file__).parent.parent))

from qa.config import Config
from qa.database import (
    Chunk,
    create_engine,
)
from qa.database import QuestionAnswer
from benchmarks.analyze_chunk_utilization import analyze_chunk_utilization
from benchmarks.analyze_real_users_domain import analyze_real_users_domain
from benchmarks.analyze_topic_coverage import analyze_topic_coverage
from benchmarks.models.real_queries_benchmark import RealQueriesBenchmark
from benchmarks.models.rag_benchmark import RAGBenchmark
from benchmarks.utils.llm_judge import LLMJudge
from benchmarks.utils.report_generator import ReportGenerator
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

# Явно загружаем .env.docker для локального использования (для разработки)
# В Docker используется .env.docker автоматически через docker compose
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env.docker")


def check_prerequisites(
    engine,
    mode: str,
    dataset_path: Optional[str] = None,
) -> bool:
    """Проверить предварительные условия для запуска бенчмарков.

    Args:
        engine: Движок базы данных
        mode: Режим запуска (synthetic/manual/real-users)
        dataset_path: Путь к датасету

    Returns:
        True если все условия выполнены
    """
    from sqlalchemy import select
    from sqlalchemy.orm import Session

    logger.info("Проверка предварительных условий...")

    with Session(engine) as session:
        total_chunks = session.scalars(select(Chunk)).all()
        chunks_with_embeddings = [
            c for c in total_chunks if c.embedding is not None and len(c.embedding) > 0
        ]

        logger.info(f"Всего чанков: {len(total_chunks)}")
        logger.info(f"Чанков с эмбеддингами: {len(chunks_with_embeddings)}")

        if len(chunks_with_embeddings) == 0:
            logger.error(
                "❌ Нет чанков с эмбеддингами! "
                "Запустите: python benchmarks/generate_embeddings.py --chunks"
            )
            return False

    if mode in {"synthetic", "manual"} and (
        not dataset_path or not os.path.exists(dataset_path)
    ):
        logger.error(
            f"❌ Датасет не найден: {dataset_path}\n"
            "Сгенерируйте датасет: "
            "python benchmarks/generate_dataset.py --max-questions 50"
        )
        return False

    if mode == "real-users":
        with Session(engine) as session:
            total_real = session.scalar(
                select(func.count(QuestionAnswer.id)).where(
                    QuestionAnswer.embedding.isnot(None)
                )
            )
        if not total_real:
            logger.error("❌ В таблице question_answer нет вопросов с эмбеддингами")
            return False

    logger.info("✅ Все предварительные условия выполнены")
    return True


def load_dataset(dataset_path: str, limit: Optional[int] = None) -> list:
    """Загрузить датасет из JSON файла.

    Args:
        dataset_path: Путь к файлу датасета
        limit: Ограничение количества записей

    Returns:
        Список записей датасета
    """
    logger.info(f"Загрузка датасета из {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]

    logger.info(f"Загружено {len(dataset)} записей")

    return dataset


def load_judge_pipeline_dataset(
    dataset_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> list:
    """Загрузить датасет для Tier Judge Pipeline.

    Это специальный датасет с парами (question, answer, ground_truth_show)
    для тестирования production judge (Mistral).

    Args:
        dataset_path: Путь к файлу датасета (по умолчанию ищет dataset_judge_pipeline_*.json)
        limit: Ограничение количества записей

    Returns:
        Список записей с полями question, answer, context, ground_truth_show
    """
    if dataset_path is None:
        candidates = sorted(glob.glob("benchmarks/data/dataset_judge_pipeline_*.json"))
        if candidates:
            dataset_path = candidates[-1]
            logger.info("Используем Judge Pipeline датасет: %s", dataset_path)
        else:
            logger.warning("Judge Pipeline датасет не найден")
            return []

    logger.info(f"Загрузка Judge Pipeline датасета из {dataset_path}")

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        if limit:
            dataset = dataset[:limit]

        logger.info(f"Загружено {len(dataset)} записей для Judge Pipeline")
        return dataset
    except FileNotFoundError:
        logger.error(f"Датасет не найден: {dataset_path}")
        return []


def resolve_dataset_path(dataset_path: str) -> str:
    """Разрешить путь к датасету, учитывая versioned файлы по умолчанию."""
    if dataset_path != "benchmarks/data/golden_dataset_synthetic.json":
        return dataset_path

    if os.path.exists(dataset_path):
        return dataset_path

    # Ищем датасеты, исключая файлы с ошибками (_errors) и специальные датасеты
    candidates = sorted(glob.glob("benchmarks/data/dataset_*.json"))
    candidates = [
        c for c in candidates if "_errors" not in c and "judge_pipeline" not in c
    ]
    if candidates:
        latest = candidates[-1]
        logger.info("Используем последний versioned датасет: %s", latest)
        return latest

    return dataset_path


def resolve_manual_dataset_path(
    manual_dataset: Optional[str], dataset_path: str
) -> str:
    """Определить путь к manual dataset."""
    if manual_dataset:
        return manual_dataset
    return dataset_path


def run_benchmark(
    engine,
    encoder: SentenceTransformer,
    judge: Optional[LLMJudge],
    dataset: Optional[list],
    tier: str,
    mode: str,
    top_k: int = 10,
    real_score_filter: int | None = 5,
    real_limit: int = 500,
    real_latest: bool = False,
    judge_pipeline_dataset_path: Optional[str] = None,
    analyze_utilization: bool = False,
    analyze_topics: bool = False,
    utilization_questions_source: str = "synthetic",
    utilization_question_limit: int = 500,
    utilization_top_k: int = 10,
    topics_question_limit: int = 2000,
    topics_count: int = 20,
    topics_top_k: int = 5,
    consistency_runs: int = 1,
    analyze_domain: bool = False,
    domain_limit: int = 5000,
):
    """Запустить бенчмарк.

    Args:
        engine: Движок базы данных
        encoder: Модель для генерации эмбеддингов
        judge: LLM-судья
        dataset: Датасет
        tier: Уровень бенчмарка (1, 2, 3, all)
        mode: Режим запуска (synthetic/manual/real-users)
        top_k: Количество результатов для поиска (Tier 1)

    Returns:
        Результаты бенчмарка
    """
    if mode == "real-users":
        real_benchmark = RealQueriesBenchmark(engine, encoder)
        results: dict[str, Any] = {
            "tier_real_users": real_benchmark.run(
                top_k=top_k,
                score_filter=real_score_filter,
                limit=real_limit,
                latest=real_latest,
            )
        }
        if analyze_utilization:
            results["utilization_metrics"] = analyze_chunk_utilization(
                engine=engine,
                encoder=encoder,
                questions_source=utilization_questions_source,
                question_limit=utilization_question_limit,
                top_k=utilization_top_k,
            )
        if analyze_topics:
            results["topic_coverage_metrics"] = analyze_topic_coverage(
                engine=engine,
                encoder=encoder,
                question_limit=topics_question_limit,
                n_topics=topics_count,
                top_k=topics_top_k,
            )
        if analyze_domain:
            results["domain_analysis_metrics"] = analyze_real_users_domain(
                engine=engine,
                limit=domain_limit,
            )
        return results

    benchmark = RAGBenchmark(engine, encoder, judge)
    dataset = dataset or []

    def attach_additional_analytics(results: dict[str, Any]) -> dict[str, Any]:
        if analyze_utilization:
            results["utilization_metrics"] = analyze_chunk_utilization(
                engine=engine,
                encoder=encoder,
                questions_source=utilization_questions_source,
                question_limit=utilization_question_limit,
                top_k=utilization_top_k,
            )
        if analyze_topics:
            results["topic_coverage_metrics"] = analyze_topic_coverage(
                engine=engine,
                encoder=encoder,
                question_limit=topics_question_limit,
                n_topics=topics_count,
                top_k=topics_top_k,
            )
        if analyze_domain:
            results["domain_analysis_metrics"] = analyze_real_users_domain(
                engine=engine,
                limit=domain_limit,
            )
        return results

    if tier == "all":
        results = benchmark.run_all_tiers(
            dataset,
            top_k=top_k,
            consistency_runs=consistency_runs,
        )
        results["tier_0"] = benchmark.run_tier_0(dataset)
        results["tier_judge"] = benchmark.run_tier_judge(dataset)
        judge_pipeline_dataset = load_judge_pipeline_dataset()
        if judge_pipeline_dataset:
            results["tier_judge_pipeline"] = benchmark.run_tier_judge_pipeline(
                judge_pipeline_dataset
            )
        results["tier_ux"] = benchmark.run_tier_ux(
            [{"questions": [item["question"] for item in dataset]}] if dataset else []
        )
        return attach_additional_analytics(results)
    elif tier == "1":
        return attach_additional_analytics(
            {
                "tier_1": benchmark.run_tier_1(
                    dataset,
                    top_k=top_k,
                    consistency_runs=consistency_runs,
                )
            }
        )
    elif tier == "2":
        return attach_additional_analytics(
            {
                "tier_2": benchmark.run_tier_2(
                    dataset,
                    consistency_runs=consistency_runs,
                )
            }
        )
    elif tier == "3":
        return attach_additional_analytics(
            {
                "tier_3": benchmark.run_tier_3(
                    dataset,
                    consistency_runs=consistency_runs,
                )
            }
        )
    elif tier == "0":
        return attach_additional_analytics({"tier_0": benchmark.run_tier_0(dataset)})
    elif tier == "judge":
        return attach_additional_analytics(
            {"tier_judge": benchmark.run_tier_judge(dataset)}
        )
    elif tier == "judge_pipeline":
        judge_pipeline_dataset = load_judge_pipeline_dataset(
            judge_pipeline_dataset_path
        )
        return attach_additional_analytics(
            {
                "tier_judge_pipeline": benchmark.run_tier_judge_pipeline(
                    judge_pipeline_dataset
                )
            }
        )
    elif tier == "ux":
        return attach_additional_analytics(
            {
                "tier_ux": benchmark.run_tier_ux(
                    [{"questions": [item["question"] for item in dataset]}]
                    if dataset
                    else []
                )
            }
        )
    else:
        raise ValueError(f"Неизвестный уровень бенчмарка: {tier}")


def save_results(
    results: dict,
    output_dir: str,
    dataset_name: str,
    engine,
    mode: str,
    judge_eval_mode: str = "direct",
    consistency_runs: int = 1,
):
    """Сохранить результаты бенчмарка.

    Args:
        results: Результаты бенчмарка
        output_dir: Директория для сохранения
        dataset_name: Имя датасета
        engine: Движок базы данных
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def to_builtin(value: Any) -> Any:
        """Преобразовать numpy-типы в стандартные типы Python."""
        if isinstance(value, dict):
            return {k: to_builtin(v) for k, v in value.items()}
        if isinstance(value, list):
            return [to_builtin(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    normalized_results: dict = to_builtin(results)

    report_generator = ReportGenerator()
    run_metadata = report_generator.get_run_metadata()
    overall_status = report_generator.evaluate_overall_status(normalized_results)

    artifact_payload = dict(normalized_results)
    artifact_payload["run_metadata"] = run_metadata
    artifact_payload["overall_status"] = overall_status
    artifact_payload["executive_summary"] = report_generator.generate_executive_summary(
        normalized_results
    )
    artifact_payload["dataset_file"] = os.path.basename(dataset_name)
    artifact_payload["dataset_type"] = mode

    json_path = os.path.join(output_dir, f"rag_benchmark_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact_payload, f, ensure_ascii=False, indent=2)

    markdown_path = os.path.join(output_dir, f"rag_benchmark_{timestamp}.md")
    markdown_report = report_generator.generate_benchmark_report(
        normalized_results,
        dataset_name=dataset_name,
        metadata=run_metadata,
    )

    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(markdown_report)

    benchmark_runs_path = os.path.join(output_dir, "benchmark_runs.json")
    run_entry = {
        "id": timestamp,
        "timestamp": datetime.now().isoformat(),
        "git_branch": run_metadata["git_branch"],
        "git_commit_hash": run_metadata["git_commit_hash"],
        "run_author": run_metadata["run_author"],
        "schema_version": "2.0",
        "dataset_file": os.path.basename(dataset_name),
        "dataset_type": mode,
        "judge_model": os.getenv("BENCHMARKS_JUDGE_MODEL")
        or os.getenv("JUDGE_MODEL")
        or Config.JUDGE_MODEL,
        "generation_model": os.getenv("GENERATION_MODEL") or Config.MISTRAL_MODEL,
        "embedding_model": Config.EMBEDDING_MODEL_PATH,
        "judge_eval_mode": judge_eval_mode,
        "consistency_runs": consistency_runs,
        "tier_0_metrics": normalized_results.get("tier_0"),
        "tier_1_metrics": normalized_results.get("tier_1"),
        "tier_2_metrics": normalized_results.get("tier_2"),
        "tier_3_metrics": normalized_results.get("tier_3"),
        "tier_judge_metrics": normalized_results.get("tier_judge"),
        "tier_judge_pipeline_metrics": normalized_results.get("tier_judge_pipeline"),
        "tier_ux_metrics": normalized_results.get("tier_ux"),
        "real_user_metrics": normalized_results.get("tier_real_users"),
        "utilization_metrics": normalized_results.get("utilization_metrics"),
        "topic_coverage_metrics": normalized_results.get("topic_coverage_metrics"),
        "domain_analysis_metrics": normalized_results.get("domain_analysis_metrics"),
        "model_runs": normalized_results.get("model_runs"),
        "executive_summary": artifact_payload.get("executive_summary", ""),
        "overall_status": overall_status,
    }

    if os.path.exists(benchmark_runs_path):
        with open(benchmark_runs_path, "r", encoding="utf-8") as f:
            runs = json.load(f)
    else:
        runs = []

    runs.append(run_entry)

    with open(benchmark_runs_path, "w", encoding="utf-8") as f:
        json.dump(runs, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Результаты сохранены:")
    logger.info(f"   JSON: {json_path}")
    logger.info(f"   Markdown: {markdown_path}")
    logger.info(f"   Runs index: {benchmark_runs_path}")


def print_results(results: dict):
    """Вывести результаты бенчмарка в консоль.

    Args:
        results: Результаты бенчмарка
    """
    print("\n" + "=" * 60)
    print("RAG BENCHMARK RESULTS")
    print("=" * 60 + "\n")

    for tier_name, tier_results in results.items():
        print(f"📊 {tier_name.upper()}")
        print("-" * 60)

        for key, value in tier_results.items():
            if key == "tier":
                continue
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print()

    print("=" * 60 + "\n")


def main():
    """Главная функция CLI скрипта."""
    parser = argparse.ArgumentParser(
        description="Запуск комплексных бенчмарков RAG-системы"
    )

    parser.add_argument(
        "--tier",
        type=str,
        choices=["0", "1", "2", "3", "judge", "judge_pipeline", "ux", "all"],
        default="all",
        help="Уровень бенчмарка (0, 1, 2, 3, judge, judge_pipeline, ux или all)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["synthetic", "manual", "real-users"],
        default="synthetic",
        help="Режим бенчмарка: synthetic, manual или real-users",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmarks/data/golden_dataset_synthetic.json",
        help=(
            "Путь к файлу датасета. Если используется путь по умолчанию и файл "
            "не найден, будет выбран последний benchmarks/data/dataset_*.json"
        ),
    )

    parser.add_argument(
        "--manual-dataset",
        type=str,
        default=None,
        help="Путь к manual dataset (используется при --mode manual)",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничение количества записей из датасета",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Количество результатов для поиска (Tier 1, default: 10)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/reports",
        help="Директория для сохранения результатов",
    )

    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Пропустить проверку предварительных условий",
    )

    parser.add_argument(
        "--real-score",
        type=int,
        default=5,
        help="Фильтр score для real-users режима (default: 5)",
    )

    parser.add_argument(
        "--real-limit",
        type=int,
        default=500,
        help="Лимит вопросов для real-users режима",
    )

    parser.add_argument(
        "--real-latest",
        action="store_true",
        help="Брать последние real-user вопросы по created_at",
    )

    parser.add_argument(
        "--analyze-utilization",
        action="store_true",
        help="Запустить дополнительный анализ chunk utilization",
    )

    parser.add_argument(
        "--utilization-questions-source",
        type=str,
        choices=["synthetic", "real"],
        default="synthetic",
        help="Источник вопросов для utilization анализа",
    )

    parser.add_argument(
        "--utilization-question-limit",
        type=int,
        default=500,
        help="Лимит вопросов для utilization анализа",
    )

    parser.add_argument(
        "--utilization-top-k",
        type=int,
        default=10,
        help="Top-k retrieval для utilization анализа",
    )

    parser.add_argument(
        "--analyze-topics",
        action="store_true",
        help="Запустить дополнительный анализ topic coverage",
    )

    parser.add_argument(
        "--topics-question-limit",
        type=int,
        default=2000,
        help="Лимит вопросов для topic coverage анализа",
    )

    parser.add_argument(
        "--topics-count",
        type=int,
        default=20,
        help="Количество тематических кластеров",
    )

    parser.add_argument(
        "--topics-top-k",
        type=int,
        default=5,
        help="Top-k retrieval для topic coverage анализа",
    )

    parser.add_argument(
        "--consistency-runs",
        type=int,
        default=1,
        help="Количество повторов запроса для consistency-метрик",
    )

    parser.add_argument(
        "--judge-eval-mode",
        choices=["direct", "reasoned"],
        default="direct",
        help="Режим LLM-судьи: direct или reasoned (CoT-style)",
    )

    parser.add_argument(
        "--judge-models",
        type=str,
        default="",
        help="CSV список judge моделей для multi-model сравнения",
    )

    parser.add_argument(
        "--generation-models",
        type=str,
        default="",
        help="CSV список generation моделей для multi-model сравнения",
    )

    parser.add_argument(
        "--analyze-domain",
        action="store_true",
        help="Запустить анализ предметной области по real-user вопросам",
    )

    parser.add_argument(
        "--domain-limit",
        type=int,
        default=5000,
        help="Лимит вопросов для domain analysis",
    )

    args = parser.parse_args()

    def parse_models(raw: str, fallback: str) -> list[str]:
        models = [item.strip() for item in raw.split(",") if item.strip()]
        return models or [fallback]

    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)

    model_path = Config.EMBEDDING_MODEL_PATH
    logger.info(f"Загрузка модели эмбеддингов: {model_path}")
    logger.info("Это может занять 1-2 минуты при первом запуске...")
    encoder = SentenceTransformer(model_path, device="cpu")
    logger.info("Модель загружена успешно")

    resolved_dataset = resolve_dataset_path(args.dataset)
    if args.mode == "manual":
        resolved_dataset = resolve_manual_dataset_path(
            args.manual_dataset, args.dataset
        )

    if not args.skip_checks:
        if not check_prerequisites(
            engine,
            mode=args.mode,
            dataset_path=resolved_dataset,
        ):
            sys.exit(1)

    dataset: Optional[list] = None
    if args.mode in {"synthetic", "manual"}:
        dataset = load_dataset(resolved_dataset, args.limit)

    needs_judge = args.mode in {"synthetic", "manual"} and args.tier in {
        "2",
        "3",
        "all",
    }

    default_judge_model = (
        os.getenv("BENCHMARKS_JUDGE_MODEL")
        or os.getenv("JUDGE_MODEL")
        or Config.JUDGE_MODEL
        or ""
    )
    default_generation_model = (
        os.getenv("GENERATION_MODEL") or Config.MISTRAL_MODEL or ""
    )
    judge_models = parse_models(args.judge_models, default_judge_model)
    generation_models = parse_models(args.generation_models, default_generation_model)

    if dataset is not None:
        logger.info(
            "Запуск бенчмарка mode=%s, tier=%s, dataset_rows=%s",
            args.mode,
            args.tier,
            len(dataset),
        )
    else:
        logger.info("Запуск бенчмарка mode=%s, tier=%s", args.mode, args.tier)

    model_runs: list[dict[str, Any]] = []
    base_results: Optional[dict] = None
    base_generation_model = Config.MISTRAL_MODEL

    for generation_model in generation_models:
        Config.MISTRAL_MODEL = generation_model
        os.environ["GENERATION_MODEL"] = generation_model

        for judge_model in judge_models:
            judge = None
            if needs_judge:
                judge = LLMJudge(
                    model=judge_model,
                    evaluation_mode=args.judge_eval_mode,
                )

            run_result = run_benchmark(
                engine,
                encoder,
                judge,
                dataset,
                args.tier,
                args.mode,
                args.top_k,
                real_score_filter=args.real_score,
                real_limit=args.real_limit,
                real_latest=args.real_latest,
                judge_pipeline_dataset_path=None
                if args.tier == "judge_pipeline"
                else args.dataset,
                analyze_utilization=args.analyze_utilization,
                analyze_topics=args.analyze_topics,
                utilization_questions_source=args.utilization_questions_source,
                utilization_question_limit=args.utilization_question_limit,
                utilization_top_k=args.utilization_top_k,
                topics_question_limit=args.topics_question_limit,
                topics_count=args.topics_count,
                topics_top_k=args.topics_top_k,
                consistency_runs=max(1, args.consistency_runs),
                analyze_domain=args.analyze_domain,
                domain_limit=max(1, args.domain_limit),
            )

            model_runs.append(
                {
                    "judge_model": judge_model,
                    "generation_model": generation_model,
                    "judge_eval_mode": args.judge_eval_mode,
                    "metrics": run_result,
                }
            )
            if base_results is None:
                base_results = run_result

    Config.MISTRAL_MODEL = base_generation_model

    if base_results is None:
        raise RuntimeError("Не удалось получить результаты бенчмарка")

    results = base_results
    if len(model_runs) > 1:
        results["model_runs"] = model_runs

    dataset_name = resolved_dataset
    if args.mode == "real-users":
        dataset_name = f"real_users_score_{args.real_score}"

    save_results(
        results,
        args.output_dir,
        dataset_name=dataset_name,
        engine=engine,
        mode=args.mode,
        judge_eval_mode=args.judge_eval_mode,
        consistency_runs=max(1, args.consistency_runs),
    )
    print_results(results)

    logger.info("✅ Бенчмарк успешно завершён")


if __name__ == "__main__":
    main()
