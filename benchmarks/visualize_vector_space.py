"""CLI для UMAP-визуализации векторного пространства чанков."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sqlalchemy import select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).parent.parent))

from qa.config import Config
from qa.database import Chunk, create_engine

logger = logging.getLogger(__name__)


def _extract_section(url: str | None) -> str:
    """Извлечь секцию из URL для цветовой группировки."""
    if not url:
        return "unknown"
    parts = [part for part in url.split("/") if part]
    if not parts:
        return "unknown"
    return parts[-1][:48]


def _load_chunk_embeddings(limit: int) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Загрузить эмбеддинги чанков и метаданные из БД."""
    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
    with Session(engine) as session:
        rows = session.scalars(
            select(Chunk)
            .where(Chunk.embedding.isnot(None))
            .order_by(Chunk.id.asc())
            .limit(limit)
        ).all()

    embeddings = np.array([np.array(row.embedding, dtype=float) for row in rows])
    metadata: list[dict[str, object]] = []
    for row in rows:
        metadata.append(
            {
                "chunk_id": row.id,
                "confluence_url": row.confluence_url,
                "section": _extract_section(row.confluence_url),
            }
        )
    return embeddings, metadata


def visualize_embeddings(
    embeddings: np.ndarray,
    metadata: list[dict[str, object]],
    output_path: str,
    dim: int = 2,
    color_by: str = "section",
) -> str:
    """Визуализировать эмбеддинги через UMAP и сохранить HTML."""
    import umap

    reducer = umap.UMAP(
        n_components=dim,
        random_state=42,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
    )
    projected = reducer.fit_transform(embeddings)

    df = pd.DataFrame(
        {
            "x": projected[:, 0],
            "y": projected[:, 1],
            "chunk_id": [item["chunk_id"] for item in metadata],
            "section": [item["section"] for item in metadata],
            "confluence_url": [item["confluence_url"] for item in metadata],
        }
    )

    if color_by == "cluster":
        n_clusters = max(2, min(20, len(df)))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        df["cluster"] = kmeans.fit_predict(embeddings).astype(str)
    else:
        df["cluster"] = "0"

    if dim == 3:
        df["z"] = projected[:, 2]
        fig = px.scatter_3d(
            df,
            x="x",
            y="y",
            z="z",
            color=color_by if color_by in df.columns else None,
            hover_data=["chunk_id", "section", "confluence_url"],
            title="UMAP 3D: Vector Space",
        )
    else:
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color=color_by if color_by in df.columns else None,
            hover_data=["chunk_id", "section", "confluence_url"],
            title="UMAP 2D: Vector Space",
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs="cdn")
    return str(output)


def main() -> None:
    """CLI-вход для визуализации векторного пространства."""
    parser = argparse.ArgumentParser(description="UMAP-визуализация эмбеддингов")
    parser.add_argument(
        "--limit",
        type=int,
        default=5000,
        help="Количество чанков для визуализации",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmarks/reports/vector_space.html",
        help="Путь к итоговому HTML-файлу",
    )
    parser.add_argument(
        "--3d",
        action="store_true",
        dest="use_3d",
        help="Использовать 3D-проекцию вместо 2D",
    )
    parser.add_argument(
        "--color-by",
        type=str,
        choices=["cluster", "chunk_id", "section"],
        default="section",
        help="Поле для раскраски точек",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    embeddings, metadata = _load_chunk_embeddings(args.limit)
    if len(embeddings) < 3:
        raise ValueError("Недостаточно эмбеддингов для UMAP-визуализации")

    dim = 3 if args.use_3d else 2
    output_path = visualize_embeddings(
        embeddings=embeddings,
        metadata=metadata,
        output_path=args.output,
        dim=dim,
        color_by=args.color_by,
    )
    logger.info("Визуализация сохранена: %s", output_path)
    print(output_path)


if __name__ == "__main__":
    main()
