"""Rename confluence_url to url and add source_type column

Revision ID: a1b2c3d4e5f6
Revises: 86c962b93381
Create Date: 2026-01-20 08:30:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "86c962b93381"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Step 1: Add new column 'url' (copy of confluence_url)
    op.add_column(
        "chunk",
        sa.Column("url", sa.Text(), nullable=True),
    )

    # Step 2: Copy data from confluence_url to url
    op.execute("""
        UPDATE chunk
        SET url = confluence_url
    """)

    # Step 3: Make url NOT NULL
    op.alter_column(
        "chunk",
        "url",
        nullable=False,
    )

    # Step 4: Add source_type column
    op.add_column(
        "chunk",
        sa.Column("source_type", sa.String(50), nullable=True),
    )

    # Step 5: Populate source_type based on URL
    op.execute("""
        UPDATE chunk
        SET source_type = CASE
            WHEN url LIKE 'https://www.utmn.ru/news/stories/%' THEN 'news'
            WHEN url LIKE 'https://www.utmn.ru/news/events/%' THEN 'events'
            WHEN url LIKE 'https://www.utmn.ru/o-tyumgu/sotrudniki/%' THEN 'employees'
            WHEN url LIKE 'https://www.utmn.ru/o-tyumgu/struktura-universiteta/%' THEN 'structure'
            WHEN url LIKE 'https://www.utmn.ru/kontakty%' THEN 'contacts'
            WHEN url LIKE 'https://www.utmn.ru/rss/%' THEN 'rss'
            WHEN url LIKE '%confluence.utmn.ru%' THEN 'confluence'
            ELSE 'other'
        END
    """)

    # Step 6: Make source_type NOT NULL
    op.alter_column(
        "chunk",
        "source_type",
        nullable=False,
    )

    # Step 7: Create index on source_type
    op.create_index(
        op.f("ix_chunk_source_type"),
        "chunk",
        ["source_type"],
    )

    # Step 8: Create index on url
    op.create_index(
        op.f("ix_chunk_url"),
        "chunk",
        ["url"],
    )

    # Step 9: Drop old confluence_url column
    op.drop_column("chunk", "confluence_url")


def downgrade() -> None:
    # Reverse the changes

    # Step 1: Add back confluence_url column
    op.add_column(
        "chunk",
        sa.Column("confluence_url", sa.Text(), nullable=True),
    )

    # Step 2: Copy data from url to confluence_url
    op.execute("""
        UPDATE chunk
        SET confluence_url = url
    """)

    # Step 3: Make confluence_url NOT NULL
    op.alter_column(
        "chunk",
        "confluence_url",
        nullable=False,
    )

    # Step 4: Drop source_type column
    op.drop_index(op.f("ix_chunk_source_type"), table_name="chunk")
    op.drop_column("chunk", "source_type")

    # Step 5: Drop url column
    op.drop_index(op.f("ix_chunk_url"), table_name="chunk")
    op.drop_column("chunk", "url")
