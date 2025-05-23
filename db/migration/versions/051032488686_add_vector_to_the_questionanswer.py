"""add vector to the QuestionAnswer

Revision ID: 051032488686
Revises: 3c9c99360a80
Create Date: 2025-03-20 11:49:52.429862

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from pgvector.sqlalchemy import Vector

revision: str = "051032488686"
down_revision: Union[str, None] = "3c9c99360a80"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "question_answer_temp",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1024)),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("confluence_url", sa.Text(), nullable=True),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.execute(
        """
        INSERT INTO question_answer_temp (question, embedding, answer, confluence_url, score, user_id, created_at, updated_at)
        SELECT question, NULL, answer, confluence_url, score, user_id, created_at, updated_at FROM question_answer;
        """
    )
    op.drop_table("question_answer")
    op.rename_table("question_answer_temp", "question_answer")
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_class WHERE relname = 'question_answer_id_seq') THEN
                CREATE SEQUENCE question_answer_id_seq;
            END IF;
        END $$;
        """
    )

    op.execute("ALTER SEQUENCE question_answer_id_seq OWNED BY question_answer.id;")
    op.execute(
        """
        SELECT setval('question_answer_id_seq', COALESCE((SELECT max(id) FROM question_answer), 1), false);
        """
    )


def downgrade() -> None:
    op.create_table(
        "question_answer_old",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("confluence_url", sa.Text(), nullable=True),
        sa.Column("score", sa.Integer(), nullable=True),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"]),
        sa.PrimaryKeyConstraint("id"),
    )

    op.execute(
        """
        INSERT INTO question_answer_old (id, question, answer, confluence_url, score, user_id, created_at, updated_at)
        SELECT id, question, answer, confluence_url, score, user_id, created_at, updated_at FROM question_answer
    """
    )

    op.drop_table("question_answer")
    op.rename_table("question_answer_old", "question_answer")
