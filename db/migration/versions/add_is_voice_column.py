"""Add is_voice column to questionanswer table

Revision ID: add_is_voice_column
Revises: 86c962b93381
Create Date: 2024-05-23 17:54:15.610157

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "add_is_voice_column"
down_revision: Union[str, None] = "86c962b93381"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "question_answer",
        sa.Column(
            "is_voice", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")
        ),
    )


def downgrade() -> None:
    op.drop_column("question_answer", "is_voice")
