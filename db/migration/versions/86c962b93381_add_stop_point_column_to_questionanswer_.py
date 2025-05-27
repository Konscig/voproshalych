"""Add stop_point column to questionanswer table

Revision ID: 86c962b93381
Revises: 051032488686
Create Date: 2025-04-03 17:54:15.610157

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "86c962b93381"
down_revision: Union[str, None] = "051032488686"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "question_answer",
        sa.Column(
            "stop_point", sa.Boolean(), nullable=False, server_default=sa.text("FALSE")
        ),
    )


def downgrade() -> None:
    op.drop_column("question_answer", "stop_point")
