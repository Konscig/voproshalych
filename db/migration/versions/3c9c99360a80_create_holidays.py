"""create holidays

Revision ID: 3c9c99360a80
Revises: 22dcc1a837cc
Create Date: 2025-02-25 03:15:44.907264

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3c9c99360a80"
down_revision: Union[str, None] = "22dcc1a837cc"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "holiday_templates",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("type", sa.Enum("fixed", "floating", name="holiday_types")),
        sa.Column("template_llm", sa.Text),
        sa.Column("vk", sa.Boolean, default=False),
        sa.Column("tg", sa.Boolean, default=False),
        sa.Column("month", sa.Integer),
        sa.Column("day", sa.Integer),
        sa.Column("week_number", sa.Integer),
        sa.Column("day_of_week", sa.Integer),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now()
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), onupdate=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("holiday_templates")
    op.execute("DROP TYPE IF EXISTS holiday_types")
