from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from qa.database import QuestionAnswer, Session
from qa.main import encoder_model, engine


# revision identifiers, used by Alembic.
revision: str = "766d54169326"
down_revision: Union[str, None] = "051032488686"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("question_answer") as batch_op:
        batch_op.add_column(sa.Column("embedding", sa.LargeBinary, nullable=True))

    with Session(engine) as session:
        for row in session.query(QuestionAnswer).all():
            row.embedding = encoder_model.encode(row.question)
        session.commit()


def downgrade() -> None:
    with op.batch_alter_table("question_answer") as batch_op:
        batch_op.drop_column("embedding")
