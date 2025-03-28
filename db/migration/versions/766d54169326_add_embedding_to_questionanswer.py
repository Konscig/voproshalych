from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
import requests
from os import environ
from dotenv import load_dotenv


# revision identifiers, used by Alembic.
revision: str = "766d54169326"
down_revision: Union[str, None] = "051032488686"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

load_dotenv(dotenv_path="../../../.env")


def upgrade() -> None:
    conn = op.get_bind()
    result = conn.execute(
        sa.text(
            """
        SELECT id FROM question_answer WHERE score = 5
    """
        )
    )

    for row in result.fetchall():
        qa_id = row.id
        try:
            response = requests.post(
                f"http://{environ.get('QA_HOST')}/answer_embed/",
                json={"answer_id": qa_id},
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to set embedding for answer {qa_id}: {e}")


def downgrade() -> None:
    op.execute(sa.text("UPDATE question_answer SET embedding = NULL WHERE score = 5"))
