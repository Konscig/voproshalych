"""Скрипт на один раз, который задаст вектора вопросам"""

from database import (
    set_embedding,
    get_all_questions_with_score,
)
from main import engine, encoder_model

all_questions = get_all_questions_with_score(engine=engine)

for question in all_questions:
    set_embedding(
        engine=engine,
        question_answer_id=question["id"],
        embed=encoder_model.encode(question["answer"]),
    )
