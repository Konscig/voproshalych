import numpy as np
from pgvector.sqlalchemy import Vector
from typing import Optional, List
from sentence_transformers import SentenceTransformer
from sqlalchemy import (
    Text,
    Column,
    DateTime,
    func,
    ForeignKey,
    BigInteger,
    Engine,
    select,
)
from sqlalchemy.orm import (
    Mapped,
    Session,
    declarative_base,
    mapped_column,
    relationship,
)
from sqlalchemy.exc import SQLAlchemyError

Base = declarative_base()


class User(Base):
    """Пользователь чат-бота

    Args:
        id (int): id пользователя
        vk_id (int | None): id пользователя ВКонтакте
        telegram_id (int | None): id пользователя Telegram
        is_subscribed (bool): состояние подписки пользователя
        question_answers (List[QuestionAnswer]): вопросы пользователя
        created_at (datetime): время создания модели
        updated_at (datetime): время обновления модели
    """

    __tablename__ = "user"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    vk_id: Mapped[Optional[int]] = mapped_column(BigInteger, unique=True)
    telegram_id: Mapped[Optional[int]] = mapped_column(BigInteger, unique=True)
    is_subscribed: Mapped[bool] = mapped_column()

    question_answers: Mapped[List["QuestionAnswer"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="desc(QuestionAnswer.created_at)",
    )

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Chunk(Base):
    """Фрагмент документа из вики-системы

    Args:
        confluence_url (str): ссылка на источник
        text (str): текст фрагмента
        embedding (Vector): векторное представление текста фрагмента размерностью 1024
        created_at (datetime): время создания модели
        updated_at (datetime): время обновления модели
    """

    __tablename__ = "chunk"

    id: Mapped[int] = mapped_column(primary_key=True)
    confluence_url: Mapped[str] = mapped_column(Text(), index=True)
    text: Mapped[str] = mapped_column(Text())
    embedding: Mapped[Vector] = mapped_column(Vector(1024))

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class QuestionAnswer(Base):
    """Вопрос пользователя с ответом на него

    Args:
        id (int): id ответа
        question (str): вопрос пользователя
        embedding (Vector): векторное представление текста вопроса размерностью 1024
        answer (str | None): ответ на вопрос пользователя
        confluence_url (str | None): ссылка на страницу в вики-системе, содержащую ответ
        score (int | None): оценка пользователем ответа
        user_id (int): id пользователя, задавшего вопрос
        user (User): пользователь, задавший вопрос
        created_at (datetime): время создания модели
        updated_at (datetime): время обновления модели
    """

    __tablename__ = "question_answer"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text())
    embedding: Mapped[Vector] = mapped_column(Vector(1024))
    answer: Mapped[Optional[str]] = mapped_column(Text())
    confluence_url: Mapped[Optional[str]] = mapped_column(Text(), index=True)
    score: Mapped[Optional[int]] = mapped_column()
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"))

    user: Mapped["User"] = relationship(back_populates="question_answers")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


def get_answer_by_id(engine: Engine, question_answer_id: int):
    with Session(engine) as session:
        answer = (
            session.execute(
                select(QuestionAnswer.answer).where(
                    QuestionAnswer.id == question_answer_id
                )
            )
            .scalars()
            .first()
        )
    return answer


def set_embedding(engine: Engine, question_answer_id: int, embed):
    with Session(engine) as session:
        try:
            with session.begin():
                question_answer = (
                    session.query(QuestionAnswer)
                    .filter_by(id=question_answer_id)
                    .first()
                )
                if question_answer:
                    question_answer.embedding = embed
                else:
                    raise ValueError("QuestionAnswer not found")
        except SQLAlchemyError as e:
            session.rollback()
            raise e


def get_the_highest_question(
    engine: Engine, encoder_model: SentenceTransformer, question: str
) -> str:
    question_embedding = encoder_model.encode(question)

    with Session(engine) as session:
        result = session.execute(
            select(QuestionAnswer.answer)
            .order_by(
                QuestionAnswer.embedding.cosine_distance(Vector(question_embedding))
            )
            .where(QuestionAnswer.score == 5)
            .limit(1)
        ).first()
    if result:
        answer = str(result[0])

    return answer
