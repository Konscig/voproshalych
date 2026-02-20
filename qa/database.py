import numpy as np
from pgvector.sqlalchemy import Vector
from typing import Optional, List, TypedDict
from sentence_transformers import SentenceTransformer
from sqlalchemy import (
    Text,
    JSON,
    Column,
    DateTime,
    create_engine,
    func,
    ForeignKey,
    BigInteger,
    Engine,
    select,
    update,
    text,
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


class BenchmarkRun(Base):
    """Результат запуска бенчмарка RAG-системы.

    Args:
        timestamp (datetime): время запуска бенчмарка
        git_branch (str | None): ветка git, на которой запущен бенчмарк
        git_commit_hash (str | None): short hash коммита
        run_author (str | None): автор запуска
        tier_1_metrics (dict | None): метрики уровня Retrieval
        tier_2_metrics (dict | None): метрики уровня Generation
        tier_3_metrics (dict | None): метрики end-to-end уровня
        overall_status (str): итоговый статус качества
        dataset_type (str | None): тип датасета (synthetic/manual/real_users)
        real_user_metrics (dict | None): метрики real-user retrieval
        created_at (datetime): время создания записи
        updated_at (datetime): время обновления записи
    """

    __tablename__ = "benchmark_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    git_branch: Mapped[Optional[str]] = mapped_column(Text(), nullable=True)
    git_commit_hash: Mapped[Optional[str]] = mapped_column(Text(), nullable=True)
    run_author: Mapped[Optional[str]] = mapped_column(Text(), nullable=True)
    dataset_file: Mapped[Optional[str]] = mapped_column(Text(), nullable=True)
    tier_0_metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    tier_1_metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    tier_2_metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    tier_3_metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    tier_judge_metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    tier_judge_pipeline_metrics: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )
    tier_ux_metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    real_user_metrics: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    dataset_type: Mapped[Optional[str]] = mapped_column(Text(), nullable=True)
    overall_status: Mapped[str] = mapped_column(Text(), nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


def ensure_benchmark_runs_schema(engine: Engine) -> None:
    """Создать таблицу benchmark_runs и недостающие колонки при необходимости."""
    Base.metadata.create_all(bind=engine, tables=[BenchmarkRun.__table__])
    with engine.begin() as connection:
        connection.execute(
            text(
                "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS dataset_file TEXT"
            )
        )
        connection.execute(
            text(
                "ALTER TABLE benchmark_runs ADD COLUMN IF NOT EXISTS dataset_type TEXT"
            )
        )
        connection.execute(
            text(
                "ALTER TABLE benchmark_runs "
                "ADD COLUMN IF NOT EXISTS real_user_metrics JSON"
            )
        )
        connection.execute(
            text(
                "ALTER TABLE benchmark_runs "
                "ADD COLUMN IF NOT EXISTS tier_0_metrics JSON"
            )
        )
        connection.execute(
            text(
                "ALTER TABLE benchmark_runs "
                "ADD COLUMN IF NOT EXISTS tier_judge_metrics JSON"
            )
        )
        connection.execute(
            text(
                "ALTER TABLE benchmark_runs "
                "ADD COLUMN IF NOT EXISTS tier_judge_pipeline_metrics JSON"
            )
        )
        connection.execute(
            text(
                "ALTER TABLE benchmark_runs "
                "ADD COLUMN IF NOT EXISTS tier_ux_metrics JSON"
            )
        )


def get_answer_by_id(engine: Engine, question_answer_id: int) -> str:
    """Получает ответ по id QuestionAnswer

    Args:
        engine (Engine): текущее подключение к БД
        question_answer_id (int): идентификатор QuestionAnswer

    Returns:
        str: текст ответа на вопрос
    """
    with Session(engine) as session:
        answer = str(
            session.execute(
                select(QuestionAnswer.answer, QuestionAnswer.question).where(
                    QuestionAnswer.id == question_answer_id
                )
            )
            .scalars()
            .first()
        )
    return answer


def set_embedding(engine: Engine, question_answer_id: int, embed):
    """Задает векторное представление вопросу в БД
    Args:
        engine (Engine): текущее подключение к БД
        question_answer_id (int): идентификатор QuestionAnswer
        embed (Embedding): векторное представление вопроса

    Raises:
        ValueError: в случае не нахождения QuestionAnswer
        e: SQL error
    """
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


class QuestionDict(TypedDict):
    """Класс вопроса с векторным представлением"""

    id: int
    question: str
    answer: str
    embedding: np.ndarray
    url: str


def get_all_questions_with_score(
    engine: Engine, highscore: bool = True
) -> List[QuestionDict]:
    """Получает все вопросы с оценкой "отлично", либо все вопросы с оценками, если `highscore` == `False`

    Args:
        engine (Engine): текущее подключение к БД

    Returns:
        List[QuestionDict]: Список QuestionDict с id вопроса, его ответом, векторным представлением и ссылкой на Confluence
    """

    if highscore:
        with Session(engine) as session:
            question_answers = session.execute(
                select(
                    QuestionAnswer.id,
                    QuestionAnswer.question,
                    QuestionAnswer.answer,
                    QuestionAnswer.embedding,
                    QuestionAnswer.confluence_url,
                ).where(QuestionAnswer.score == 5)
            ).all()

        result: List[QuestionDict] = [
            {
                "id": int(qa_id),
                "question": str(question_text),
                "answer": str(answer_text),
                "embedding": (
                    np.array(embedding) if embedding is not None else np.array([])
                ),
                "url": str(url),
            }
            for qa_id, question_text, answer_text, embedding, url in question_answers
        ]

        return result
    else:
        with Session(engine) as session:
            question_answers = session.execute(
                select(
                    QuestionAnswer.id,
                    QuestionAnswer.question,
                    QuestionAnswer.answer,
                    QuestionAnswer.embedding,
                    QuestionAnswer.confluence_url,
                ).where(QuestionAnswer.score != None)
            ).all()

        result: List[QuestionDict] = [
            {
                "id": int(qa_id),
                "question": str(question_text),
                "answer": str(answer_text),
                "embedding": (
                    np.array(embedding) if embedding is not None else np.array([])
                ),
                "url": str(url),
            }
            for qa_id, question_text, answer_text, embedding, url in question_answers
        ]

        return result


def get_document_by_url(engine: Engine, url: str) -> str | None:
    """Собирает документ из чанков по ссылке на Confluence

    Args:
        engine (Engine): подключение к БД
        url (str): ссылка на Confluence

    Returns:
        str | None: полный текст документа или None, если документ не найден
    """
    if not url:
        return None

    with Session(engine) as session:
        document = session.execute(
            select(func.string_agg(Chunk.text, "\n\n").label("full_text")).where(
                Chunk.confluence_url == url, Chunk.text != None
            )
        ).scalar_one_or_none()

    return document


def delete_score(engine: Engine, qa_id: int) -> None:
    """Удаляет оценку у вопроса и эмбеддинг, если оценка 5"""
    with Session(engine) as session:
        qa = session.query(QuestionAnswer).filter(QuestionAnswer.id == qa_id).first()

        if qa:
            if qa.score == 5:
                session.execute(
                    update(QuestionAnswer)
                    .where(QuestionAnswer.id == qa_id)
                    .values(embedding=None)
                )
            session.execute(
                update(QuestionAnswer)
                .where(QuestionAnswer.id == qa_id)
                .values(score=None)
            )
            session.commit()
