import base64
from typing import Optional, List
from bcrypt import hashpw, gensalt, checkpw
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from sqlalchemy import (
    BigInteger,
    Column,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    func,
    and_,
)
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from cluster_analysis import mark_of_question
from config import app
from pandas import date_range
from urllib.parse import unquote, parse_qs, urlparse
import re

db = SQLAlchemy(app)


class Chunk(db.Model):
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


class User(db.Model):
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


class QuestionAnswer(db.Model):
    """Вопрос пользователя с ответом на него

    Args:
        id (int): id ответа
        question (str): вопрос пользователя
        answer (str | None): ответ на вопрос пользователя
        confluence_url (str | None): ссылка на страницу в вики-системе, содержащую ответ
        score (int | None): оценка пользователем ответа
        user_id (int): id пользователя, задавшего вопрос
        user (User): пользователь, задавший вопрос
        created_at (datetime): время создания модели
        updated_at (datetime): время обновления модели
        stop_point (bool): флаг, указывающий что это конечная точка диалога (True - диалог завершен, False - продолжается, по умолчанию False)
    """

    __tablename__ = "question_answer"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text())
    answer: Mapped[Optional[str]] = mapped_column(Text())
    confluence_url: Mapped[Optional[str]] = mapped_column(Text(), index=True)
    score: Mapped[Optional[int]] = mapped_column()
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"))

    user: Mapped["User"] = relationship(back_populates="question_answers")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    stop_point: Mapped[bool] = mapped_column(Boolean(), default=False)


class Admin(db.Model, UserMixin):
    """Администратор панели

    Args:
        id (int): id администратора
        name (str): имя
        surname (str): фамилия
        last_name (str | None): отчество (опционально)
        email (str): корпоративная электронная почта
        department (str): подразделение
        created_at (datetime): время создания модели
        updated_at (datetime): время обновления модели
    """

    __tablename__ = "admin"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text())
    surname: Mapped[str] = mapped_column(Text())
    last_name: Mapped[Optional[str]] = mapped_column(Text())
    email: Mapped[str] = mapped_column(Text(), unique=True)
    department: Mapped[str] = mapped_column(Text())
    password_hash: Mapped[str] = mapped_column(Text())

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def set_password(self, password: str) -> None:
        """Метод хеширования пароля администратора.

        Args:
            password (str): пароль администратора
        """
        hashed = hashpw(password.encode("utf-8"), gensalt())
        self.password_hash = base64.b64encode(hashed).decode("utf-8")

    def check_password(self, password: str) -> bool:
        """Метод проверки введенного администратором пароля

        Args:
            password (str): введенный администратором пароль

        Returns:
            bool: проверка, совпадает ли введенный пароль с хешированным паролем
        """
        hashed = base64.b64decode(self.password_hash.encode("utf-8"))
        return checkpw(password.encode("utf-8"), hashed)


def get_questions_for_clusters(
    time_start: str,
    time_end: str,
    have_not_answer: bool,
    have_low_score: bool,
    have_high_score: bool,
    have_not_score: bool,
) -> list[dict[str, str | mark_of_question]]:
    """Функция для выгрузки вопросов для обработки в классе `ClusterAnalysis`

    Args:
        time_start (str): дата, от которой нужно сортировать вопросы
        time_end (str): дата, до которой нужно сортировать вопросы
        have_not_answer (bool): вопросы без ответа
        have_low_score (bool): вопросы с низкой оценкой
        have_high_score (bool): вопросы с высокой оценкой
        have_not_score (bool): вопросы без оценки

    Returns:
        list[dict[str, str | mark_of_question]]: список вопросов - словарей с ключами `text, `date` и `type`
    """

    with Session(db.engine) as session:
        query = session.query(QuestionAnswer).filter(
            QuestionAnswer.created_at.between(time_start, time_end)
        )
        questions = []
        if have_not_answer:
            for qa in (
                query.filter(QuestionAnswer.answer == "")
                .order_by(QuestionAnswer.id)
                .all()
            ):
                questions.append(
                    {
                        "text": qa.question,
                        "date": qa.created_at.strftime("%Y-%m-%d"),
                        "type": mark_of_question.have_not_answer,
                    }
                )
        if have_low_score:
            for qa in (
                query.filter(QuestionAnswer.score == 1)
                .order_by(QuestionAnswer.id)
                .all()
            ):
                questions.append(
                    {
                        "text": qa.question,
                        "date": qa.created_at.strftime("%Y-%m-%d"),
                        "type": mark_of_question.have_low_score,
                    }
                )
        if have_high_score:
            for qa in (
                query.filter(QuestionAnswer.score == 5)
                .order_by(QuestionAnswer.id)
                .all()
            ):
                questions.append(
                    {
                        "text": qa.question,
                        "date": qa.created_at.strftime("%Y-%m-%d"),
                        "type": mark_of_question.have_high_score,
                    }
                )
        if have_not_score:
            for qa in (
                query.filter(
                    and_(QuestionAnswer.score == None, QuestionAnswer.answer != "")
                )
                .order_by(QuestionAnswer.id)
                .all()
            ):
                questions.append(
                    {
                        "text": qa.question,
                        "date": qa.created_at.strftime("%Y-%m-%d"),
                        "type": mark_of_question.have_not_score,
                    }
                )
        return questions


def get_questions_count(
    time_start: str,
    time_end: str,
) -> dict[str, list[int]]:
    """Функция подсчёта вопросов, заданных в вк и телеграм, по дням для графиков на `main-page.html`

    Args:
        time_start (str): дата начала периода
        time_end (str): дата конца периода

    Returns:
        dict[str, list[int]]: словарь из дат с количеством вопросов по дням в vk и telegram
    """

    with Session(db.engine) as session:
        vk_questions_count = (
            session.query(
                func.date_trunc("day", QuestionAnswer.created_at), func.count()
            )
            .join(User)
            .filter(
                User.vk_id != None,
                QuestionAnswer.created_at.between(time_start, time_end),
            )
            .group_by(func.date_trunc("day", QuestionAnswer.created_at))
            .all()
        )
        telegram_questions_count = (
            session.query(
                func.date_trunc("day", QuestionAnswer.created_at), func.count()
            )
            .join(User)
            .filter(
                User.telegram_id != None,
                QuestionAnswer.created_at.between(time_start, time_end),
            )
            .group_by(func.date_trunc("day", QuestionAnswer.created_at))
            .all()
        )

        dates = date_range(time_start, time_end).strftime("%Y-%m-%d").tolist()
        questions_count = {date: [0, 0] for date in dates}
        for date, count in vk_questions_count:
            questions_count[date.strftime("%Y-%m-%d")][0] = count
        for date, count in telegram_questions_count:
            questions_count[date.strftime("%Y-%m-%d")][1] = count
        return questions_count


def get_admins() -> list[Admin]:
    """Функция для выгрузки администраторов из БД

    Returns:
        list[Admin]: список администраторов
    """
    with Session(db.engine) as session:
        admins = [
            Admin(
                name=admin.name,
                surname=admin.surname,
                last_name=admin.last_name,
                email=admin.email,
                department=admin.department,
            )
            for admin in session.query(Admin).order_by(Admin.id).all()
        ]
    return admins


def get_documents_count() -> int:
    """Возвращает количество уникальных документов (confluence_url)"""
    with Session(db.engine) as session:
        return session.query(func.count(func.distinct(Chunk.confluence_url))).scalar()


def get_chunks_count() -> int:
    """Возвращает общее количество фрагментов"""
    with Session(db.engine) as session:
        return session.query(func.count(Chunk.id)).scalar()


def get_last_sync_date() -> str:
    """Возвращает дату последнего обновления или создания фрагмента"""
    with Session(db.engine) as session:
        last_updated = (
            session.query(func.max(Chunk.updated_at))
            .filter(Chunk.updated_at.is_not(None))
            .scalar()
        )
        if not last_updated:
            last_updated = session.query(func.max(Chunk.created_at)).scalar()
        return last_updated.strftime("%Y-%m-%d") if last_updated else "N/A"


def get_unique_docs_urls() -> list[dict]:
    """
    Функция возвращает уникальные документы Confluence в формате {title: название, url: ссылка}
    Если функция не находит название в url странице, то название берётся из первого чанка для этого документа
    """
    with Session(db.engine) as session:
        unique_urls = session.query(Chunk.confluence_url).distinct().all()

        documents = []
        for (raw_url,) in unique_urls:
            decoded_url = unquote(raw_url)
            parsed_url = urlparse(decoded_url)
            query_params = parse_qs(parsed_url.query)

            title = "Неизвестный документ"
            if "preview" in query_params:
                preview_path = unquote(query_params["preview"][0])
                filename = preview_path.split("/")[-1]
                clean_name = filename.rsplit(".", 1)[0]
                clean_name = clean_name.replace("+", " ")
                clean_name = unquote(clean_name)
                clean_name = re.sub(r"\s*\(\d+\)\s*", "", clean_name)
                title = " ".join(
                    word.capitalize() for word in clean_name.replace("-", " ").split()
                )

            if title == "Неизвестный документ":
                first_chunk = (
                    session.query(Chunk.text)
                    .filter(Chunk.confluence_url == raw_url)
                    .order_by(Chunk.created_at.asc())
                    .first()
                )

                if first_chunk and first_chunk.text:
                    words = first_chunk.text.split()[:7]
                    title = " ".join(words).capitalize()

            documents.append({"title": title, "url": raw_url})

    return documents
