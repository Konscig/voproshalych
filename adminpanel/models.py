import base64
from typing import Optional, List
from bcrypt import hashpw, gensalt, checkpw
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from sqlalchemy import (
    BigInteger,
    Integer,
    Column,
    DateTime,
    ForeignKey,
    Text,
    Enum,
    event,
    func,
    and_,
)
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from cluster_analysis import mark_of_question
from config import app
from pandas import date_range
from datetime import date, timedelta

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


class HolidayTemplate(db.Model):
    __tablename__ = "holiday_template"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(Text(), nullable=False)
    type: Mapped[str] = mapped_column(Enum("fixed", "floating", name="holiday_types"))
    template_llm: Mapped[str] = mapped_column(Text())
    vk: Mapped[bool] = mapped_column()
    tg: Mapped[bool] = mapped_column()
    male_holiday: Mapped[bool] = mapped_column()
    female_holiday: Mapped[bool] = mapped_column()
    month: Mapped[Optional[int]] = mapped_column(Integer)
    day: Mapped[Optional[int]] = mapped_column(Integer)
    week_number: Mapped[Optional[int]] = mapped_column(Integer)
    day_of_week: Mapped[Optional[int]] = mapped_column(Integer)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    __mapper_args__ = {"polymorphic_on": type, "polymorphic_identity": "base"}

    @property
    def current_year_date(self) -> date:
        return self.calculate_for_year(date.today().year)

    def calculate_for_year(self, year: int) -> date:
        raise NotImplementedError()


class FixedHoliday(HolidayTemplate):
    __mapper_args__ = {"polymorphic_identity": "fixed"}

    def calculate_for_year(self, year) -> date:
        if self.month is None or self.day is None:
            raise ValueError("Month and day must be set for FixedHoliday")
        return date(year, self.month, self.day)


class FloatingHoliday(HolidayTemplate):
    __mapper_args__ = {"polymorphic_identity": "floating"}

    def calculate_for_year(self, year: int) -> date:
        if self.month is None or self.day_of_week is None or self.week_number is None:
            raise ValueError(
                "Month, day_of_week, and week_number must be set for FloatingHoliday"
            )

        first_day = date(year, self.month, 1)
        offset = (self.day_of_week - first_day.weekday()) % 7
        first_occurrence = first_day + timedelta(days=offset)

        if self.week_number == 5:
            next_month = first_day.replace(day=28) + timedelta(days=4)
            last_day = next_month - timedelta(days=next_month.day)
            return last_day - timedelta(
                days=(last_day.weekday() - self.day_of_week) % 7
            )

        return first_occurrence + timedelta(weeks=self.week_number - 1)


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
