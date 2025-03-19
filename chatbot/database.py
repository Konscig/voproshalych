from datetime import datetime, timedelta
from typing import Optional, List
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Engine,
    ForeignKey,
    Text,
    Enum,
    Integer,
    func,
    select,
    and_,
)
from sqlalchemy.orm import (
    Mapped,
    Session,
    declarative_base,
    mapped_column,
    relationship,
)
from datetime import timedelta, date

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


class HolidayTemplate(Base):
    """Шаблон праздника

    Args:
        id (int): id шаблона.
        name (str): название праздника.
        type (str): тип праздника (fixed или floating).
        template_llm (str): шаблон для LLM.
        vk (bool): флаг для VK.
        tg (bool): флаг для TG.
        male_holiday (bool): флаг для мужского праздника.
        female_holiday (bool): флаг для женского праздника.
        month (int | None): месяц (для fixed).
        day (int | None): день (для fixed).
        week_number (int | None): номер недели (для floating).
        day_of_week (int | None): день недели (для floating).
        created_at (datetime): время создания модели.
        updated_at (datetime): время обновления модели.
    """

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
    """Фиксированный праздник.

    Наследует все атрибуты и методы от HolidayTemplate.
    """

    __mapper_args__ = {"polymorphic_identity": "fixed"}

    def calculate_for_year(self, year) -> date:
        if self.month is None or self.day is None:
            raise ValueError("Month and day must be set for FixedHoliday")
        return date(year, self.month, self.day)


class FloatingHoliday(HolidayTemplate):
    """Плавающий праздник.

    Наследует все атрибуты и методы от HolidayTemplate.
    """

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


def add_user(
    engine: Engine, vk_id: int | None = None, telegram_id: int | None = None
) -> tuple[bool, int]:
    """Функция добавления в БД пользователя виртуального помощника

    Args:
        engine (Engine): подключение к БД
        vk_id (int | None): id пользователя ВКонтакте
        telegram_id (int | None): id пользователя Telegram

    Raises:
        TypeError: vk_id и telegram_id не могут быть None в одно время

    Returns:
        tuple[bool, int]: добавился пользователь или нет, какой у него id в БД
    """

    with Session(engine) as session:
        if vk_id is not None:
            user = session.scalar(select(User).where(User.vk_id == vk_id))
        elif telegram_id is not None:
            user = session.scalar(select(User).where(User.telegram_id == telegram_id))
        else:
            raise TypeError("vk_id and telegram_id can't be None at the same time")
        if user is None:
            user = User(vk_id=vk_id, telegram_id=telegram_id, is_subscribed=True)
            session.add(user)
            session.commit()
            return True, user.id
        return False, user.id


def get_user_id(
    engine: Engine, vk_id: int | None = None, telegram_id: int | None = None
) -> int | None:
    """Функция получения из БД пользователя

    Args:
        engine (Engine): подключение к БД
        vk_id (int | None): id пользователя ВКонтакте
        telegram_id (int | None): id пользователя Telegram

    Raises:
        TypeError: vk_id и telegram_id не могут быть None в одно время

    Returns:
        int | None: id пользователя или None
    """

    with Session(engine) as session:
        if vk_id is not None:
            user = session.scalar(select(User).where(User.vk_id == vk_id))
        elif telegram_id is not None:
            user = session.scalar(select(User).where(User.telegram_id == telegram_id))
        else:
            raise TypeError("vk_id and telegram_id can't be None at the same time")
        if user is None:
            return None
        return user.id


def subscribe_user(engine: Engine, user_id: int) -> bool:
    """Функция оформления подписки пользователя на рассылку

    Args:
        engine (Engine): подключение к БД
        user_id (int): id пользователя

    Returns:
        bool: подписан пользователь или нет
    """

    with Session(engine) as session:
        user = session.scalars(select(User).where(User.id == user_id)).first()
        if user is None:
            return False
        user.is_subscribed = not user.is_subscribed
        session.commit()
        return user.is_subscribed


def check_subscribing(engine: Engine, user_id: int) -> bool:
    """Функция проверки подписки пользователя на рассылку

    Args:
        engine (Engine): подключение к БД
        user_id (int): id пользователя

    Returns:
        bool: подписан пользователь или нет
    """

    with Session(engine) as session:
        user = session.scalars(select(User).where(User.id == user_id)).first()
        if user is None:
            return False
        return user.is_subscribed


def check_spam(engine: Engine, user_id: int) -> bool:
    """Функция проверки на спам

    Args:
        engine (Engine): подключение к БД
        user_id (int): id пользователя

    Returns:
        bool: пользователь задал пять вопросов за последнюю минуту
    """

    with Session(engine) as session:
        user = session.scalars(select(User).where(User.id == user_id)).first()
        if user is None:
            return False
        if len(user.question_answers) > 5:
            minute_ago = datetime.now() - timedelta(minutes=1)
            fifth_message_date = user.question_answers[4].created_at
            return minute_ago < fifth_message_date.replace(tzinfo=None)
        return False


def add_question_answer(
    engine: Engine, question: str, answer: str, confluence_url: str | None, user_id: int
) -> int:
    """Функция добавления в БД вопроса пользователя с ответом на него

    Args:
        engine (Engine): подключение к БД
        question (str): вопрос пользователя
        answer (str): ответ на вопрос пользователя
        confluence_url (str | None): ссылка на страницу в вики-системе, содержащую ответ
        user_id (int): id пользователя

    Returns:
        int: id вопроса с ответом на него
    """

    with Session(engine) as session:
        question_answer = QuestionAnswer(
            question=question,
            answer=answer,
            confluence_url=confluence_url,
            user_id=user_id,
        )
        session.add(question_answer)
        session.flush()
        session.refresh(question_answer)
        session.commit()
        if question_answer.id is None:
            return 0
        return question_answer.id


def rate_answer(engine: Engine, question_answer_id: int, score: int) -> bool:
    """Функция оценивания ответа на вопрос

    Args:
        engine (Engine): подключение к БД
        question_answer_id (int): id вопроса с ответом
        score (int): оценка ответа

    Returns:
        bool: удалось добавить в БД оценку ответа или нет
    """

    with Session(engine) as session:
        question_answer = session.scalars(
            select(QuestionAnswer).where(QuestionAnswer.id == question_answer_id)
        ).first()
        if question_answer is None:
            return False
        question_answer.score = score
        session.commit()
        return True


def get_subscribed_users(engine: Engine) -> tuple[list[int | None], list[int | None]]:
    """Функция для получения подписанных на рассылки пользователей

    Args:
        engine (Engine): подключение к БД

    Returns:
        tuple[list[int], list[int]]: кортеж списков с id пользователей VK и Telegram
    """
    with Session(engine) as session:
        vk_users = [
            user.vk_id
            for user in session.execute(
                select(User).where(and_(User.vk_id != None, User.is_subscribed))
            ).scalars()
        ]
        tg_users = [
            user.telegram_id
            for user in session.execute(
                select(User).where(and_(User.telegram_id != None, User.is_subscribed))
            ).scalars()
        ]
    return vk_users, tg_users


def get_today_holidays(engine: Engine) -> List[HolidayTemplate]:
    """Получает список праздников на сегодня.

    Args:
        engine (Engine): подключение к БД

    Returns:
        List[HolidayTemplate]: список праздников
    """
    with Session(engine) as session:
        today = date.today()
        holidays = []

        fixed_holidays = session.query(FixedHoliday).all()
        floating_holidays = session.query(FloatingHoliday).all()

        for holiday in fixed_holidays:
            if holiday.calculate_for_year(today.year) == today:
                holidays.append(holiday)

        for holiday in floating_holidays:
            if holiday.calculate_for_year(today.year) == today:
                holidays.append(holiday)

        return holidays
