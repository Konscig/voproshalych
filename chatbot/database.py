from typing import Optional, List
from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, BigInteger, Text, Column, DateTime, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "user"
    id: Mapped[int] = mapped_column(primary_key=True)
    vk_id: Mapped[Optional[int]] = mapped_column(BigInteger, unique=True)
    telegram_id: Mapped[Optional[int]] = mapped_column(BigInteger, unique=True)
    is_subscribed: Mapped[bool] = mapped_column()

    questions: Mapped[List["Question"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self) -> str:
        return f"User(id={self.id!r}, is_subscribed={self.is_subscribed!r})"


class Question(Base):
    __tablename__ = "question"
    id: Mapped[int] = mapped_column(primary_key=True)
    question: Mapped[str] = mapped_column(Text())
    answer: Mapped[str] = mapped_column(Text())
    score: Mapped[Optional[int]] = mapped_column()
    user_id: Mapped[int] = mapped_column(ForeignKey("user.id"))
    embedding: Mapped[Optional[Vector]] = mapped_column(Vector(312))

    user: Mapped["User"] = relationship(back_populates="questions")

    time_created = Column(DateTime(timezone=True), server_default=func.now())
    time_updated = Column(DateTime(timezone=True), onupdate=func.now())

    def __repr__(self) -> str:
        return (f"Question(question={self.question!r}, answer={self.answer!r}, "
                f"score={self.score!r})")


# migrations
if __name__ == "__main__":
    import time
    from sqlalchemy import create_engine
    from config import Config
    while True:
        try:
            engine = create_engine(Config.SQLALCHEMY_DATABASE_URI, echo=True)
            with Session(engine) as session:
                session.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
                session.commit()
            Base.metadata.create_all(engine)
            break
        except Exception as e:
            print(e)
            time.sleep(2)
