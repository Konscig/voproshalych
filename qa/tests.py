from atlassian import Confluence
from config import Config
from main import get_answer
from time import sleep
from confluence_retrieving import get_document_content_by_id
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from database import Chunk
from utmn_retrieving import check_utmn_index, reindex_utmn
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


def test_llm():
    """тест взаимодействия с LLM"""

    context = """Чтобы поменять занятия по физической культуре на посещение
    частного фитнес-клуба, необходимо заполнить заявление и обратиться
    к курирующему преподавателю"""
    question = "Что нужно сделать, чтобы заменить физру на частный клуб?"
    assert "препод" in get_answer([], context, question).lower()
    sleep(1.1)
    assert "не найден" in get_answer([], context, "Когда наступит лето?").lower()


def test_confluence():
    """тест взаимодействия с Confluence"""

    confluence = Confluence(url=Config.CONFLUENCE_HOST, token=Config.CONFLUENCE_TOKEN)
    main_space = confluence.get_space(
        Config.CONFLUENCE_SPACES[0], expand="description.plain,homepage"
    )
    page_content, page_link = get_document_content_by_id(
        confluence, str(main_space["homepage"]["id"])
    )
    assert page_content is not None
    assert len(page_content) > 10
    assert (
        page_link
        == main_space["_links"]["base"] + main_space["homepage"]["_links"]["webui"]
    )


def test_utmn_indexing():
    """тест индексации UTMN источников в базу данных"""

    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4096,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )
    encoder_model = SentenceTransformer(
        "saved_models/multilingual-e5-large-wikiutmn", device="cpu"
    )

    try:
        # Запускаем реиндекс UTMN с ограничением по страницам (для скорости теста)
        reindex_utmn(
            engine=engine,
            text_splitter=text_splitter,
            encoder_model=encoder_model,
            max_pages=2  # Только 2 страницы для быстрого теста
        )

        # Проверяем, что чанки появились в БД
        with Session(engine) as session:
            # Ищем UTMN чанки (URL начинаются с https://www.utmn.ru/)
            utmn_chunks = session.query(Chunk).filter(
                Chunk.url.like("https://www.utmn.ru/%")
            ).all()

            # Должен быть хотя бы один чанк
            assert len(utmn_chunks) > 0, "Должен быть создан хотя бы один UTMN чанк"

            # Проверяем структуру первого чанка
            first_chunk = utmn_chunks[0]
            assert first_chunk.text is not None, "Текст чанка не должен быть пустым"
            assert len(first_chunk.text) > 0, "Текст чанка должен содержать символы"
            assert first_chunk.embedding is not None, "Эмбеддинг должен быть создан"
            assert first_chunk.url.startswith("https://www.utmn.ru/"), \
                "URL должен указывать на UTMN сайт"

        print(f"Тест пройден: создано {len(utmn_chunks)} UTMN чанков")

    finally:
        # Очистка: удаляем созданные чанки
        with Session(engine) as session:
            deleted = session.query(Chunk).filter(
                Chunk.url.like("https://www.utmn.ru/%")
            ).delete()
            session.commit()
            print(f"Очистка: удалено {deleted} тестовых чанков")


def test_check_utmn_index():
    """тест проверки наличия UTMN индекса"""

    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)

    # Перед тестом убедимся, что нет UTMN чанков (чистая БД для UTMN)
    with Session(engine) as session:
        session.query(Chunk).filter(
            Chunk.url.like("https://www.utmn.ru/%")
        ).delete()
        session.commit()

    # Проверяем, что функция возвращает False когда нет чанков
    has_index = check_utmn_index(engine)
    assert not has_index, "check_utmn_index должен возвращать False когда нет чанков"

    print("Тест check_utmn_index пройден")


# ========== Async тесты для проверки полноты парсинга UTMN источников ==========

import asyncio
import pytest
from sources.utmn_employees_source import UTMNEmployeesSource
from sources.utmn_news_source import UTMNNewsSource
from sources.utmn_events_source import UTMNEventsSource


@pytest.mark.asyncio
async def test_employees_fetch_documents():
    """Проверка, что парсятся все сотрудники (async)"""
    source = UTMNEmployeesSource(delay=0.5)

    # Используем max_pages=3 для быстрого теста (вместо None для полного парсинга)
    documents = await source.fetch_documents_async(
        max_pages=3,
        max_concurrent=5  # Меньше для теста
    )

    # Для max_pages=3 должно быть как минимум 3*20=60 сотрудников
    assert len(documents) >= 60, f"Expected >=60 employees for 3 pages, got {len(documents)}"

    # Проверка source_type
    for doc in documents:
        assert doc.metadata.get('source_type') == 'employees', "source_type should be 'employees'"
        assert 'utmn.ru/sotrudniki' in doc.metadata.get('url', ''), "URL should point to employees"

    print(f"Тест test_employees_fetch_documents пройден: {len(documents)} сотрудников")


@pytest.mark.asyncio
async def test_news_fetch_documents():
    """Проверка, что парсятся новости за 2+ месяца (async)"""
    source = UTMNNewsSource(delay=0.5)

    # Используем max_pages=3 для быстрого теста
    documents = await source.fetch_documents_async(
        max_pages=3,
        max_months=2,
        max_concurrent=5
    )

    # Для 3 страниц должно быть как минимум 3*14=42 новости (до фильтрации по дате)
    # После фильтрации по дате может быть меньше, но хотя бы 10
    assert len(documents) >= 10, f"Expected >=10 news articles for 3 pages, got {len(documents)}"

    # Проверка source_type
    for doc in documents:
        assert doc.metadata.get('source_type') == 'news', "source_type should be 'news'"
        assert 'utmn.ru/news' in doc.metadata.get('url', ''), "URL should point to news"

    print(f"Тест test_news_fetch_documents пройден: {len(documents)} новостей")


@pytest.mark.asyncio
async def test_events_fetch_documents():
    """Проверка, что парсятся события за 2+ месяца (async)"""
    source = UTMNEventsSource(delay=0.5)

    # Используем max_pages=3 для быстрого теста
    documents = await source.fetch_documents_async(
        max_pages=3,
        max_months=2,
        max_concurrent=5
    )

    # Для 3 страниц должно быть как минимум 3*14=42 события (до фильтрации по дате)
    # После фильтрации по дате может быть меньше, но хотя бы 10
    assert len(documents) >= 10, f"Expected >=10 events for 3 pages, got {len(documents)}"

    # Проверка source_type
    for doc in documents:
        assert doc.metadata.get('source_type') == 'events', "source_type should be 'events'"
        assert 'utmn.ru/events' in doc.metadata.get('url', ''), "URL should point to events"

    print(f"Тест test_events_fetch_documents пройден: {len(documents)} событий")

