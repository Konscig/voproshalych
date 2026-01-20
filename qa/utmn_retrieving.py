"""
Функции для индексации UTMN источников в базу данных
"""

import asyncio
import logging
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from database import Chunk
from sources.utmn_combined_source import UTMNCombinedSource


def reindex_utmn(
    engine: Engine,
    text_splitter: RecursiveCharacterTextSplitter,
    encoder_model: SentenceTransformer,
    max_pages: int = 5,
):
    """Пересоздаёт векторный индекс текстов из UTMN источников.

    Args:
        engine: Экземпляр подключения к БД
        text_splitter: Разделитель текста на фрагменты
        encoder_model: Модель получения векторных представлений
        max_pages: Максимальное количество страниц для источников с пагинацией
    """
    logging.warning("START CREATE UTMN INDEX")

    # Инициализируем источник
    utmn_source = UTMNCombinedSource(delay=1.0, use_confluence=False)

    try:
        # Получаем все документы
        documents = utmn_source.fetch_all_documents(max_pages=max_pages)
        logging.info(f"Fetched {len(documents)} documents from UTMN sources")

        if len(documents) == 0:
            logging.warning("No documents fetched from UTMN sources")
            return

        # Разбиваем на чанки
        all_splits = text_splitter.split_documents(documents)
        logging.info(f"Split into {len(all_splits)} chunks")

        # Удаляем старые записи UTMN из БД (по префиксу utmn_ в url)
        with Session(engine) as session:
            # Удаляем только UTMN записи (не Confluence)
            deleted = session.query(Chunk).filter(
                Chunk.url.like("https://www.utmn.ru/%")
            ).delete()
            session.commit()
            logging.info(f"Deleted {deleted} old UTMN chunks from database")

            # Сохраняем новые чанки
            saved_count = 0
            for chunk in all_splits:
                # Используем URL и source_type из метаданных
                url = chunk.metadata.get('url', '')
                source_type = chunk.metadata.get('source_type', 'other')
                if not url:
                    continue

                session.add(
                    Chunk(
                        url=url,
                        source_type=source_type,
                        text=chunk.page_content,
                        embedding=encoder_model.encode(chunk.page_content),
                    )
                )
                saved_count += 1

                # Логируем прогресс каждые 10 чанков
                if saved_count % 10 == 0:
                    logging.info(f"Saved {saved_count}/{len(all_splits)} chunks")

            session.commit()
            logging.warning(f"UTMN INDEX CREATED: {saved_count} chunks saved")

    except Exception as e:
        logging.error(f"Failed to create UTMN index: {e}")
        raise
    finally:
        utmn_source.close()


def check_utmn_index(engine: Engine) -> bool:
    """Проверяет, есть ли чанки из UTMN источников в базе данных

    Args:
        engine: Экземпляр подключения к БД

    Returns:
        True если есть хотя бы один UTMN чанк, иначе False
    """
    with Session(engine) as session:
        count = session.query(Chunk).filter(
            Chunk.url.like("https://www.utmn.ru/%")
        ).count()
        return count > 0


# ========== Async версия для параллельного парсинга ==========

async def reindex_utmn_async(
    engine: Engine,
    text_splitter: RecursiveCharacterTextSplitter,
    encoder_model: SentenceTransformer,
    max_pages: int = 5,
    max_concurrent_news_events: int = 10,
):
    """Пересоздаёт векторный индекс текстов из UTMN источников (async)

    Использует параллельный парсинг для ускорения индексации.

    Args:
        engine: Экземпляр подключения к БД
        text_splitter: Разделитель текста на фрагменты
        encoder_model: Модель получения векторных представлений
        max_pages: Максимальное количество страниц для источников с пагинацией
        max_concurrent_news_events: Макс. одновременных запросов для новостей/событий
    """
    logging.warning("START CREATE UTMN INDEX (ASYNC)")

    # Инициализируем источник
    utmn_source = UTMNCombinedSource(delay=0.3, use_confluence=False)

    try:
        # Получаем все документы (async)
        documents = await utmn_source.fetch_all_documents_async(
            max_pages=max_pages,
            max_concurrent_news_events=max_concurrent_news_events
        )
        logging.info(f"Fetched {len(documents)} documents from UTMN sources")

        if len(documents) == 0:
            logging.warning("No documents fetched from UTMN sources")
            return

        # Разбиваем на чанки
        all_splits = text_splitter.split_documents(documents)
        total_chunks = len(all_splits)
        logging.info(f"Split into {total_chunks} chunks")

        # === ОПТИМИЗАЦИЯ: Батчевое кодирование ===
        logging.info("Creating embeddings with batch processing...")
        batch_size = 32

        # Подготавливаем данные для батчевого кодирования
        chunk_texts = []
        chunk_metadata = []

        for chunk in all_splits:
            url = chunk.metadata.get('url', '')
            source_type = chunk.metadata.get('source_type', 'other')
            if not url:
                continue

            chunk_texts.append(chunk.page_content)
            chunk_metadata.append({
                'url': url,
                'source_type': source_type
            })

        logging.info(f"Encoding {len(chunk_texts)} chunks in batches of {batch_size}...")

        # Батчевое кодирование (быстро!)
        from tqdm import tqdm

        all_embeddings = []
        with tqdm(total=len(chunk_texts), desc="Creating embeddings", unit="chunk") as pbar:
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i+batch_size]
                batch_embeddings = encoder_model.encode(
                    batch_texts,
                    batch_size=batch_size,
                    show_progress_bar=False
                )
                all_embeddings.extend(batch_embeddings)

                # Обновляем прогресс
                pbar.update(len(batch_texts))

                # Промежуточный прогресс в логах
                progress_pct = ((i + len(batch_texts)) / len(chunk_texts)) * 100
                if (i + len(batch_texts)) % 100 == 0 or (i + len(batch_texts)) == len(chunk_texts):
                    logging.info(f"Progress: {progress_pct:.1f}% ({i + len(batch_texts)}/{len(chunk_texts)} chunks encoded)")

        # Удаляем старые записи UTMN из БД (по префиксу utmn_ в url)
        with Session(engine) as session:
            # Удаляем только UTMN записи (не Confluence)
            deleted = session.query(Chunk).filter(
                Chunk.url.like("https://www.utmn.ru/%")
            ).delete()
            session.commit()
            logging.info(f"Deleted {deleted} old UTMN chunks from database")

            # Сохраняем новые чанки (пачками для эффективности)
            logging.info(f"Saving {len(chunk_texts)} chunks to database...")
            commit_interval = 500

            for idx, (text, embedding, meta) in enumerate(zip(chunk_texts, all_embeddings, chunk_metadata)):
                session.add(
                    Chunk(
                        url=meta['url'],
                        source_type=meta['source_type'],
                        text=text,
                        embedding=embedding.tolist(),
                    )
                )

                # Промежуточный коммит каждые N чанков
                if (idx + 1) % commit_interval == 0:
                    session.commit()
                    progress_pct = ((idx + 1) / len(chunk_texts)) * 100
                    logging.info(f"Committed & saved: {idx + 1}/{len(chunk_texts)} chunks ({progress_pct:.1f}%)")

            # Финальный коммит
            session.commit()
            logging.warning(f"UTMN INDEX CREATED (ASYNC): {len(chunk_texts)} chunks saved")

    except Exception as e:
        logging.error(f"Failed to create UTMN index (async): {e}")
        raise
    finally:
        utmn_source.close()


def reindex_utmn_async_wrapper(
    engine: Engine,
    text_splitter: RecursiveCharacterTextSplitter,
    encoder_model: SentenceTransformer,
    max_pages: int = 5,
    max_concurrent_news_events: int = 10,
):
    """Wrapper для вызова async функции из sync кода

    Args:
        engine: Экземпляр подключения к БД
        text_splitter: Разделитель текста на фрагменты
        encoder_model: Модель получения векторных представлений
        max_pages: Максимальное количество страниц для источников с пагинацией
        max_concurrent_news_events: Макс. одновременных запросов для новостей/событий
    """
    asyncio.run(reindex_utmn_async(
        engine=engine,
        text_splitter=text_splitter,
        encoder_model=encoder_model,
        max_pages=max_pages,
        max_concurrent_news_events=max_concurrent_news_events,
    ))
