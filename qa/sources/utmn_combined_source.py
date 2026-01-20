"""
Комбинированный источник данных для ТюмГУ

Агрегирует данные из всех источников (Confluence + Web-парсеры).
Все web-парсеры используют requests вместо Selenium.
Также поддерживает async/await для параллельного парсинга.
"""

import asyncio
import logging
from typing import List

from langchain_core.documents import Document

from .utmn_base_source import UTMNBaseSource
from .utmn_contacts_source import UTMNContactsSource
from .utmn_structure_source import UTMNStructureSource
from .utmn_news_source import UTMNNewsSource
from .utmn_events_source import UTMNEventsSource
from .utmn_rss_source import UTMNRssSource


# Confluence импортируем отдельно
try:
    from confluence_retrieving import ConfluenceSource
except ImportError:
    ConfluenceSource = None


class UTMNCombinedSource:
    """Комбинированный источник данных для ТюмГУ

    Агрегирует данные из всех источников:
    - Confluence (если доступен)
    - Web-парсеры на requests (Contacts, Structure, News, Events, RSS)
    """

    def __init__(self, base_url: str = "https://www.utmn.ru", delay: float = 1.0, use_confluence: bool = True):
        """Инициализация комбинированного источника

        Args:
            base_url: Базовый URL сайта ТюмГУ
            delay: Задержка между запросами
            use_confluence: Использовать Confluence источник (если доступен)
        """
        self.base_url = base_url
        self.delay = delay

        # Инициализируем все web-источники
        self.contacts_source = UTMNContactsSource(base_url, delay)
        self.structure_source = UTMNStructureSource(base_url, delay)
        self.news_source = UTMNNewsSource(base_url, delay)
        self.events_source = UTMNEventsSource(base_url, delay)
        self.rss_source = UTMNRssSource(base_url, delay)

        # Confluence (опционально)
        self.confluence_source = None
        if use_confluence and ConfluenceSource is not None:
            try:
                self.confluence_source = ConfluenceSource()
                logging.info("Confluence source initialized")
            except Exception as e:
                logging.warning(f"Failed to initialize Confluence: {e}")

    def close(self):
        """Закрывает все источники"""
        for source in [self.contacts_source, self.structure_source,
                       self.news_source, self.events_source, self.rss_source]:
            if hasattr(source, 'close'):
                source.close()

    def fetch_all_documents(self, max_pages: int = 5) -> List[Document]:
        """Получает документы из всех источников
        
        Args:
            max_pages: Максимальное количество страниц для источников с пагинацией
            
        Returns:
            Список всех документов из всех источников
        """
        all_documents = []

        # Web-источники
        logging.info("=" * 60)
        logging.info("Fetching documents from web sources...")
        logging.info("=" * 60)

        # Contacts
        try:
            docs = self.contacts_source.fetch_documents()
            all_documents.extend(docs)
            logging.info(f"Contacts: {len(docs)} documents")
        except Exception as e:
            logging.error(f"Failed to fetch contacts: {e}")

        # Structure
        try:
            docs = self.structure_source.fetch_documents()
            all_documents.extend(docs)
            logging.info(f"Structure: {len(docs)} documents")
        except Exception as e:
            logging.error(f"Failed to fetch structure: {e}")

        # News
        try:
            docs = self.news_source.fetch_documents(max_pages=max_pages)
            all_documents.extend(docs)
            logging.info(f"News: {len(docs)} documents")
        except Exception as e:
            logging.error(f"Failed to fetch news: {e}")

        # Events
        try:
            docs = self.events_source.fetch_documents(max_pages=max_pages)
            all_documents.extend(docs)
            logging.info(f"Events: {len(docs)} documents")
        except Exception as e:
            logging.error(f"Failed to fetch events: {e}")

        # RSS
        try:
            docs = self.rss_source.fetch_documents()
            all_documents.extend(docs)
            logging.info(f"RSS: {len(docs)} documents")
        except Exception as e:
            logging.error(f"Failed to fetch RSS: {e}")

        # Confluence
        if self.confluence_source:
            logging.info("=" * 60)
            logging.info("Fetching documents from Confluence...")
            logging.info("=" * 60)
            try:
                docs = self.confluence_source.fetch_documents()
                all_documents.extend(docs)
                logging.info(f"Confluence: {len(docs)} documents")
            except Exception as e:
                logging.error(f"Failed to fetch Confluence: {e}")

        logging.info("=" * 60)
        logging.info(f"TOTAL: {len(all_documents)} documents from all sources")
        logging.info("=" * 60)

        return all_documents

    # ========== Async методы для параллельного парсинга ==========

    async def fetch_all_documents_async(
        self,
        max_pages: int = 5,
        max_concurrent_news_events: int = 10
    ) -> List[Document]:
        """Получает документы из всех источников (async)

        Args:
            max_pages: Максимальное количество страниц для источников с пагинацией
            max_concurrent_news_events: Макс. одновременных запросов для новостей/событий

        Returns:
            Список всех документов из всех источников
        """
        all_documents = []

        # Web-источники (параллельно)
        logging.info("=" * 60)
        logging.info("Fetching documents from web sources (async)...")
        logging.info("=" * 60)

        # Запускаем все источники параллельно
        tasks = [
            self._fetch_contacts_async(),
            self._fetch_structure_async(),
            self._fetch_news_async(max_pages=max_pages, max_concurrent=max_concurrent_news_events),
            self._fetch_events_async(max_pages=max_pages, max_concurrent=max_concurrent_news_events),
            self._fetch_rss_async(),
        ]

        # Ждем завершения всех задач
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Обрабатываем результаты
        sources = ['Contacts', 'Structure', 'News', 'Events', 'RSS']
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Failed to fetch {sources[i]}: {result}")
            elif result is not None:
                all_documents.extend(result)
                logging.info(f"{sources[i]}: {len(result)} documents")

        # Confluence (синхронно, так как не поддерживает async)
        if self.confluence_source:
            logging.info("=" * 60)
            logging.info("Fetching documents from Confluence (sync)...")
            logging.info("=" * 60)
            try:
                docs = self.confluence_source.fetch_documents()
                all_documents.extend(docs)
                logging.info(f"Confluence: {len(docs)} documents")
            except Exception as e:
                logging.error(f"Failed to fetch Confluence: {e}")

        logging.info("=" * 60)
        logging.info(f"TOTAL: {len(all_documents)} documents from all sources")
        logging.info("=" * 60)

        return all_documents

    async def _fetch_contacts_async(self) -> List[Document]:
        """Парсит контакты (async)"""
        try:
            return await self.contacts_source.fetch_documents_async()
        except Exception as e:
            logging.error(f"Failed to fetch contacts: {e}")
            return []

    async def _fetch_structure_async(self) -> List[Document]:
        """Парсит структуру (async)"""
        try:
            return await self.structure_source.fetch_documents_async()
        except Exception as e:
            logging.error(f"Failed to fetch structure: {e}")
            return []

    async def _fetch_news_async(self, max_pages: int = 5, max_concurrent: int = 10) -> List[Document]:
        """Парсит новости (async)"""
        try:
            return await self.news_source.fetch_documents_async(
                max_pages=max_pages,
                max_months=2,
                max_concurrent=max_concurrent
            )
        except Exception as e:
            logging.error(f"Failed to fetch news: {e}")
            return []

    async def _fetch_events_async(self, max_pages: int = 5, max_concurrent: int = 10) -> List[Document]:
        """Парсит события (async)"""
        try:
            return await self.events_source.fetch_documents_async(
                max_pages=max_pages,
                max_months=2,
                max_concurrent=max_concurrent
            )
        except Exception as e:
            logging.error(f"Failed to fetch events: {e}")
            return []

    async def _fetch_rss_async(self) -> List[Document]:
        """Парсит RSS (async)"""
        try:
            return await self.rss_source.fetch_documents_async()
        except Exception as e:
            logging.error(f"Failed to fetch RSS: {e}")
            return []
