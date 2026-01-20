"""
Источник: RSS лента новостей ТюмГУ
URL: https://www.utmn.ru/rss/

Версия на основе requests+BeautifulSoup.
Также поддерживает async/await с aiohttp.
"""

import asyncio
import logging
import xml.etree.ElementTree as ET
from typing import List, Dict, Optional

import aiohttp
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from .utmn_base_source import UTMNBaseSource


class UTMNRssSource(UTMNBaseSource):
    """Источник: RSS лента новостей ТюмГУ
    
    Парсит RSS/Atom ленту с новостями университета.
    """

    def __init__(self, base_url: str = "https://www.utmn.ru", delay: float = 1.0):
        super().__init__(base_url, delay)

    def parse_rss_feed(self, url: str = None) -> List[Dict]:
        """Парсит RSS ленту
        
        Args:
            url: URL RSS ленты
            
        Returns:
            Список словарей с данными новостей
        """
        if url is None:
            url = f"{self.base_url}/rss/"

        xml_content = self._get_page_html(url)
        if not xml_content:
            return []

        try:
            root = ET.fromstring(xml_content)
            items = []

            # RSS 2.0 формат
            channel = root.find('.//channel')
            if channel is not None:
                for item in channel.findall('.//item'):
                    title_elem = item.find('title')
                    link_elem = item.find('link')
                    desc_elem = item.find('description')
                    date_elem = item.find('pubDate')

                    title = title_elem.text if title_elem is not None else ''
                    link = link_elem.text if link_elem is not None else ''
                    description = desc_elem.text if desc_elem is not None else ''
                    date = date_elem.text if date_elem is not None else ''

                    # Очистка description от HTML тегов
                    if description:
                        soup = BeautifulSoup(description, 'html.parser')
                        description = soup.get_text(strip=True)

                    if title and link:
                        items.append({
                            'title': title,
                            'text': description or title,
                            'date': date,
                            'url': link,
                            'type': 'rss'
                        })

            logging.info(f"Parsed {len(items)} items from RSS feed")
            return items

        except ET.ParseError as e:
            logging.error(f"Failed to parse RSS feed: {e}")
            return []

    def fetch_documents(self) -> List[Document]:
        """Получает все документы из RSS ленты
        
        Returns:
            Список документов LangChain
        """
        logging.info("Fetching RSS feed...")

        documents = []
        items = self.parse_rss_feed()

        for item in items:
            if item['text']:
                documents.append(Document(
                    page_content=item['text'],
                    metadata={
                        'source_type': 'rss',
                        'url': item['url'],
                        'title': item['title'],
                        'type': item['type'],
                        'date': item['date']
                    }
                ))

        logging.info(f"Total RSS documents: {len(documents)}")
        return documents

    # ========== Async методы ==========

    async def parse_rss_feed_async(
        self,
        url: str = None,
        session: aiohttp.ClientSession = None
    ) -> List[Dict]:
        """Парсит RSS ленту (async)

        Args:
            url: URL RSS ленты
            session: aiohttp ClientSession

        Returns:
            Список словарей с данными новостей
        """
        if url is None:
            url = f"{self.base_url}/rss/"

        if session is None:
            async with aiohttp.ClientSession(headers=self.headers) as session:
                return await self._parse_rss_with_session(url, session)
        return await self._parse_rss_with_session(url, session)

    async def _parse_rss_with_session(
        self,
        url: str,
        session: aiohttp.ClientSession
    ) -> List[Dict]:
        """Вспомогательный метод для парсинга с существующей сессией"""
        xml_content = await self._get_page_html_async(url, session)
        if not xml_content:
            return []

        try:
            root = ET.fromstring(xml_content)
            items = []

            # RSS 2.0 формат
            channel = root.find('.//channel')
            if channel is not None:
                for item in channel.findall('.//item'):
                    title_elem = item.find('title')
                    link_elem = item.find('link')
                    desc_elem = item.find('description')
                    date_elem = item.find('pubDate')

                    title = title_elem.text if title_elem is not None else ''
                    link = link_elem.text if link_elem is not None else ''
                    description = desc_elem.text if desc_elem is not None else ''
                    date = date_elem.text if date_elem is not None else ''

                    # Очистка description от HTML тегов
                    if description:
                        soup = BeautifulSoup(description, 'html.parser')
                        description = soup.get_text(strip=True)

                    if title and link:
                        items.append({
                            'title': title,
                            'text': description or title,
                            'date': date,
                            'url': link,
                            'type': 'rss'
                        })

            logging.info(f"Parsed {len(items)} items from RSS feed")
            return items

        except ET.ParseError as e:
            logging.error(f"Failed to parse RSS feed: {e}")
            return []

    async def fetch_documents_async(self) -> List[Document]:
        """Получает все документы из RSS ленты (async)

        Returns:
            Список документов LangChain
        """
        logging.info("Fetching RSS feed (async)...")

        documents = []
        async with aiohttp.ClientSession(headers=self.headers) as session:
            items = await self.parse_rss_feed_async(session=session)

            for item in items:
                if item['text']:
                    documents.append(Document(
                        page_content=item['text'],
                        metadata={
                            'source_type': 'rss',
                            'url': item['url'],
                            'title': item['title'],
                            'type': item['type'],
                            'date': item['date']
                        }
                    ))

        logging.info(f"Total RSS documents: {len(documents)}")
        return documents
