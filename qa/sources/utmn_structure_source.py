"""
Источник: Структура университета ТюмГУ
URL: https://www.utmn.ru/o-tyumgu/struktura-universiteta/

Версия на основе requests+BeautifulSoup.
Также поддерживает async/await с aiohttp.
"""

import asyncio
import logging
from typing import List, Dict, Optional

import aiohttp
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from .utmn_base_source import UTMNBaseSource


class UTMNStructureSource(UTMNBaseSource):
    """Источник: Структура университета ТюмГУ
    
    Парсит страницу с руководством и структурой университета.
    """

    def __init__(self, base_url: str = "https://www.utmn.ru", delay: float = 1.0):
        super().__init__(base_url, delay)

    def parse_structure_page(self, url: str = None) -> Optional[Dict]:
        """Парсит страницу структуры университета
        
        Args:
            url: URL страницы структуры
            
        Returns:
            Словарь с данными страницы или None
        """
        if url is None:
            url = f"{self.base_url}/o-tyumgu/struktura-universiteta/"

        html = self._get_page_html(url)
        if not html:
            return None

        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Заголовок страницы
            title_elem = soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else 'Структура университета'

            # Основной контент
            content = soup.find('main') or soup.find('div', class_='content') or soup.find('article')
            if content:
                text = content.get_text(separator=' ', strip=True)
            else:
                body = soup.find('body')
                text = body.get_text(separator=' ', strip=True) if body else ''

            return {
                'title': title,
                'text': text,
                'date': '',
                'url': url,
                'type': 'structure'
            }
        except Exception as e:
            logging.error(f"Failed to parse structure page {url}: {e}")
            return None

    def fetch_documents(self) -> List[Document]:
        """Получает все документы из источника структуры
        
        Returns:
            Список документов LangChain
        """
        logging.info("Fetching structure page...")

        documents = []
        data = self.parse_structure_page()
        if data and data['text']:
            documents.append(Document(
                page_content=data['text'],
                metadata={
                    'source_type': 'structure',
                    'url': data['url'],
                    'title': data['title'],
                    'type': data['type'],
                    'date': data['date']
                }
            ))
            logging.info(f"Fetched: {data['title']}")

        logging.info(f"Total structure documents: {len(documents)}")
        return documents

    # ========== Async методы ==========

    async def parse_structure_page_async(
        self,
        url: str = None,
        session: aiohttp.ClientSession = None
    ) -> Optional[Dict]:
        """Парсит страницу структуры университета (async)

        Args:
            url: URL страницы структуры
            session: aiohttp ClientSession

        Returns:
            Словарь с данными страницы или None
        """
        if url is None:
            url = f"{self.base_url}/o-tyumgu/struktura-universiteta/"

        if session is None:
            # Создаем временную сессию если не передана
            async with aiohttp.ClientSession(headers=self.headers) as session:
                return await self._parse_structure_page_with_session(url, session)
        return await self._parse_structure_page_with_session(url, session)

    async def _parse_structure_page_with_session(
        self,
        url: str,
        session: aiohttp.ClientSession
    ) -> Optional[Dict]:
        """Вспомогательный метод для парсинга с существующей сессией"""
        html = await self._get_page_html_async(url, session)
        if not html:
            return None

        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Заголовок страницы
            title_elem = soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else 'Структура университета'

            # Основной контент
            content = soup.find('main') or soup.find('div', class_='content') or soup.find('article')
            if content:
                text = content.get_text(separator=' ', strip=True)
            else:
                body = soup.find('body')
                text = body.get_text(separator=' ', strip=True) if body else ''

            return {
                'title': title,
                'text': text,
                'date': '',
                'url': url,
                'type': 'structure'
            }
        except Exception as e:
            logging.error(f"Failed to parse structure page {url}: {e}")
            return None

    async def fetch_documents_async(self) -> List[Document]:
        """Получает все документы (async)

        Returns:
            Список документов LangChain
        """
        logging.info("Fetching structure page (async)...")

        documents = []
        async with aiohttp.ClientSession(headers=self.headers) as session:
            data = await self.parse_structure_page_async(session=session)
            if data and data['text']:
                documents.append(Document(
                    page_content=data['text'],
                    metadata={
                        'source_type': 'structure',
                        'url': data['url'],
                        'title': data['title'],
                        'type': data['type'],
                        'date': data['date']
                    }
                ))
                logging.info(f"Fetched: {data['title']}")

        logging.info(f"Total structure documents: {len(documents)}")
        return documents
