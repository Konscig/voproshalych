"""
Базовый класс для источников данных utmn.ru на основе requests+BeautifulSoup

Легкая альтернатива Selenium-based базовому классу.
Использует requests с правильными headers для обхода базовой защиты.

Также поддерживает async/await с aiohttp для параллельного парсинга.
"""

import asyncio
import logging
import time
from typing import List, Dict, Optional

import aiohttp
import requests
from bs4 import BeautifulSoup


class UTMNBaseSource:
    """Базовый класс для источников данных из utmn.ru на основе requests

    Использует requests + BeautifulSoup вместо Selenium для более быстрой
    и легкой работы с веб-страницами.
    """

    def __init__(self, base_url: str = "https://www.utmn.ru", delay: float = 1.0):
        """Инициализация источника

        Args:
            base_url: Базовый URL сайта ТюмГУ
            delay: Задержка между запросами в секундах (для избежания блокировки)
        """
        self.base_url = base_url
        self.delay = delay
        self.session = requests.Session()

        # Настройка headers для имитации обычного браузера
        # Упрощенная версия, похожая на curl
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': '*/*',
        }
        self.session.headers.update(self.headers)

    def __del__(self):
        """Закрывает сессию при удалении объекта"""
        self.close()

    def close(self):
        """Закрывает сессию requests"""
        if self.session:
            try:
                self.session.close()
                logging.info("Requests session closed")
            except Exception as e:
                logging.error(f"Error closing session: {e}")

    def _get_page_html(self, url: str, timeout: int = 30) -> Optional[str]:
        """Получает HTML страницы используя requests

        Args:
            url: URL страницы
            timeout: Таймаут запроса

        Returns:
            HTML текст или None в случае ошибки
        """
        try:
            logging.info(f"Loading page: {url}")

            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()

            # Проверка на Forbidden
            if response.status_code == 403:
                logging.warning(f"Forbidden (403) for URL: {url}")
                return None

            html = response.text

            # Если страница слишком короткая, вероятно это ошибка
            if len(html) < 1000:
                logging.warning(f"Page seems too short ({len(html)} chars): {url}")
                logging.warning(f"Page content: {html[:500]}")

            return html
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to get page {url}: {e}")
            return None

    def _get_soup(self, url: str, timeout: int = 30) -> Optional[BeautifulSoup]:
        """Получает BeautifulSoup объект для страницы

        Args:
            url: URL страницы
            timeout: Таймаут запроса

        Returns:
            BeautifulSoup объект или None в случае ошибки
        """
        html = self._get_page_html(url, timeout)
        if html:
            return BeautifulSoup(html, 'html.parser')
        return None

    def _sleep(self):
        """Задержка между запросами"""
        if self.delay > 0:
            time.sleep(self.delay)

    # ========== Async методы для параллельного парсинга ==========

    async def _get_page_html_async(
        self,
        url: str,
        session: aiohttp.ClientSession,
        timeout: int = 30
    ) -> Optional[str]:
        """Получает HTML страницы используя aiohttp (async)

        Args:
            url: URL страницы
            session: aiohttp ClientSession
            timeout: Таймаут запроса

        Returns:
            HTML текст или None в случае ошибки
        """
        try:
            logging.debug(f"Loading page (async): {url}")

            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                response.raise_for_status()

                if response.status == 403:
                    logging.warning(f"Forbidden (403) for URL: {url}")
                    return None

                html = await response.text()

                # Если страница слишком короткая, вероятно это ошибка
                if len(html) < 1000:
                    logging.warning(f"Page seems too short ({len(html)} chars): {url}")
                    logging.warning(f"Page content: {html[:500]}")

                return html

        except asyncio.TimeoutError:
            logging.error(f"Timeout while fetching {url}")
            return None
        except aiohttp.ClientError as e:
            logging.error(f"Failed to get page {url}: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error fetching {url}: {e}")
            return None

    async def _get_soup_async(
        self,
        url: str,
        session: aiohttp.ClientSession,
        timeout: int = 30
    ) -> Optional[BeautifulSoup]:
        """Получает BeautifulSoup объект для страницы (async)

        Args:
            url: URL страницы
            session: aiohttp ClientSession
            timeout: Таймаут запроса

        Returns:
            BeautifulSoup объект или None в случае ошибки
        """
        html = await self._get_page_html_async(url, session, timeout)
        if html:
            return BeautifulSoup(html, 'html.parser')
        return None
