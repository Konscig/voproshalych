"""
Источник: События и мероприятия ТюмГУ
URL: https://www.utmn.ru/news/events/

Версия на основе requests+BeautifulSoup.
Также поддерживает async/await с aiohttp для параллельного парсинга.
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional, Tuple

import aiohttp
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from .utmn_base_source import UTMNBaseSource


# Месяцы на русском для парсинга дат
RUSSIAN_MONTHS = {
    'янв': 1, 'января': 1, 'январь': 1,
    'фев': 2, 'февраля': 2, 'февраль': 2,
    'мар': 3, 'марта': 3, 'март': 3,
    'апр': 4, 'апреля': 4, 'апрель': 4,
    'май': 5, 'мая': 5,
    'июн': 6, 'июня': 6, 'июнь': 6,
    'июл': 7, 'июля': 7, 'июль': 7,
    'авг': 8, 'августа': 8, 'август': 8,
    'сен': 9, 'сентября': 9, 'сентябрь': 9,
    'окт': 10, 'октября': 10, 'октябрь': 10,
    'ноя': 11, 'ноября': 11, 'ноябрь': 11,
    'дек': 12, 'декабря': 12, 'декабрь': 12
}


def parse_russian_date(date_str: str) -> Optional[datetime]:
    """Парсит дату из русского формата

    Args:
        date_str: Строка с датой (различные форматы)

    Returns:
        datetime объект или None если парсинг не удался
    """
    if not date_str:
        return None

    date_str = date_str.strip().lower()

    # Различные форматы дат
    # "19 января 2026"
    match = re.match(r'(\d{1,2})\s+(\w+)\s+(\d{4})', date_str)
    if match:
        day, month_str, year = match.groups()
        month = RUSSIAN_MONTHS.get(month_str)
        if month:
            try:
                return datetime(int(year), month, int(day), tzinfo=timezone.utc)
            except ValueError:
                pass

    # "января 19 2026"
    parts = date_str.split()
    if len(parts) == 3:
        for i, part in enumerate(parts):
            if part in RUSSIAN_MONTHS:
                month_str = parts[i]
                # День может быть до или после месяца
                other_parts = [p for p in parts if p != month_str]
                if len(other_parts) == 2:
                    # Пытаемся определить день и год
                    for part in other_parts:
                        if part.isdigit():
                            day_candidate = int(part)
                            if 1 <= day_candidate <= 31:
                                day = day_candidate
                                other = [p for p in other_parts if p != str(day)][0]
                                if other.isdigit() and len(other) == 4:
                                    year = int(other)
                                    month = RUSSIAN_MONTHS[month_str]
                                    try:
                                        return datetime(year, month, day, tzinfo=timezone.utc)
                                    except ValueError:
                                        pass

    return None


def is_within_months(date: datetime, months: int = 2) -> bool:
    """Проверяет, что дата не старше указанного количества месяцев

    Args:
        date: Проверяемая дата
        months: Количество месяцев (по умолчанию 2)

    Returns:
        True если дата не старше указанного количества месяцев
    """
    if not date:
        return True

    now = datetime.now(timezone.utc)
    threshold = now - timedelta(days=months * 30)

    return date >= threshold


class UTMNEventsSource(UTMNBaseSource):
    """Источник: События и мероприятия ТюмГУ
    
    Парсит страницы с событиями и мероприятиями университета.
    """

    def __init__(self, base_url: str = "https://www.utmn.ru", delay: float = 1.0):
        super().__init__(base_url, delay)

    def _parse_event(self, soup: BeautifulSoup, url: str) -> Dict:
        """Парсит страницу события

        Args:
            soup: BeautifulSoup объект
            url: URL страницы

        Returns:
            Словарь с данными события
        """
        # Заголовок
        title_elem = soup.find('h1')
        title = title_elem.get_text(strip=True) if title_elem else ''

        # Дата и время проведения
        date_elem = soup.find('time') or soup.find(class_='event-date')
        date_str = date_elem.get_text(strip=True) if date_elem else ''
        parsed_date = parse_russian_date(date_str) if date_str else None

        # Текст
        content = soup.find('div', class_='event-content') or soup.find('article')
        if content:
            text = content.get_text(separator=' ', strip=True)
        else:
            body = soup.find('body')
            text = body.get_text(separator=' ', strip=True) if body else ''

        return {
            'title': title,
            'text': text,
            'date': date_str,
            'parsed_date': parsed_date,
            'url': url,
            'type': 'event'
        }

    def parse_page(self, url: str) -> Optional[Dict]:
        """Парсит одну страницу события
        
        Args:
            url: URL страницы для парсинга
            
        Returns:
            Словарь с данными страницы или None
        """
        html = self._get_page_html(url)
        if not html:
            return None

        try:
            soup = BeautifulSoup(html, 'html.parser')
            return self._parse_event(soup, url)
        except Exception as e:
            logging.error(f"Failed to parse {url}: {e}")
            return None

    def _get_events_urls_from_page(self, soup: BeautifulSoup) -> List[Tuple[str, Optional[datetime]]]:
        """Получает URL событий и их даты из одной страницы

        Args:
            soup: BeautifulSoup объект страницы

        Returns:
            Список кортежей (URL, parsed_date)
        """
        results = []

        # Ищем все карточки событий
        event_items = soup.find_all('article', class_=lambda x: x and 'article' in str(x).split())

        for item in event_items:
            # Извлекаем дату из карточки
            date_elem = item.find('div', class_='date')
            parsed_date = None
            if date_elem:
                month = date_elem.find('div', class_='month')
                day = date_elem.find('div', class_='day')
                year = date_elem.find('div', class_='year')
                if month and day and year:
                    date_str = f"{month.get_text(strip=True)} {day.get_text(strip=True)} {year.get_text(strip=True)}"
                    parsed_date = parse_russian_date(date_str)

            # Ищем ссылку на событие
            link = item.find('a', href=lambda x: x and '/news/events/' in x)
            if link:
                href = link.get('href')
                if href and '/news/events/' in href and href != '/news/events/':
                    if href.startswith('/'):
                        full_url = f"{self.base_url}{href}"
                    else:
                        full_url = href

                    # Берем только URL с ID события
                    url_parts = full_url.rstrip('/').split('/')
                    has_id = any(part.isdigit() for part in url_parts)

                    if has_id:
                        results.append((full_url, parsed_date))

        return results

    def _get_all_events_urls(self, max_pages: int = 5, max_months: int = 2) -> List[str]:
        """Получает все URLы событий с фильтрацией по дате (синхронный метод для совместимости)

        Args:
            max_pages: Максимальное количество страниц (по умолчанию 5, None = без ограничений по дате)
            max_months: Максимальный возраст событий в месяцах (по умолчанию 2)

        Returns:
            Список URLов событий
        """
        all_urls = []
        page_num = 1
        consecutive_old_events = 0
        max_consecutive_old = 3

        while True:
            if page_num == 1:
                page_url = f"{self.base_url}/news/events/"
            else:
                page_url = f"{self.base_url}/news/events/?PAGEN_1={page_num}"

            if max_pages is not None and page_num > max_pages:
                logging.info(f"Reached max_pages limit ({max_pages}), stopping pagination")
                break

            logging.info(f"Fetching events page {page_num}...")
            html = self._get_page_html(page_url)

            if not html:
                logging.warning(f"Failed to get events page {page_num}")
                consecutive_old_events += 1
                if consecutive_old_events >= max_consecutive_old:
                    break
                page_num += 1
                continue

            soup = BeautifulSoup(html, 'html.parser')
            page_results = self._get_events_urls_from_page(soup)  # Теперь возвращает (URL, date)

            if len(page_results) == 0:
                consecutive_old_events += 1
                if consecutive_old_events >= max_consecutive_old:
                    break
                page_num += 1
                self._sleep()
                continue

            # Фильтруем по дате прямо из списка
            page_valid_count = 0
            for url, parsed_date in page_results:
                # Если даты нет или она в пределах max_months - добавляем
                if parsed_date is None or is_within_months(parsed_date, max_months):
                    if url not in all_urls:
                        all_urls.append(url)
                        page_valid_count += 1

            if page_valid_count == 0:
                consecutive_old_events += 1
                logging.info(f"Page {page_num}: no valid events within {max_months} months (consecutive: {consecutive_old_events})")
                if consecutive_old_events >= max_consecutive_old:
                    break
            else:
                consecutive_old_events = 0
                logging.info(f"Page {page_num}: found {page_valid_count} valid event URLs (total: {len(all_urls)})")

            self._sleep()
            page_num += 1

        logging.info(f"Total events URLs fetched: {len(all_urls)}")
        return all_urls

    async def _get_all_events_urls_async(
        self,
        max_pages: int = 5,
        max_months: int = 2,
        session: aiohttp.ClientSession = None
    ) -> List[str]:
        """Получает все URLы событий с async-фильтрацией по дате

        Args:
            max_pages: Максимальное количество страниц
            max_months: Максимальный возраст событий в месяцах
            session: aiohttp ClientSession

        Returns:
            Список URLов событий
        """
        if session is None:
            raise ValueError("session parameter is required for async method")

        all_urls = []
        page_num = 1
        consecutive_old_events = 0
        max_consecutive_old = 3

        while True:
            if page_num == 1:
                page_url = f"{self.base_url}/news/events/"
            else:
                page_url = f"{self.base_url}/news/events/?PAGEN_1={page_num}"

            if max_pages is not None and page_num > max_pages:
                logging.info(f"Reached max_pages limit ({max_pages}), stopping pagination")
                break

            logging.info(f"Fetching events page {page_num}...")
            html = await self._get_page_html_async(page_url, session)

            if not html:
                logging.warning(f"Failed to get events page {page_num}")
                consecutive_old_events += 1
                if consecutive_old_events >= max_consecutive_old:
                    break
                page_num += 1
                continue

            soup = BeautifulSoup(html, 'html.parser')
            page_results = self._get_events_urls_from_page(soup)  # Возвращает (URL, date)

            if len(page_results) == 0:
                consecutive_old_events += 1
                if consecutive_old_events >= max_consecutive_old:
                    break
                page_num += 1
                continue

            # Фильтруем по дате прямо из списка (без дополнительных HTTP запросов!)
            page_valid_count = 0
            for url, parsed_date in page_results:
                if parsed_date is None or is_within_months(parsed_date, max_months):
                    if url not in all_urls:
                        all_urls.append(url)
                        page_valid_count += 1

            if page_valid_count == 0:
                consecutive_old_events += 1
                logging.info(f"Page {page_num}: no valid events within {max_months} months (consecutive: {consecutive_old_events})")
                if consecutive_old_events >= max_consecutive_old:
                    break
            else:
                consecutive_old_events = 0
                logging.info(f"Page {page_num}: found {page_valid_count} valid event URLs (total: {len(all_urls)})")

            page_num += 1

        logging.info(f"Total events URLs fetched: {len(all_urls)}")
        return all_urls

    def fetch_documents(self, max_pages: int = 5, max_months: int = 2) -> List[Document]:
        """Получает документы из источника событий

        Args:
            max_pages: Максимальное количество страниц (резерв)
            max_months: Максимальный возраст событий в месяцах (по умолчанию 2)

        Returns:
            Список документов LangChain
        """
        all_documents = []

        logging.info(f"Fetching events pages (max {max_months} months old)...")
        events_urls = self._get_all_events_urls(max_pages=max_pages, max_months=max_months)

        for i, url in enumerate(events_urls, 1):
            logging.info(f"Fetching event {i}/{len(events_urls)}: {url}")
            data = self.parse_page(url)
            if data and data['text']:
                # Дополнительная проверка даты
                parsed_date = data.get('parsed_date')
                if parsed_date and not is_within_months(parsed_date, max_months):
                    logging.info(f"Skipping {url} - too old ({data['date']})")
                    continue

                all_documents.append(Document(
                    page_content=data['text'],
                    metadata={
                        'source_type': 'events',
                        'url': data['url'],
                        'title': data['title'],
                        'type': data['type'],
                        'date': data['date']
                    }
                ))
            self._sleep()

        logging.info(f"Total events documents: {len(all_documents)}")
        return all_documents

    def fetch_latest(self, days: int = 7) -> List[Document]:
        """Получает только свежие события за указанное количество дней

        Args:
            days: Количество дней (по умолчанию 7)

        Returns:
            Список документов LangChain
        """
        all_documents = []

        # Используем 1 месяц как ceiling
        max_months = 1

        logging.info(f"Fetching latest events (last {days} days)...")
        events_urls = self._get_all_events_urls(max_pages=1, max_months=max_months)

        now = datetime.now(timezone.utc)
        threshold = now - timedelta(days=days)

        for i, url in enumerate(events_urls, 1):
            logging.info(f"Fetching event {i}/{len(events_urls)}: {url}")
            data = self.parse_page(url)
            if data and data['text']:
                parsed_date = data.get('parsed_date')
                if parsed_date:
                    if parsed_date >= threshold:
                        all_documents.append(Document(
                            page_content=data['text'],
                            metadata={
                                'source_type': 'events',
                                'url': data['url'],
                                'title': data['title'],
                                'type': data['type'],
                                'date': data['date']
                            }
                        ))
                    else:
                        logging.info(f"Skipping {url} - too old ({data['date']})")
                else:
                    # Если дату не удалось распарсить, включаем событие
                    all_documents.append(Document(
                        page_content=data['text'],
                        metadata={
                            'source_type': 'events',
                            'url': data['url'],
                            'title': data['title'],
                            'type': data['type'],
                            'date': data['date']
                        }
                    ))
            else:
                self._sleep()

        logging.info(f"Total latest events documents: {len(all_documents)}")
        return all_documents

    # ========== Async методы для параллельного парсинга ==========

    async def parse_page_async(
        self,
        url: str,
        session: aiohttp.ClientSession
    ) -> Optional[Dict]:
        """Парсит одну страницу события (async)

        Args:
            url: URL страницы для парсинга
            session: aiohttp ClientSession

        Returns:
            Словарь с данными страницы или None в случае ошибки
        """
        html = await self._get_page_html_async(url, session)
        if not html:
            return None

        try:
            soup = BeautifulSoup(html, 'html.parser')
            return self._parse_event(soup, url)
        except Exception as e:
            logging.error(f"Failed to parse {url}: {e}")
            return None

    async def fetch_documents_async(
        self,
        max_pages: int = 5,
        max_months: int = 2,
        max_concurrent: int = 10
    ) -> List[Document]:
        """Получает документы с параллельным парсингом (async)

        Args:
            max_pages: Максимальное количество страниц (резерв)
            max_months: Максимальный возраст событий в месяцах (по умолчанию 2)
            max_concurrent: Максимальное количество одновременных запросов (по умолчанию 10)

        Returns:
            Список документов LangChain
        """
        # Создаем session для async запросов
        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Получаем URLы с async-фильтрацией по дате
            events_urls = await self._get_all_events_urls_async(
                max_pages=max_pages,
                max_months=max_months,
                session=session
            )

            if not events_urls:
                logging.warning("No events URLs found")
                return []

            logging.info(f"Fetching {len(events_urls)} event pages with {max_concurrent} concurrent requests...")

            all_documents = []
            semaphore = asyncio.Semaphore(max_concurrent)

            async def fetch_with_semaphore(url: str, index: int) -> Optional[Document]:
                """Парсит страницу с ограничением семафора"""
                async with semaphore:
                    logging.info(f"Fetching event {index + 1}/{len(events_urls)}: {url}")
                    data = await self.parse_page_async(url, session)

                    if data and data['text']:
                        return Document(
                            page_content=data['text'],
                            metadata={
                                'source_type': 'events',
                                'url': data['url'],
                                'title': data['title'],
                                'type': data['type'],
                                'date': data['date']
                            }
                        )
                    return None

            tasks = [
                fetch_with_semaphore(url, i)
                for i, url in enumerate(events_urls)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Task failed with exception: {result}")
                elif result is not None:
                    all_documents.append(result)

        logging.info(f"Total events documents fetched: {len(all_documents)}")
        return all_documents

