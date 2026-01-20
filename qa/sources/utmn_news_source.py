"""
Источник: Новости Тюменского государственного университета
URL: https://www.utmn.ru/news/stories/

Версия на основе requests+BeautifulSoup вместо Selenium.
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
    'янв': 1, 'январь': 1,
    'фев': 2, 'февраль': 2,
    'мар': 3, 'март': 3,
    'апр': 4, 'апрель': 4,
    'май': 5,
    'июн': 6, 'июнь': 6,
    'июл': 7, 'июль': 7,
    'авг': 8, 'август': 8,
    'сен': 9, 'сентябрь': 9,
    'окт': 10, 'октябрь': 10,
    'ноя': 11, 'ноябрь': 11,
    'дек': 12, 'декабрь': 12
}


def parse_russian_date(date_str: str) -> Optional[datetime]:
    """Парсит дату из русского формата

    Args:
        date_str: Строка с датой (например, "янв 19 2026" или "19 января 2026")

    Returns:
        datetime объект или None если парсинг не удался
    """
    if not date_str:
        return None

    date_str = date_str.strip().lower()

    # Формат: "янв 19 2026"
    parts = date_str.split()
    if len(parts) == 3:
        month_str, day, year = parts
        month = RUSSIAN_MONTHS.get(month_str)
        if month:
            try:
                return datetime(int(year), month, int(day), tzinfo=timezone.utc)
            except ValueError:
                pass

    # Формат: "19 января 2026"
    if len(parts) == 3:
        day, month_str, year = parts
        month = RUSSIAN_MONTHS.get(month_str)
        if month:
            try:
                return datetime(int(year), month, int(day), tzinfo=timezone.utc)
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
        return True  # Если даты нет, считаем что подходит

    now = datetime.now(timezone.utc)
    threshold = now - timedelta(days=months * 30)  # Приблизительно 30 дней в месяце

    return date >= threshold


class UTMNNewsSource(UTMNBaseSource):
    """Источник: Новости ТюмГУ

    Парсит новостные статьи с сайта utmn.ru.
    """

    def __init__(self, base_url: str = "https://www.utmn.ru", delay: float = 1.0):
        super().__init__(base_url, delay)

    def _parse_news_article(self, soup: BeautifulSoup, url: str) -> Dict:
        """Парсит новостную статью

        Args:
            soup: BeautifulSoup объект
            url: URL страницы

        Returns:
            Словарь с данными статьи
        """
        # Заголовок
        title_elem = soup.find('h1')
        title = title_elem.get_text(strip=True) if title_elem else ''

        # Дата публикации
        # Согласно документации: class="date" с div class="month", "day", "year"
        date_elem = soup.find('div', class_='date')
        date_str = ''
        parsed_date = None
        if date_elem:
            # Формат: <div class="month">янв</div><div class="day">19</div><div class="year">2026</div>
            month = date_elem.find('div', class_='month')
            day = date_elem.find('div', class_='day')
            year = date_elem.find('div', class_='year')
            if month and day and year:
                date_str = f"{month.get_text(strip=True)} {day.get_text(strip=True)} {year.get_text(strip=True)}"
                parsed_date = parse_russian_date(date_str)

        # Текст статьи
        # Согласно документации: class="article-detail_text"
        content = soup.find('div', class_='article-detail_text') or soup.find('article')
        if content:
            text = content.get_text(separator=' ', strip=True)
        else:
            # Fallback: взять весь body
            body = soup.find('body')
            text = body.get_text(separator=' ', strip=True) if body else ''

        return {
            'title': title,
            'text': text,
            'date': date_str,
            'parsed_date': parsed_date,
            'url': url,
            'type': 'news'
        }

    def parse_page(self, url: str) -> Optional[Dict]:
        """Парсит одну страницу новости и возвращает словарь с данными

        Args:
            url: URL страницы для парсинга

        Returns:
            Словарь с данными страницы или None в случае ошибки
        """
        html = self._get_page_html(url)
        if not html:
            return None

        try:
            soup = BeautifulSoup(html, 'html.parser')
            return self._parse_news_article(soup, url)
        except Exception as e:
            logging.error(f"Failed to parse {url}: {e}")
            return None

    def _get_news_urls_from_page(self, soup: BeautifulSoup) -> List[Tuple[str, Optional[datetime]]]:
        """Получает URL новостей и их даты из одной страницы

        Args:
            soup: BeautifulSoup объект страницы

        Returns:
            Список кортежей (URL, parsed_date)
        """
        results = []

        news_items = soup.find_all('li', class_='news-page__el')

        for item in news_items:
            # Извлекаем дату из карточки
            date_elem = item.find('div', class_='date')
            parsed_date = None
            if date_elem:
                month = date_elem.find('div', class_='month')
                day = date_elem.find('div', class_='day')
                year = date_elem.find('div', class_='year')
                if month and day and year:
                    # День может быть внутри тега <a>, берем текст оттуда
                    day_text = day.get_text(strip=True)
                    date_str = f"{month.get_text(strip=True)} {day_text} {year.get_text(strip=True)}"
                    parsed_date = parse_russian_date(date_str)

            # Ищем все ссылки на статьи внутри элемента
            links = item.find_all('a', href=lambda x: x and '/news/stories/' in x)
            for link in links:
                href = link.get('href')
                if href and '/news/stories/' in href:
                    # Пропускаем параметры пагинации и фильтры
                    if '?' in href or 'PAGEN' in href:
                        continue

                    if href.startswith('/'):
                        full_url = f"{self.base_url}{href}"
                    else:
                        full_url = href

                    # Берем только URL с ID статьи (содержат цифры в пути)
                    url_parts = full_url.rstrip('/').split('/')
                    has_id = any(part.isdigit() for part in url_parts)

                    # Исключаем корневую категорию без ID
                    is_not_category = not full_url.rstrip('/').endswith('/news/stories')

                    if has_id and is_not_category:
                        results.append((full_url, parsed_date))
                        break  # Берем только первую ссылку (это ссылка на статью)

        return results

    def _get_all_news_urls(self, max_pages: int = 5, max_months: int = 2) -> List[str]:
        """Получает все URLы новостей с фильтрацией по дате (синхронный метод для совместимости)

        Args:
            max_pages: Максимальное количество страниц для парсинга (по умолчанию 5, None = без ограничений по датой)
            max_months: Максимальный возраст новостей в месяцах (по умолчанию 2)

        Returns:
            Список URLов новостей
        """
        all_urls = []
        page_num = 1
        consecutive_old_news = 0
        max_consecutive_old = 3

        while True:
            if page_num == 1:
                page_url = f"{self.base_url}/news/stories/"
            else:
                page_url = f"{self.base_url}/news/stories/?PAGEN_1={page_num}"

            if max_pages is not None and page_num > max_pages:
                logging.info(f"Reached max_pages limit ({max_pages}), stopping pagination")
                break

            logging.info(f"Fetching news page {page_num}...")
            html = self._get_page_html(page_url)

            if not html:
                logging.warning(f"Failed to get news page {page_num}")
                consecutive_old_news += 1
                if consecutive_old_news >= max_consecutive_old:
                    break
                page_num += 1
                continue

            soup = BeautifulSoup(html, 'html.parser')
            page_results = self._get_news_urls_from_page(soup)  # Теперь возвращает (URL, date)

            if len(page_results) == 0:
                consecutive_old_news += 1
                if consecutive_old_news >= max_consecutive_old:
                    break
                page_num += 1
                self._sleep()
                continue

            # Фильтруем по дате прямо из списка
            page_valid_count = 0
            for url, parsed_date in page_results:
                if parsed_date is None or is_within_months(parsed_date, max_months):
                    if url not in all_urls:
                        all_urls.append(url)
                        page_valid_count += 1

            if page_valid_count == 0:
                consecutive_old_news += 1
                logging.info(f"Page {page_num}: no valid news within {max_months} months (consecutive: {consecutive_old_news})")
                if consecutive_old_news >= max_consecutive_old:
                    break
            else:
                consecutive_old_news = 0
                logging.info(f"Page {page_num}: found {page_valid_count} valid news URLs (total: {len(all_urls)})")

            self._sleep()
            page_num += 1

        logging.info(f"Total news URLs fetched: {len(all_urls)}")
        return all_urls

    async def _get_all_news_urls_async(
        self,
        max_pages: int = 5,
        max_months: int = 2,
        session: aiohttp.ClientSession = None
    ) -> List[str]:
        """Получает все URLы новостей с async-фильтрацией по дате

        Args:
            max_pages: Максимальное количество страниц для парсинга
            max_months: Максимальный возраст новостей в месяцах
            session: aiohttp ClientSession

        Returns:
            Список URLов новостей
        """
        if session is None:
            raise ValueError("session parameter is required for async method")

        all_urls = []
        page_num = 1
        consecutive_old_news = 0
        max_consecutive_old = 3

        while True:
            if page_num == 1:
                page_url = f"{self.base_url}/news/stories/"
            else:
                page_url = f"{self.base_url}/news/stories/?PAGEN_1={page_num}"

            if max_pages is not None and page_num > max_pages:
                logging.info(f"Reached max_pages limit ({max_pages}), stopping pagination")
                break

            logging.info(f"Fetching news page {page_num}...")
            html = await self._get_page_html_async(page_url, session)

            if not html:
                logging.warning(f"Failed to get news page {page_num}")
                consecutive_old_news += 1
                if consecutive_old_news >= max_consecutive_old:
                    break
                page_num += 1
                continue

            soup = BeautifulSoup(html, 'html.parser')
            page_results = self._get_news_urls_from_page(soup)  # Возвращает (URL, date)

            if len(page_results) == 0:
                consecutive_old_news += 1
                if consecutive_old_news >= max_consecutive_old:
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
                consecutive_old_news += 1
                logging.info(f"Page {page_num}: no valid news within {max_months} months (consecutive: {consecutive_old_news})")
                if consecutive_old_news >= max_consecutive_old:
                    break
            else:
                consecutive_old_news = 0
                logging.info(f"Page {page_num}: found {page_valid_count} valid news URLs (total: {len(all_urls)})")

            page_num += 1

        logging.info(f"Total news URLs fetched: {len(all_urls)}")
        return all_urls

    def fetch_documents(self, max_pages: int = 5, max_months: int = 2) -> List[Document]:
        """Получает документы из источника новостей

        Args:
            max_pages: Максимальное количество страниц для парсинга (резерв)
            max_months: Максимальный возраст новостей в месяцах (по умолчанию 2)

        Returns:
            Список документов LangChain
        """
        all_documents = []

        logging.info(f"Fetching news pages (max {max_months} months old)...")
        news_urls = self._get_all_news_urls(max_pages=max_pages, max_months=max_months)

        for i, url in enumerate(news_urls, 1):
            logging.info(f"Fetching news {i}/{len(news_urls)}: {url}")
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
                        'source_type': 'news',
                        'url': data['url'],
                        'title': data['title'],
                        'type': data['type'],
                        'date': data['date']
                    }
                ))
            self._sleep()

        logging.info(f"Total news documents fetched: {len(all_documents)}")
        return all_documents

    def fetch_latest(self, days: int = 7) -> List[Document]:
        """Получает только свежие новости за указанное количество дней

        Args:
            days: Количество дней (по умолчанию 7)

        Returns:
            Список документов LangChain
        """
        all_documents = []

        # Используем 1 месяц как ceiling для фильтрации по дням
        max_months = 1

        logging.info(f"Fetching latest news (last {days} days)...")
        news_urls = self._get_all_news_urls(max_pages=1, max_months=max_months)

        now = datetime.now(timezone.utc)
        threshold = now - timedelta(days=days)

        for i, url in enumerate(news_urls, 1):
            logging.info(f"Fetching news {i}/{len(news_urls)}: {url}")
            data = self.parse_page(url)
            if data and data['text']:
                parsed_date = data.get('parsed_date')
                if parsed_date:
                    if parsed_date >= threshold:
                        all_documents.append(Document(
                            page_content=data['text'],
                            metadata={
                                'source_type': 'news',
                                'url': data['url'],
                                'title': data['title'],
                                'type': data['type'],
                                'date': data['date']
                            }
                        ))
                    else:
                        logging.info(f"Skipping {url} - too old ({data['date']})")
                        # Сортируем по URL (предполагаем что новые идут первыми)
                        # Если нашли старую, продолжаем (могут быть новее в списке)
                else:
                    # Если дату не удалось распарсить, включаем новость
                    all_documents.append(Document(
                        page_content=data['text'],
                        metadata={
                            'source_type': 'news',
                            'url': data['url'],
                            'title': data['title'],
                            'type': data['type'],
                            'date': data['date']
                        }
                    ))
            else:
                self._sleep()

        logging.info(f"Total latest news documents fetched: {len(all_documents)}")
        return all_documents

    # ========== Async методы для параллельного парсинга ==========

    async def parse_page_async(
        self,
        url: str,
        session: aiohttp.ClientSession
    ) -> Optional[Dict]:
        """Парсит одну страницу новости (async)

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
            return self._parse_news_article(soup, url)
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
            max_months: Максимальный возраст новостей в месяцах (по умолчанию 2)
            max_concurrent: Максимальное количество одновременных запросов (по умолчанию 10)

        Returns:
            Список документов LangChain
        """
        # Создаем session для async запросов
        async with aiohttp.ClientSession(headers=self.headers) as session:
            # Получаем URLы с async-фильтрацией по дате
            news_urls = await self._get_all_news_urls_async(
                max_pages=max_pages,
                max_months=max_months,
                session=session
            )

            if not news_urls:
                logging.warning("No news URLs found")
                return []

            logging.info(f"Fetching {len(news_urls)} news pages with {max_concurrent} concurrent requests...")

            all_documents = []
            semaphore = asyncio.Semaphore(max_concurrent)

            async def fetch_with_semaphore(url: str, index: int) -> Optional[Document]:
                """Парсит страницу с ограничением семафора"""
                async with semaphore:
                    logging.info(f"Fetching news {index + 1}/{len(news_urls)}: {url}")
                    data = await self.parse_page_async(url, session)

                    if data and data['text']:
                        return Document(
                            page_content=data['text'],
                            metadata={
                                'source_type': 'news',
                                'url': data['url'],
                                'title': data['title'],
                                'type': data['type'],
                                'date': data['date']
                            }
                        )
                    return None

            tasks = [
                fetch_with_semaphore(url, i)
                for i, url in enumerate(news_urls)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Task failed with exception: {result}")
                elif result is not None:
                    all_documents.append(result)

        logging.info(f"Total news documents fetched: {len(all_documents)}")
        return all_documents

