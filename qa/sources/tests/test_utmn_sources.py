"""
Тесты для всех модулей на основе requests+BeautifulSoup
"""

import pytest

from sources.utmn_contacts_source import UTMNContactsSource
from sources.utmn_structure_source import UTMNStructureSource
from sources.utmn_employees_source import UTMNEmployeesSource
from sources.utmn_news_source import UTMNNewsSource
from sources.utmn_events_source import UTMNEventsSource
from sources.utmn_rss_source import UTMNRssSource
from sources.utmn_combined_source import UTMNCombinedSource


class TestUTMNEmployeesSource:
    """Тесты модуля сотрудников"""

    @pytest.fixture
    def source(self):
        return UTMNEmployeesSource(delay=0)

    def test_init(self, source):
        assert source.base_url == "https://www.utmn.ru"

    def test_get_employees_urls(self, source):
        urls = source._get_all_employees_urls(max_pages=1)
        assert len(urls) > 0

    def test_parse_employee_page(self, source):
        test_url = "https://www.utmn.ru/o-tyumgu/sotrudniki/m.v.avriskin/"
        data = source.parse_page(test_url)
        assert data is not None
        assert 'title' in data

    def test_fetch_documents(self, source):
        docs = source.fetch_documents(max_pages=1)
        assert len(docs) > 0


class TestUTMNNewsSource:
    """Тесты модуля новостей"""

    @pytest.fixture
    def source(self):
        return UTMNNewsSource(delay=0)

    def test_init(self, source):
        assert source.base_url == "https://www.utmn.ru"

    def test_get_news_urls(self, source):
        urls = source._get_all_news_urls()
        assert len(urls) > 0

    def test_parse_news_article(self, source):
        test_url = "https://www.utmn.ru/news/stories/priyem/1342978/"
        data = source.parse_page(test_url)
        assert data is not None
        assert 'title' in data

    def test_fetch_documents(self, source):
        docs = source.fetch_documents()
        assert len(docs) > 0


class TestUTMNContactsSource:
    """Тесты модуля контактов"""

    @pytest.fixture
    def source(self):
        return UTMNContactsSource(delay=0)

    def test_init(self, source):
        assert source.base_url == "https://www.utmn.ru"

    def test_parse_contacts_page(self, source):
        data = source.parse_contacts_page()
        assert data is not None
        assert 'title' in data

    def test_fetch_documents(self, source):
        docs = source.fetch_documents()
        assert len(docs) > 0


class TestUTMNStructureSource:
    """Тесты модуля структуры"""

    @pytest.fixture
    def source(self):
        return UTMNStructureSource(delay=0)

    def test_init(self, source):
        assert source.base_url == "https://www.utmn.ru"

    def test_parse_structure_page(self, source):
        data = source.parse_structure_page()
        assert data is not None
        assert 'title' in data

    def test_fetch_documents(self, source):
        docs = source.fetch_documents()
        assert len(docs) > 0


class TestUTMNEventsSource:
    """Тесты модуля событий"""

    @pytest.fixture
    def source(self):
        return UTMNEventsSource(delay=0)

    def test_init(self, source):
        assert source.base_url == "https://www.utmn.ru"

    def test_get_events_urls(self, source):
        urls = source._get_all_events_urls()
        assert len(urls) > 0

    def test_fetch_documents(self, source):
        docs = source.fetch_documents()
        assert len(docs) > 0


class TestUTMNRssSource:
    """Тесты RSS модуля"""

    @pytest.fixture
    def source(self):
        return UTMNRssSource(delay=0)

    def test_init(self, source):
        assert source.base_url == "https://www.utmn.ru"

    def test_parse_rss_feed(self, source):
        items = source.parse_rss_feed()
        assert len(items) > 0

    def test_fetch_documents(self, source):
        docs = source.fetch_documents()
        assert len(docs) > 0


class TestUTMNCombinedSource:
    """Тесты комбинированного источника"""

    @pytest.fixture
    def source(self):
        src = UTMNCombinedSource(delay=0, use_confluence=False)
        yield src
        src.close()

    def test_init(self, source):
        assert source.base_url == "https://www.utmn.ru"
        assert source.contacts_source is not None
        assert source.structure_source is not None

    def test_fetch_all_documents(self, source):
        docs = source.fetch_all_documents(max_pages=1)
        assert len(docs) > 0
        # Проверяем, что есть документы из разных источников
        source_types = set(d.metadata['source_type'] for d in docs)
        assert len(source_types) > 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
