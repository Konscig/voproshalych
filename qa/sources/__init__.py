"""
Пакет с источниками данных для парсинга официального сайта Тюменского государственного университета (utmn.ru)

Содержит классы для парсинга следующих источников (все на основе requests):
- Контакты и реквизиты
- Структура университета (ректорат, проректоры)
- Новости (с пагинацией)
- События и мероприятия
"""

from .utmn_base_source import UTMNBaseSource
from .utmn_contacts_source import UTMNContactsSource
from .utmn_structure_source import UTMNStructureSource
from .utmn_news_source import UTMNNewsSource
from .utmn_events_source import UTMNEventsSource
from .utmn_combined_source import UTMNCombinedSource

__all__ = [
    'UTMNBaseSource',
    'UTMNContactsSource',
    'UTMNStructureSource',
    'UTMNNewsSource',
    'UTMNEventsSource',
    'UTMNCombinedSource'
]
