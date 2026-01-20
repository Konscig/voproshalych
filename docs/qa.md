# Микросервис QA

## Описание
Микросервис, предоставляющий API для:
 * генерации ответа на вопрос, опираясь на документы из вики-системы;
 * обновления векторного индекса текстов документов из вики-системы.

 > [!IMPORTANT]
> Информация о настройке взаимодействия с вики-системой Confluence представлена в [confluence-integration.md](confluence-integration.md).

## [main](../qa/main.py)

### `get_answer(context: str, question: str) -> str`
Возвращает сгенерированный LLM ответ на вопрос пользователя по заданному документу в соответствии с промтом

    Args:
        context (str): текст документа (или фрагмента документа)
        question (str): вопрос пользователя

    Returns:
        str: экземпляр класса Chunk — фрагмент документа

### `qa(request: web.Request) -> web.Response`
Возвращает ответ на вопрос пользователя и ссылку на источник

    Args:
        request (web.Request): запрос, содержащий `question`

    Returns:
        web.Response: ответ

### `generate_greeting(request: web.Request) -> web.Response`
Генерирует поздравления с использованием LLM

    Args:
        request (web.Request): запрос, содержащий `template`, `user_name`, `holiday_name`

    Returns:
        web.Response: ответ с поздравлением

### `reindex(request: web.Request) -> web.Response`
Пересоздаёт векторный индекс текстов для ответов на вопросы

    Args:
        request (web.Request): запрос

    Returns:
        web.Response: ответ

## [config](../qa/config.py)

### `class Config`
Класс с переменными окружения

#### `get_mistral_headers() -> dict`
Возвращает заголовки для запросов к Mistral API

#### `get_default_prompt(context: str, question: str)`
Создаёт payload с промптом для Mistral API

## [database](../qa/database.py)

### `class Chunk(Base)`
Фрагмент документа из вики-системы

    Args:
        confluence_url (str): ссылка на источник
        text (str): текст фрагмента
        embedding (Vector): векторное представление текста фрагмента размерностью 1024
        created_at (datetime): время создания модели
        updated_at (datetime): время обновления модели

## [confluence_retrieving](../qa/confluence_retrieving.py)

### `get_document_content_by_id(confluence: Confluence, page_id: str) -> tuple[str | None, str | None]`
Возвращает содержимое страницы на Confluence после предобработки с помощью PyPDF или BS4 и ссылку на страницу

    Args:
        confluence (Confluence): экземпляр Confluence
        page_id (str): ID страницы

    Returns:
        tuple[str | None, str | None]: содержимое страницы, ссылка на страницу

### `reindex_confluence(engine: Engine, text_splitter: TextSplitter, encoder_model: SentenceTransformer)`
Пересоздаёт векторный индекс текстов для ответов на вопросы. При этом обрабатываются страницы, не имеющие вложенных страниц.

    Args:
        engine (Engine): экземпляр подключения к БД
        text_splitter (TextSplitter): разделитель текста на фрагменты
        encoder_model (SentenceTransformer): модель получения векторных представлений Sentence Transformer

### `get_chunk(engine: Engine, encoder_model: SentenceTransformer, question: str) -> Chunk | None`
Возвращает ближайший к вопросу фрагмент документа Chunk из векторной базы данных

    Args:
        engine (Engine): экземпляр подключения к БД
        encoder_model (SentenceTransformer): модель получения векторных представлений SentenceTransformer
        question (str): вопрос пользователя

    Returns:
        Chunk | None: экземпляр класса Chunk — фрагмент документа

## [tests](../qa/tests.py)

### `test_llm()`
тест взаимодействия с LLM

### `test_confluence()`
тест взаимодействия с Confluence

## [utmn_retrieving](../qa/utmn_retrieving.py)

### `reindex_utmn(engine, text_splitter, encoder_model, max_pages=5)`
Пересоздаёт векторный индекс текстов из UTMN источников (синхронная версия)

    Args:
        engine (Engine): экземпляр подключения к БД
        text_splitter (RecursiveCharacterTextSplitter): разделитель текста на фрагменты
        encoder_model (SentenceTransformer): модель получения векторных представлений
        max_pages (int): максимальное количество страниц для источников с пагинацией

### `reindex_utmn_async(engine, text_splitter, encoder_model, max_pages=5, max_concurrent_news_events=10)`
Пересоздаёт векторный индекс текстов из UTMN источников (async версия с батчевым кодированием)

    Args:
        engine (Engine): экземпляр подключения к БД
        text_splitter (RecursiveCharacterTextSplitter): разделитель текста на фрагменты
        encoder_model (SentenceTransformer): модель получения векторных представлений
        max_pages (int): максимальное количество страниц для источников с пагинацией
        max_concurrent_news_events (int): макс. одновременных запросов для новостей/событий

    **Оптимизации:**
    - Батчевое кодирование (batch_size=32) для 3-5x ускорения
    - tqdm прогресс-бар для видимости процесса
    - Промежуточные коммиты каждые 500 чанков

### `reindex_utmn_async_wrapper(engine, text_splitter, encoder_model, max_pages=5, max_concurrent_news_events=10)`
Wrapper для вызова async функции из sync кода

## UTMN Источники

Модуль `qa/sources/` содержит парсеры официального сайта Тюменского государственного университета (utmn.ru).

### Доступные источники

| Источник | Файл | Описание |
|----------|------|----------|
| Контакты | `utmn_contacts_source.py` | Контакты и реквизиты |
| Структура | `utmn_structure_source.py` | Ректорат, проректоры |
| Сотрудники | `utmn_employees_source.py` | Телефонный справочник (опционально) |
| Новости | `utmn_news_source.py` | Новости с пагинацией |
| События | `utmn_events_source.py` | События и мероприятия |
| RSS | `utmn_rss_source.py` | RSS-лента |
| Комбо | `utmn_combined_source.py` | Агрегация всех источников |

### UTMNCombinedSource

Комбинированный источник для агрегации данных из всех UTMN источников.

```python
from sources.utmn_combined_source import UTMNCombinedSource

# Инициализация
source = UTMNCombinedSource(
    base_url="https://www.utmn.ru",
    delay=0.3,
    use_confluence=False
)

# Синхронный парсинг
documents = source.fetch_all_documents(max_pages=5)

# Async парсинг (рекомендуется)
documents = await source.fetch_all_documents_async(
    max_pages=5,
    max_concurrent_news_events=10
)

# Закрытие
source.close()
```

### Производительность

| Метрика | До | После | Ускорение |
|---------|----|----|----|----|
| Парсинг | ~4-5 мин | <30 сек | 10x |
| Дата-фильтрация | ~140 запросов | 0 | ∞ |
| Кодирование | ~1.8 ч | ~20-30 мин | 3-5x |

### Результаты парсинга

**Ветка feat/utmn-parsers-with-employees** (полное решение):
- Сотрудники: 2699 документов
- Новости: 135 документов
- События: 26 документов
- **Итого: ~2860 документов**

**Ветка feat/utmn-parsers-clean** (без сотрудников):
- Контакты: ~5-10 документов
- Структура: ~20-30 документов
- Новости: 135 документов
- События: 26 документов
- RSS: ~10-20 документов
- **Итого: ~200-250 документов**

Подробнее см. `docs/03_utmn_parsers_solution.md` в персональном репозитории.
