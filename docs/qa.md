# Микросервис QA

## Описание
Микросервис, предоставляющий API для:
 * генерации ответа на вопрос, опираясь на документы из вики-системы;
 * обновления векторного индекса текстов документов из вики-системы;
 * обработки голосовых сообщений и их транскрибации.

> [!IMPORTANT]
> Информация о настройке взаимодействия с вики-системой Confluence представлена в [confluence-integration.md](confluence-integration.md).

## [main](../qa/main.py)

### `get_answer(context: str, question: str) -> str`
Возвращает сгенерированный LLM ответ на вопрос пользователя по заданному документу в соответствии с промтом

    Args:
        context (str): текст документа (или фрагмента документа)
        question (str): вопрос пользователя

    Returns:
        str: ответ на вопрос

### `qa(request: web.Request) -> web.Response`
Возвращает ответ на вопрос пользователя и ссылку на источник. Поддерживает как текстовые, так и голосовые вопросы.

    Args:
        request (web.Request): запрос, содержащий `question` или аудиофайл

    Returns:
        web.Response: ответ с текстом ответа и ссылкой на источник

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

### `class QuestionAnswer(Base)`
Вопрос пользователя с ответом на него

    Args:
        id (int): id ответа
        question (str): вопрос пользователя
        embedding (Vector): векторное представление текста вопроса размерностью 1024
        answer (str | None): ответ на вопрос пользователя
        confluence_url (str | None): ссылка на страницу в вики-системе, содержащую ответ
        score (int | None): оценка пользователем ответа
        user_id (int): id пользователя, задавшего вопрос
        user (User): пользователь, задавший вопрос
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

## [conftest](../qa/conftest.py)
### `load_env()`
Фикстура для загрузки переменных окружения из корневого .env файла. Автоматически применяется ко всем тестам.

### `client()`
Фикстура для создания тестового клиента aiohttp. Создает сессию для выполнения HTTP-запросов в тестах.

### `test_audio_file()`
Фикстура для тестового аудиофайла. Создает временный WAV файл для тестирования и удаляет его после завершения теста.

## [tests](../qa/tests.py)
### `test_llm()`
тест взаимодействия с LLM

### `test_confluence()`
тест взаимодействия с Confluence

## [test_audio](../qa/test_audio.py)
### `test_transcribe_audio()`
Тест функции транскрибации аудиофайла с использованием модели STT.
