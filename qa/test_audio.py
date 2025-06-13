import os
import pytest
import pytest_asyncio
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
import logging
from unittest.mock import patch

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

pytestmark = pytest.mark.asyncio

@pytest_asyncio.fixture
async def app():
    """Создает тестовое приложение"""
    app = web.Application()

    async def handle_qa(request):
        """Обработчик для тестового эндпоинта /qa/"""
        try:
            data = await request.post()
            if 'file' not in data:
                return web.json_response({'error': 'No file provided'}, status=400)
            file = data['file']
            if not file.filename.endswith('.wav'):
                return web.json_response({'error': 'Invalid file type'}, status=400)
            with patch('am.transcribe_audio', return_value="Тестовый текст транскрибации"):
                return web.json_response({
                    'answer': 'Test answer',
                    'confluence_url': 'https://example.com/test'
                })
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return web.json_response({'error': str(e)}, status=500)

    app.router.add_post('/qa/', handle_qa)
    return app

@pytest_asyncio.fixture
async def client(app):
    """Создает тестовый клиент"""
    async with TestClient(TestServer(app)) as client:
        yield client

@pytest.fixture
def test_audio_file():
    """Фикстура для тестового аудиофайла"""
    logger.debug("Creating test audio file...")
    test_file = "test_audio.wav"
    with open(test_file, "wb") as f:
        f.write(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    logger.debug(f"Test audio file created: {test_file}")
    yield test_file
    if os.path.exists(test_file):
        logger.debug(f"Removing test audio file: {test_file}")
        os.remove(test_file)

@pytest.mark.asyncio
async def test_qa_audio_endpoint(client, test_audio_file):
    """Тест эндпоинта /qa/ с аудиофайлом"""
    logger.debug("Starting test_qa_audio_endpoint...")
    form = FormData()
    with open(test_audio_file, 'rb') as f:
        file_content = f.read()
    form.add_field('file', file_content, filename='test.wav', content_type='audio/wav')
    logger.debug("Sending POST request to /qa/ endpoint...")
    async with client.post('/qa/?audio=1', data=form) as response:
        logger.debug(f"Received response with status: {response.status}")
        assert response.status == 200
        data = await response.json()
        logger.debug(f"Response data: {data}")
        assert 'answer' in data
        assert 'confluence_url' in data
        assert isinstance(data['answer'], str)
        assert isinstance(data['confluence_url'], str)

@pytest.mark.asyncio
async def test_qa_audio_endpoint_async(client, test_audio_file):
    """Тест эндпоинта /qa/ с аудиофайлом через FormData"""
    logger.debug("Starting test_qa_audio_endpoint_async...")
    form = FormData()
    with open(test_audio_file, 'rb') as f:
        file_content = f.read()
    form.add_field('file', file_content, filename='test.wav', content_type='audio/wav')
    logger.debug("Sending POST request to /qa/ endpoint...")
    async with client.post('/qa/?audio=1', data=form) as response:
        logger.debug(f"Received response with status: {response.status}")
        assert response.status == 200
        result = await response.json()
        logger.debug(f"Response data: {result}")
        assert 'answer' in result
        assert 'confluence_url' in result
        assert isinstance(result['answer'], str)
        assert isinstance(result['confluence_url'], str)

@pytest.mark.asyncio
async def test_qa_audio_invalid_file(client):
    """Тест обработки невалидного аудиофайла"""
    logger.debug("Starting test_qa_audio_invalid_file...")
    form = FormData()
    form.add_field('file', b'invalid content', filename='test.txt', content_type='text/plain')
    logger.debug("Sending POST request with invalid file...")
    async with client.post('/qa/?audio=1', data=form) as response:
        logger.debug(f"Received response with status: {response.status}")
        assert response.status == 400

def test_transcribe_audio_function(test_audio_file):
    """Тест функции транскрибации аудио"""
    logger.debug("Starting test_transcribe_audio_function...")
    with patch('am.transcribe_audio', return_value="Тестовый текст транскрибации") as mock:
        result = mock(test_audio_file)
        logger.debug(f"Transcription result: {result}")
        assert isinstance(result, str)
