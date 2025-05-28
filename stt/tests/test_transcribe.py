import pytest
from fastapi import UploadFile
import os
import asyncio
from unittest.mock import Mock, patch
import aiohttp
from aiohttp import FormData
import uvicorn
import threading
import time
from app import app

def start_server():
    """Запускает FastAPI сервер в отдельном потоке"""
    uvicorn.run(app, host="127.0.0.1", port=8000)

@pytest.fixture(scope="session", autouse=True)
def server():
    """Фикстура для запуска и остановки тестового сервера"""
    server_thread = threading.Thread(target=start_server)
    server_thread.daemon = True
    server_thread.start()
    time.sleep(1)
    yield

def test_transcribe_endpoint(client, test_audio_file):
    """Тест эндпоинта транскрибации с тестовым аудиофайлом"""
    with open(test_audio_file, 'rb') as audio_file:
        files = {'file': ('test.wav', audio_file, 'audio/wav')}
        response = client.post('/transcribe/', files=files)

        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert data['status'] == 'success'
        assert 'transcription' in data
        assert isinstance(data['transcription'], str)

@pytest.mark.asyncio
async def test_transcribe_endpoint_async(test_audio_file):
    """Асинхронный тест эндпоинта транскрибации"""
    async with aiohttp.ClientSession() as session:
        form = FormData()
        with open(test_audio_file, 'rb') as f:
            file_content = f.read()
        form.add_field('file', file_content, filename='test.wav', content_type='audio/wav')

        async with session.post('http://127.0.0.1:8000/transcribe/', data=form) as response:
            assert response.status == 200
            result = await response.json()
            assert result['status'] == 'success'
            assert 'transcription' in result
            assert isinstance(result['transcription'], str)

def test_transcribe_invalid_file(client):
    """Тест обработки невалидного файла"""
    files = {'file': ('test.txt', b'invalid content', 'text/plain')}
    response = client.post('/transcribe/', files=files)
    assert response.status_code == 500

def test_root_endpoint(client):
    """Тест корневого эндпоинта"""
    response = client.get('/')
    assert response.status_code == 200
    data = response.json()
    assert data == {"message": "hello, world!"}
