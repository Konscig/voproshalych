import pytest
from fastapi.testclient import TestClient
from app import app
import os
from pydub import AudioSegment
import tempfile
from gtts import gTTS
import aiohttp
from aiohttp import FormData
import asyncio
from pathlib import Path
import numpy as np

TEST_TEMP_DIR = Path(tempfile.gettempdir()) / "stt_test_files"
TEST_TEMP_DIR.mkdir(exist_ok=True)

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def test_audio_file():
    """Создает тестовый аудиофайл с русской речью"""
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
        tts = gTTS(text='Привет, это тестовое сообщение для проверки распознавания речи.', lang='ru')
        tts.save(temp_file.name)

        audio = AudioSegment.from_mp3(temp_file.name)
        wav_path = temp_file.name.replace('.mp3', '.wav')
        audio.export(wav_path, format='wav')

        os.unlink(temp_file.name)

        yield wav_path

    os.unlink(wav_path)

@pytest.fixture
def mock_tg_voice_file():
    """Создает мок файла голосового сообщения Telegram"""
    with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
        tts = gTTS(text='Тестовое сообщение', lang='ru')
        tts.save(temp_file.name)
        yield temp_file.name
    os.unlink(temp_file.name)

@pytest.fixture
def mock_vk_voice_url():
    """Создает мок URL голосового сообщения VK"""
    return "https://vk.com/voice_message.mp3"

@pytest.fixture
async def mock_vk_voice_content():
    """Создает мок содержимого голосового сообщения VK"""
    with tempfile.NamedTemporaryFile(suffix='.ogg', delete=False) as temp_file:
        tts = gTTS(text='Тестовое сообщение', lang='ru')
        tts.save(temp_file.name)
        with open(temp_file.name, 'rb') as f:
            content = f.read()
        os.unlink(temp_file.name)
        return content

@pytest.fixture
def test_user_id():
    """Тестовый ID пользователя"""
    return 12345
