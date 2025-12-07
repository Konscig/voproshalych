import pytest
import pytest_asyncio
import os
import aiohttp
from pathlib import Path
from dotenv import load_dotenv

@pytest.fixture(autouse=True)
def load_env():
    """Фикстура для загрузки переменных окружения из корневого .env файла"""
    root_env = Path(__file__).parent.parent / '.env'
    if root_env.exists():
        load_dotenv(root_env)

@pytest_asyncio.fixture
async def client():
    """Фикстура для создания тестового клиента"""
    async with aiohttp.ClientSession() as session:
        yield session

@pytest.fixture
def test_audio_file():
    """Фикстура для тестового аудиофайла"""
    test_file = "test_audio.wav"
    with open(test_file, "wb") as f:
        f.write(b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00D\xac\x00\x00\x88X\x01\x00\x02\x00\x10\x00data\x00\x00\x00\x00")
    yield test_file
    if os.path.exists(test_file):
        os.remove(test_file)
