import os
import time
import requests
import ffmpeg
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
import asyncio
import httpx


load_dotenv()
TOKEN = os.getenv("TG_ACCESS_TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher()

FASTAPI_URL = "http://stt:8000/transcribe/"

# скачивает файл с сервера telegram по переданному file_path ,
# конвертирует его из формата .oga в .wav и сохраняет
async def download_and_convert(file_path: str, user_id: str):

    file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"

    timestamp = int(time.time())
    wav_path = f"voice_{user_id}_{timestamp}.wav"
    ogg_path = f"voice_{user_id}_{timestamp}.oga"

    response = requests.get(file_url)
    with open(ogg_path, "wb") as f:
        f.write(response.content)

    try:
        ffmpeg.input(ogg_path).output(wav_path, format="wav").run(overwrite_output=True)
    except Exception as e:
        print(f"Ошибка при конвертации: {e}")

    os.remove(ogg_path)
    print(f"Файл сохранён: {wav_path}")
    return wav_path

# отправляет конвертированный аудиофайл на сервер для обработки ;
async def send_to_server(wav_path: str) -> str:
    async with httpx.AsyncClient() as client:
        with open(wav_path, "rb") as audio_file:
            response = await client.post(
                FASTAPI_URL,
                files={"file": (wav_path, audio_file, "audio/wav")}
            )

    if response.status_code == 200:
        return response.json().get("transcription", "Ошибка при обработке")
    else:
        return f"Ошибка {response.status_code}: {response.text}"

# обработчик голосовых сообщений :
async def handle_voice(message: Message):

    file_info = await bot.get_file(message.voice.file_id)
    file_path = file_info.file_path

    wav_path = await download_and_convert(file_path, message.from_user.id)
    transcription = await send_to_server(wav_path)
    await message.reply(f"Распознанный текст: {transcription}")

# регистрация функции handle_voice как обработчика для голосовых сообщений ;
dp.message.register(handle_voice, lambda message: hasattr(message, "voice") and message.voice is not None)

if __name__ == '__main__':
    print('telegram-bot:: started in pull mode')
    asyncio.run(dp.start_polling(bot))
