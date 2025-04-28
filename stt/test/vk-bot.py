import os
import time
import aiofiles
import httpx
import ffmpeg
from dotenv import load_dotenv
from vkbottle.bot import Bot, Message

load_dotenv()
GROUP_TOKEN = os.getenv("VK_ACCESS_GROUP_TOKEN")
FASTAPI_URL = "http://stt:8000/transcribe/"

bot = Bot(token=GROUP_TOKEN)

# скачивает .ogg по прямой ссылке, конвертирует в .wav и возвращает путь к wav-файлу
async def download_and_convert(url: str, user_id: int) -> str:
    timestamp = int(time.time())
    ogg_path = f"voice_{user_id}_{timestamp}.ogg"
    wav_path = f"voice_{user_id}_{timestamp}.wav"

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        async with aiofiles.open(ogg_path, "wb") as f:
            await f.write(response.content)

    try:
        ffmpeg.input(ogg_path).output(wav_path, format="wav").run(overwrite_output=True)
    except Exception as e:
        print(f"Ошибка при конвертации: {e}")
    finally:
        os.remove(ogg_path)

    print(f"Файл сохранён: {wav_path}")
    return wav_path

# отправляет wav-файл на внешний FastAPI‑сервер и возвращает транскрипцию
async def send_to_server(wav_path: str) -> str:
    async with httpx.AsyncClient() as client:
        async with aiofiles.open(wav_path, "rb") as audio_file:
            audio_data = await audio_file.read()
            files = {"file": (wav_path, audio_data, "audio/wav")}
            response = await client.post(FASTAPI_URL, files=files)

    if response.status_code == 200:
        return response.json().get("transcription", "Ошибка при обработке")
    else:
        return f"Ошибка {response.status_code}: {response.text}"

# обработчик голосовых сообщений
@bot.on.message()
async def handle_voice(message: Message):
    if message.attachments:
        for attachment in message.attachments:
            if attachment.type == "audio_message":
                audio = attachment.audio_message
                ogg_url = audio.link_ogg
                wav_path = await download_and_convert(ogg_url, message.from_id)
                transcription = await send_to_server(wav_path)
                await message.answer(f"Распознанный текст: {transcription}")
                os.remove(wav_path)
                break

if __name__ == "__main__":
    print('vk-bot:: started in pull mode')
    bot.run_forever()
