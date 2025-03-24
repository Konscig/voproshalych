import os                      
import requests                
import ffmpeg                  
from aiogram import Bot, Dispatcher, types  
from aiogram.types import Message            
from dotenv import load_dotenv               
import asyncio   
import httpx              

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=TOKEN)
dp = Dispatcher()

FASTAPI_URL = "http://127.0.0.1:8000/transcribe/"

# скачивает файл с сервера telegram по переданному file_path , 
# конвертирует его из формата .oga в .wav и сохраняет как example.wav

async def download_and_convert(file_path: str):
 
    file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
    
    ogg_path = "example.oga"
    wav_path = "example.wav"

    response = requests.get(file_url) # скачивает файл по сформированному URL
    with open(ogg_path, "wb") as f:   # сохраняет скачанный файл в формате .oga
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
        with open(wav_path, "rb") as audio_file:  # открываем файл в бинарном режиме
            response = await client.post(
                FASTAPI_URL,
                files={"file": (wav_path, audio_file, "audio/wav")}  # отправляем файл
            )

    if response.status_code == 200:
        return response.json().get("transcription", "Ошибка при обработке")
    else:
        return f"Ошибка {response.status_code}: {response.text}"
    
# обработчик голосовых сообщений :
# получает file_id голосового сообщения ;
# запрашивает у Telegram API информацию о файле (включая file_path) ;
# вызывает функцию для скачивания и конвертации файла ;
# вызыввает функцию отпраквки аудиофайла на сервер для обработки ;

async def handle_voice(message: Message):

    file_info = await bot.get_file(message.voice.file_id)
    file_path = file_info.file_path
    
    wav_path = await download_and_convert(file_path)
    transcription = await send_to_server(wav_path)
    await message.reply(f"Распознанный текст: {transcription}")

# регистрация функции handle_voice как обработчика для голосовых сообщений ;
# бот запоминает, что при получении сообщения, 
# удовлетворяющего условию (наличие voice), нужно вызвать handle_voice ;

dp.message.register(handle_voice, lambda message: hasattr(message, "voice") and message.voice is not None)

# запускать код ниже только если этот файл выполняется напрямую, а не импортируется как модуль ;
if __name__ == '__main__':
    asyncio.run(dp.start_polling(bot))
