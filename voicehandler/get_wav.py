import os                      # работа с файловой системой и переменными окружения
import requests                # для выполнения HTTP-запросов (скачивания файла по URL)
import ffmpeg                  # для конвертации аудиофайлов с помощью ffmpeg
from aiogram import Bot, Dispatcher, types  # импорт основных классов aiogram
from aiogram.types import Message            # тип сообщения Telegram
from dotenv import load_dotenv               # загрузка переменных из файла .env
import asyncio                 # для работы с асинхронным циклом

# загружаем переменные окружения из файла .env
load_dotenv()

# извлекаем токен бота из переменной окружения
TOKEN = os.getenv("BOT_TOKEN")

# создаем объект бота с указанным токеном
bot = Bot(token=TOKEN)

# создаем диспетчер без передачи объекта бота (в aiogram 3 это так)
dp = Dispatcher()

async def download_and_convert(file_path: str):
    """
    функция скачивает файл с сервера Telegram по переданному file_path,
    конвертирует его из формата .oga в .wav и сохраняет как example.wav.
    """
    # формируем URL для скачивания файла, подставляя токен и относительный путь к файлу
    file_url = f"https://api.telegram.org/file/bot{TOKEN}/{file_path}"
    
    # определяем имена файлов: временный (.oga) и конечный (.wav)
    ogg_path = "example.oga"
    wav_path = "example.wav"

    # отправляем GET-запрос по URL и сохраняем ответ (содержимое файла)
    response = requests.get(file_url)
    # записываем содержимое файла в локальный файл в бинарном режиме
    with open(ogg_path, "wb") as f:
        f.write(response.content)

    # пытаемся конвертировать скачанный файл из .oga в .wav с помощью ffmpeg
    try:
        ffmpeg.input(ogg_path).output(wav_path, format="wav").run(overwrite_output=True)
    except Exception as e:
        print(f"Ошибка при конвертации: {e}")

    # удаляем временный файл .oga, так как он больше не нужен
    os.remove(ogg_path)
    print(f"Файл сохранён: {wav_path}")

async def handle_voice(message: Message):
    """
    обработчик голосовых сообщений:
    1. Получает file_id голосового сообщения.
    2. Запрашивает у Telegram API информацию о файле (включая file_path).
    3. Вызывает функцию для скачивания и конвертации файла.
    """
    # получаем информацию о файле по его file_id через Telegram API
    file_info = await bot.get_file(message.voice.file_id)
    # извлекаем относительный путь к файлу (file_path)
    file_path = file_info.file_path
    # вызываем функцию для скачивания и конвертации файла
    await download_and_convert(file_path)

# регистрируем обработчик для сообщений, содержащих голос.
# здесь мы вызываем dp.message.register, передавая:
# - функцию-обработчик handle_voice
# - фильтр в виде lambda-функции, которая возвращает True, если в объекте сообщения есть атрибут voice и он не равен None.
dp.message.register(handle_voice, lambda message: hasattr(message, "voice") and message.voice is not None)

# запускаем polling (опрос обновлений) с помощью asyncio.run(), передавая объект бота
if __name__ == '__main__':
    asyncio.run(dp.start_polling(bot))
