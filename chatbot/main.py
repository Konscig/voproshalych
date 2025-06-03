import asyncio
import json
import logging
import math
import os
from multiprocessing import Process
import tempfile
from pathlib import Path

import ffmpeg
import aiofiles
import aiohttp
import aiogram as tg
from aiogram import filters, F
from aiogram.exceptions import TelegramUnauthorizedError as TUerror
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiohttp import web, ClientError, ClientSession
from aiohttp import FormData
from sqlalchemy import create_engine
import vkbottle as vk
from vkbottle.bot import Message as VKMessage
from vkbottle.http import aiohttp as vk_aiohttp

from config import Config
from confluence_interaction import (
    make_markup_by_confluence,
    parse_confluence_by_page_id,
)
from database import (
    add_user,
    get_user_id,
    subscribe_user,
    check_subscribing,
    check_spam,
    add_question_answer,
    rate_answer,
    get_subscribed_users,
    get_today_holidays,
    get_history_of_chat,
    filter_chat_history,
    set_stop_point,
)
from strings import Strings
from datetime import datetime, timedelta

STT_URL = Config.STT_URL

VOICE_TEMP_DIR = Path(tempfile.gettempdir()) / "voice_messages"
VOICE_TEMP_DIR.mkdir(exist_ok=True)


def remove_file(path: str):
    try:
        os.remove(path)
    except OSError:
        pass


async def send_to_stt(wav_path: str) -> str:
    """
    Отправляет WAV-файл в микросервис распознавания речи (STT) и возвращает текстовую расшифровку

    Args:
        wav_path (str): Путь к WAV-файлу с голосовым сообщением

    Returns:
        str: Распознанный текст, полученный от микросервиса STT (или пустая строка в случае ошибки)
    """

    form = FormData()
    async with aiofiles.open(wav_path, "rb") as f:
        wav_bytes = await f.read()
    form.add_field(
        name="file",
        value=wav_bytes,
        filename=os.path.basename(wav_path),
        content_type="audio/wav",
    )
    async with ClientSession() as session:
        async with session.post(
            f"http://{Config.QA_HOST}/qa?audio=1/", data=form
        ) as resp:
            if resp.status == 200:
                result = await resp.json()
                return result.get("transcription", "")
            else:
                logging.error(f"STT service returned status {resp.status}")
                return ""


async def download_and_convert_tg(file: tg.types.File, user_id: int) -> str:
    """
    Скачивает голосовое сообщение из Telegram и конвертирует его в формат WAV

    Args:
        file (tg.types.File): Объект файла, полученный из Telegram API
        user_id (int): Идентификатор пользователя Telegram (используется для именования файла)

    Returns:
        str: Путь к сконвертированному WAV-файлу
    """
    ogg_path = (
        VOICE_TEMP_DIR / f"voice_tg_{user_id}_{int(datetime.now().timestamp())}.oga"
    )
    wav_path = ogg_path.with_suffix(".wav")

    await tg_bot.download_file(file.file_path, destination=str(ogg_path))

    try:
        ffmpeg.input(str(ogg_path)).output(str(wav_path), format="wav").run(
            overwrite_output=True
        )
    except Exception as e:
        logging.error(f"TG conversion error: {e}")
        if wav_path.exists():
            wav_path.unlink()
        raise
    finally:
        if ogg_path.exists():
            ogg_path.unlink()
    return str(wav_path)


async def download_and_convert_vk(url: str, user_id: int) -> str:
    """
    Скачивает голосовое сообщение из ВКонтакте и конвертирует его в формат WAV

    Args:
        url (str): Ссылка на аудио-сообщение (.ogg), полученная от VK API
        user_id (int): Идентификатор пользователя ВКонтакте (используется для именования файла)

    Returns:
        str: Путь к сконвертированному WAV-файлу
    """
    ogg_path = (
        VOICE_TEMP_DIR / f"voice_vk_{user_id}_{int(datetime.now().timestamp())}.ogg"
    )
    wav_path = ogg_path.with_suffix(".wav")

    async with vk_aiohttp.ClientSession() as session:
        resp = await session.get(url)
        content = await resp.read()
        async with aiofiles.open(ogg_path, "wb") as f:
            await f.write(content)

    try:
        ffmpeg.input(str(ogg_path)).output(str(wav_path), format="wav").run(
            overwrite_output=True
        )
    except Exception as e:
        logging.error(f"VK conversion error: {e}")
        if wav_path.exists():
            wav_path.unlink()
        raise
    finally:
        if ogg_path.exists():
            ogg_path.unlink()
    return str(wav_path)


class Permission(vk.ABCRule[VKMessage]):
    def __init__(self, user_ids: list):
        self.uids = user_ids

    async def check(self, event: VKMessage):
        return event.from_id in self.uids


routes = web.RouteTableDef()
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
vk_bot = vk.Bot(token=Config.VK_ACCESS_GROUP_TOKEN)
vk_bot.labeler.vbml_ignore_case = True
vk_bot.labeler.custom_rules["permission"] = Permission
tg_bot = tg.Bot(token=Config.TG_ACCESS_TOKEN)
dispatcher = tg.Dispatcher()


def vk_keyboard_choice(notify_text: str) -> str:
    """Возвращает клавиатуру из кнопок предоставления справочной информации
    и подписки на рассылку (если пользователь подписан, то отписки от неё)
    для чат-бота ВКонтакте

    Args:
        notify_text (str): "Подписаться на рассылку" если пользователь не подписан, иначе "Отписаться от рассылки"

    Returns:
        str: JSON-объект, описывающий клавиатуру с шаблонами сообщений
    """

    keyboard = (
        vk.Keyboard()
        .add(vk.Text(Strings.ConfluenceButton))
        .row()
        .add(vk.Text(notify_text))
        .row()
        .add(vk.Text(Strings.NewDialog))
    )
    return keyboard.get_json()


def tg_keyboard_choice(notify_text: str) -> tg.types.ReplyKeyboardMarkup:
    """Возвращает клавиатуру из кнопок предоставления справочной информации
    и подписки на рассылку (если пользователь подписан, то отписки от неё)
    для чат-бота Telegram

    Args:
        notify_text (str): "Подписаться на рассылку" если пользователь не подписан, иначе "Отписаться от рассылки"

    Returns:
        tg.types.ReplyKeyboardMarkup: клавиатура с шаблонами сообщений
    """

    keyboard = tg.types.ReplyKeyboardMarkup(
        keyboard=[
            [tg.types.KeyboardButton(text=Strings.ConfluenceButton)],
            [tg.types.KeyboardButton(text=(notify_text))],
            [tg.types.KeyboardButton(text=Strings.NewDialog)],
        ],
        resize_keyboard=True,
    )
    return keyboard


async def vk_send_confluence_keyboard(message: VKMessage, question_types: list):
    """Создаёт inline-кнопки для чат-бота ВКонтакте на основе справочной структуры
    пространства в вики-системе

    Args:
        message (VKMessage): сообщение пользователя
        question_types (list): страницы или подстраницы из структуры пространства в вики-системе
    """

    keyboards = [
        vk.Keyboard(inline=True) for _ in range(math.ceil(len(question_types) / 5))
    ]
    for i in range(len(question_types)):
        keyboards[i // 5].row()
        keyboards[i // 5].add(
            vk.Text(
                (
                    question_types[i]["content"]["title"]
                    if len(question_types[i]["content"]["title"]) < 40
                    else question_types[i]["content"]["title"][:37] + "..."
                ),
                payload={"conf_id": int(question_types[i]["content"]["id"])},
            )
        )
    keyboard_message = Strings.WhichInfoDoYouWant
    for i in range(len(keyboards)):
        await message.answer(
            message=keyboard_message, keyboard=keyboards[i].get_json(), random_id=0
        )
        keyboard_message = "⠀"


async def tg_send_confluence_keyboard(message: tg.types.Message, question_types: list):
    """Создаёт inline-кнопки для чат-бота Telegram на основе справочной структуры
    пространства в вики-системе

    Args:
        message (tg.types.Message): сообщение пользователя
        question_types (list): страницы или подстраницы из структуры пространства в вики-системе
    """

    keyboard_builder = InlineKeyboardBuilder()

    for item in question_types:
        keyboard_builder.button(
            text=item["content"]["title"],
            callback_data=f"conf_id{item['content']['id']}",
        )

    keyboard_builder.adjust(1)

    await message.answer(
        text=Strings.WhichInfoDoYouWant, reply_markup=keyboard_builder.as_markup()
    )


@vk_bot.on.message(text=[Strings.ConfluenceButton])
async def vk_handler(message: VKMessage):
    """Обработчик события (для чат-бота ВКонтакте), при котором пользователь запрашивает
    справочную информацию

    Args:
        message (VKMessage): сообщение, отправленное пользователем при запросе справочной информации
    """

    question_types = make_markup_by_confluence()
    await vk_send_confluence_keyboard(message, question_types)


@dispatcher.message(tg.F.text.in_([Strings.ConfluenceButton]))
async def tg_handler(message: tg.types.Message):
    """Обработчик события (для чат-бота Telegram), при котором пользователь запрашивает
    справочную информацию

    Args:
        message (tg.types.Message): сообщение, отправленное пользователем при запросе справочной информации
    """

    question_types = make_markup_by_confluence()

    await tg_send_confluence_keyboard(message, question_types)


@vk_bot.on.message(
    func=lambda message: (
        "conf_id" in message.payload if message.payload is not None else False
    )
)
async def vk_confluence_parse(message: VKMessage):
    """Обработчик события (для чат-бота ВКонтакте), при котором пользователь нажимает
    на кнопку, относящуюся к типу или подтипу вопросов

    Args:
        message (VKMessage): сообщение пользователя
    """

    parse = parse_confluence_by_page_id(json.loads(message.payload)["conf_id"])
    if isinstance(parse, list):
        await vk_send_confluence_keyboard(message, parse)
    elif isinstance(parse, str):
        await message.answer(message=parse, random_id=0)


@dispatcher.callback_query(lambda c: c.data.startswith("conf_id"))
async def tg_confluence_parse(callback: tg.types.CallbackQuery):
    """Обработчик события (для чат-бота Telegram), при котором пользователь нажимает
    на кнопку, относящуюся к типу или подтипу вопросов

    Args:
        callback (tg.types.CallbackQuery): запрос при нажатии на inline-кнопку
    """

    parse = parse_confluence_by_page_id(callback.data[7:])
    if isinstance(parse, list):
        await tg_send_confluence_keyboard(callback.message, parse)
    elif isinstance(parse, str):
        await callback.message.answer(text=parse)


@vk_bot.on.message(
    func=lambda message: (
        "score" in message.payload if message.payload is not None else False
    )
)
async def vk_rate(message: VKMessage):
    """Обработчик события (для чат-бота ВКонтакте), при котором пользователь оценивает
    ответ на вопрос

    Args:
        message (VKMessage): сообщение пользователя
    """

    payload_data = json.loads(message.payload)
    if payload_data["score"] == 5:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{Config.QA_HOST}/answer_embed/",
                json={"answer_id": payload_data["question_answer_id"]},
            ):
                pass
    if rate_answer(engine, payload_data["question_answer_id"], payload_data["score"]):
        await message.answer(message=Strings.ThanksForFeedback, random_id=0)


@dispatcher.callback_query()
async def tg_rate(callback_query: tg.types.CallbackQuery):
    """Обработчик события (для чат-бота Telegram), при котором пользователь оценивает
    ответ на вопрос

    Args:
        callback_query (tg.types.CallbackQuery): запрос при нажатии на inline-кнопку
    """

    score, question_answer_id = map(int, str(callback_query.data).split())
    if score == 5:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"http://{Config.QA_HOST}/answer_embed/",
                json={"answer_id": question_answer_id},
            ):
                pass
    if rate_answer(engine, question_answer_id, score):
        await callback_query.answer(text=Strings.ThanksForFeedback)


@vk_bot.on.message(text=[Strings.Subscribe, Strings.Unsubscribe])
async def vk_subscribe(message: VKMessage):
    """Обработчик события (для чат-бота ВКонтакте), при котором пользователь оформляет
    или снимает подписку на рассылку

    Args:
        message (VKMessage): сообщение пользователя
    """

    user_id = get_user_id(engine, vk_id=message.from_id)
    if user_id is None:
        await message.answer(message=Strings.NoneUserVK, random_id=0)
        return
    is_subscribed = subscribe_user(engine, user_id)
    if is_subscribed:
        await message.answer(
            message=Strings.SubscribeMessage,
            keyboard=vk_keyboard_choice(Strings.Unsubscribe),
            random_id=0,
        )
    else:
        await message.answer(
            message=Strings.UnsubscribeMessage,
            keyboard=vk_keyboard_choice(Strings.Subscribe),
            random_id=0,
        )


@dispatcher.message(tg.F.text.in_([Strings.Subscribe, Strings.Unsubscribe]))
async def tg_subscribe(message: tg.types.Message):
    """Обработчик события (для чат-бота Telegram), при котором пользователь оформляет
    или снимает подписку на рассылку

    Args:
        message (tg.types.Message): сообщение пользователя
    """

    user_id = get_user_id(engine, telegram_id=message.from_user.id)
    if user_id is None:
        await message.answer(text=Strings.NoneUserTelegram)
        return
    is_subscribed = subscribe_user(engine, user_id)
    if is_subscribed:
        await message.reply(
            text=Strings.SubscribeMessage,
            reply_markup=tg_keyboard_choice(Strings.Unsubscribe),
        )
    else:
        await message.reply(
            text=Strings.UnsubscribeMessage,
            reply_markup=tg_keyboard_choice(Strings.Subscribe),
        )


@dispatcher.message(filters.CommandStart())
async def tg_start(message: tg.types.Message):
    """Обработчик события (для чат-бота Telegram), при котором пользователь отправляет
    команду /start

    Args:
        message (tg.types.Message): сообщение пользователя
    """
    is_user_added, user_id = add_user(engine, telegram_id=message.from_user.id)
    notify_text = (
        Strings.Unsubscribe if check_subscribing(engine, user_id) else Strings.Subscribe
    )
    if (
        is_user_added
        or Strings.Start in message.text.lower()
        or Strings.StartEnglish in message.text.lower()
    ):
        await message.answer(
            text=Strings.FirstMessage, reply_markup=tg_keyboard_choice(notify_text)
        )


@vk_bot.on.message(text=[Strings.Start, Strings.StartEnglish])
async def vk_start(message: VKMessage):
    """Обработчик события (для чат-бота ВКонтакте), при котором пользователь отправляет
    команду /start

    Args:
        message (VKMessage): сообщение пользователя
    """
    is_user_added, user_id = add_user(engine, vk_id=message.from_id)
    notify_text = (
        Strings.Unsubscribe if check_subscribing(engine, user_id) else Strings.Subscribe
    )
    if is_user_added:
        await message.answer(
            message=Strings.FirstMessage,
            keyboard=vk_keyboard_choice(notify_text),
            random_id=0,
        )


async def process_message_text(
    text: str,
    user_id: int,
    chat_id: int,
    platform: str,
    bot_instance: tg.Bot | vk.Bot,
    random_id: int = 0,
) -> None:
    """Универсальная функция для обработки текстовых сообщений

    Args:
        text (str): Текст сообщения для обработки
        user_id (int): ID пользователя
        chat_id (int): ID чата
        platform (str): Платформа ('telegram' или 'vk')
        bot_instance (tg.Bot | vk.Bot): Экземпляр бота для отправки сообщений
        random_id (int, optional): ID для VK сообщений. По умолчанию 0.
    """
    if platform == "telegram":
        is_user_added, user_id = add_user(engine, telegram_id=user_id)
    else:
        is_user_added, user_id = add_user(engine, vk_id=user_id)

    notify_text = (
        Strings.Unsubscribe if check_subscribing(engine, user_id) else Strings.Subscribe
    )

    if len(text) < 4:
        if platform == "telegram":
            await bot_instance.send_message(chat_id=chat_id, text=Strings.Less4Symbols)
        else:
            await bot_instance.api.messages.send(
                user_id=chat_id, message=Strings.Less4Symbols, random_id=random_id
            )
        return

    if user_id is None:
        if platform == "telegram":
            await bot_instance.send_message(
                chat_id=chat_id, text=Strings.NoneUserTelegram
            )
        else:
            await bot_instance.api.messages.send(
                user_id=chat_id, message=Strings.NoneUserVK, random_id=random_id
            )
        return

    if check_spam(engine, user_id):
        if platform == "telegram":
            await bot_instance.send_message(chat_id=chat_id, text=Strings.SpamWarning)
        else:
            await bot_instance.api.messages.send(
                user_id=chat_id, message=Strings.SpamWarning, random_id=random_id
            )
        return

    if platform == "telegram":
        processing = await bot_instance.send_message(
            chat_id=chat_id, text=Strings.TryFindAnswer
        )
    else:
        processing = await bot_instance.api.messages.send(
            user_id=chat_id, message=Strings.TryFindAnswer, random_id=random_id
        )

    answer, confluence_url = await get_answer(text, user_id=user_id)
    question_answer_id = add_question_answer(
        engine, text, answer, confluence_url, user_id
    )

    if platform == "telegram":
        await bot_instance.delete_message(
            chat_id=chat_id, message_id=processing.message_id
        )
    else:
        await bot_instance.api.messages.delete(
            message_ids=[processing.message_id], peer_id=chat_id, delete_for_all=True
        )

    if confluence_url is None:
        if platform == "telegram":
            await bot_instance.send_message(chat_id=chat_id, text=Strings.NotFound)
        else:
            await bot_instance.api.messages.send(
                user_id=chat_id, message=Strings.NotFound, random_id=random_id
            )
        return

    if len(answer) == 0:
        answer = Strings.NotAnswer

    if platform == "telegram":
        keyboard = tg.types.InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    tg.types.InlineKeyboardButton(
                        text="👎", callback_data=f"1 {question_answer_id}"
                    ),
                    tg.types.InlineKeyboardButton(
                        text="❤", callback_data=f"5 {question_answer_id}"
                    ),
                ]
            ]
        )
        await bot_instance.send_message(
            chat_id=chat_id,
            text=f"{answer}\n\n{Strings.SourceURL} {confluence_url}",
            reply_markup=keyboard,
        )
    else:
        keyboard = (
            vk.Keyboard(inline=True)
            .add(
                vk.Text(
                    "👎", payload={"score": 1, "question_answer_id": question_answer_id}
                )
            )
            .add(
                vk.Text(
                    "❤", payload={"score": 5, "question_answer_id": question_answer_id}
                )
            )
        )
        await bot_instance.api.messages.send(
            user_id=chat_id,
            message=f"{answer}\n\n{Strings.SourceURL} {confluence_url}",
            keyboard=keyboard.get_json(),
            random_id=random_id,
        )


@dispatcher.message(F.voice)
async def tg_voice_handler(message: tg.types.Message):
    """
    Обрабатывает голосовое сообщение из Telegram: скачивает, конвертирует, распознаёт и отвечает текстом

    Args:
        message (tg.types.Message): Входящее сообщение с голосом от пользователя Telegram
    """
    try:
        logging.info("Starting Telegram voice message processing")
        file = await tg_bot.get_file(message.voice.file_id)
        logging.info(f"Got file info: {file.file_path}")

        wav = await download_and_convert_tg(file, message.from_user.id)
        logging.info(f"Converted to WAV: {wav}")

        text = await send_to_stt(wav)
        logging.info(f"STT result: {text}")

        remove_file(wav)
        logging.info("Temporary files cleaned up")

        if not text:
            logging.warning("Empty transcription received")
            await message.reply("Не удалось распознать голосовое сообщение")
            return

        await process_message_text(
            text=text,
            user_id=message.from_user.id,
            chat_id=message.chat.id,
            platform="telegram",
            bot_instance=tg_bot,
        )
        logging.info("Voice message processing completed successfully")

    except Exception as e:
        logging.error(
            f"Error processing voice message in Telegram: {str(e)}", exc_info=True
        )
        await message.reply("Произошла ошибка при обработке голосового сообщения")


@dispatcher.message()
async def tg_answer(message: tg.types.Message):
    """Обработчик текстовых сообщений для Telegram

    Args:
        message (tg.types.Message): Входящее текстовое сообщение
    """
    await process_message_text(
        text=message.text,
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        platform="telegram",
        bot_instance=tg_bot,
    )


@vk_bot.on.message(
    func=lambda m: m.attachments
    and any(att.type == "audio_message" for att in m.attachments)
)
async def vk_voice_handler(message: VKMessage):
    """
    Обрабатывает голосовое сообщение из ВКонтакте: скачивает, конвертирует, распознаёт и отвечает текстом

    Args:
        message (VKMessage): Входящее сообщение от пользователя ВКонтакте с вложенным аудио-сообщением
    """
    try:
        logging.info("Starting VK voice message processing")
        for att in message.attachments:
            if att.type == "audio_message":
                logging.info(
                    f"Found audio message attachment: {att.audio_message.link_ogg}"
                )

                wav = await download_and_convert_vk(
                    att.audio_message.link_ogg, message.from_id
                )
                logging.info(f"Converted to WAV: {wav}")

                text = await send_to_stt(wav)
                logging.info(f"STT result: {text}")

                remove_file(wav)
                logging.info("Temporary files cleaned up")

                if not text:
                    logging.warning("Empty transcription received")
                    await message.answer(
                        "Не удалось распознать голосовое сообщение", random_id=0
                    )
                    return

                await process_message_text(
                    text=text,
                    user_id=message.from_id,
                    chat_id=message.peer_id,
                    platform="vk",
                    bot_instance=vk_bot,
                    random_id=0,
                )
                logging.info("Voice message processing completed successfully")
                break

    except Exception as e:
        logging.error(f"Error processing voice message in VK: {str(e)}", exc_info=True)
        await message.answer(
            "Произошла ошибка при обработке голосового сообщения", random_id=0
        )


@vk_bot.on.message()
async def vk_answer(message: VKMessage):
    """Обработчик текстовых сообщений для ВКонтакте

    Args:
        message (VKMessage): Входящее текстовое сообщение
    """
    await process_message_text(
        text=message.text,
        user_id=message.from_id,
        chat_id=message.peer_id,
        platform="vk",
        bot_instance=vk_bot,
        random_id=0,
    )


@dispatcher.message(tg.F.text.in_([Strings.NewDialog]))
async def tg_new_dialog(message: tg.types.Message):
    """Обработчик для начала нового диалога"""
    user_id = get_user_id(engine, telegram_id=message.from_user.id)

    if user_id is None:
        await message.answer(text=Strings.NoneUserTelegram)
        return

    set_stop_point(engine, user_id, True)
    await message.answer(text=Strings.DialogReset)


@vk_bot.on.message(text=[Strings.NewDialog])
async def vk_new_dialog(message: VKMessage):
    user_id = get_user_id(engine, vk_id=message.from_id)

    if user_id is None:
        await message.answer(message=Strings.NoneUserVK, random_id=0)
        return

    set_stop_point(engine, user_id, True)
    await message.answer(
        message=Strings.DialogReset,
        random_id=0,
    )


async def get_answer(question: str, user_id: int) -> tuple[str, str | None]:
    """Получение ответа на вопрос с использованием микросервиса

    Args:
        question (str): вопрос пользователя

    Returns:
        tuple[str, str | None]: ответ на вопрос и ссылка на страницу в вики-системе
    """
    chat_history = get_history_of_chat(engine, user_id)
    answered_pairs, recent_unanswered = filter_chat_history(chat_history)
    question = question.strip().lower()

    dialog_context = []
    for qa in answered_pairs:
        dialog_context.append(f"Q: {qa.question}")
        dialog_context.append(f"A: {qa.answer}")

    for unanswered_question in recent_unanswered:
        dialog_context.append(f"Q: {unanswered_question.question}")

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://{Config.QA_HOST}/qa?audio=0/",
            json={"question": question, "dialog_context": dialog_context},
        ) as response:
            if response.status == 200:
                resp = await response.json()
                return resp["answer"], resp["confluence_url"]
            else:
                return ("", None)


@routes.post("/broadcast/")
async def broadcast(request: web.Request) -> web.Response:
    """Создает рассылку в ВК и/или ТГ

    Args:
        request (web.Request): запрос, содержащий `text`, булевые `tg`, `vk`

    Returns:
        web.Response: ответ
    """

    try:
        data = await request.json()
        if not data["vk"] and not data["tg"]:
            raise Exception("ничего не выбрано")
        if len(data["text"]) < 3:
            raise Exception("в сообщении меньше трёх символов")
        vk_users, tg_users = get_subscribed_users(engine)
        vk_count, tg_count = 0, 0
        if data["vk"]:
            for user_id in vk_users:
                try:
                    await vk_bot.api.messages.send(
                        user_id=user_id, message=data["text"], random_id=0
                    )
                    vk_count += 1
                except vk.VKAPIError:
                    continue
        if data["tg"]:
            for user_id in tg_users:
                try:
                    await tg_bot.send_message(chat_id=user_id, text=data["text"])
                    tg_count += 1
                except TUerror:
                    await asyncio.sleep(1)
        return web.Response(
            text=f"Осуществлена массовая рассылка пользователям VK: {vk_count}, Telegram: {tg_count}",
            status=200,
        )
    except Exception as e:
        return web.Response(
            text=f"В ходе массовой рассылки произошла ошибка: {str(e)}",
            status=500,
        )


async def get_greeting(
    template: str, user_name: str, holiday_name: str, retries=3
) -> str:
    """
    Отправляет запрос к QA-сервису для генерации поздравления и возвращает полученный результат.

    Args:
        template (str): Шаблон поздравления.
        user_name (str): Имя пользователя.
        holiday_name (str): Название праздника.
        retries (int, optional): Количество повторных попыток запроса. По умолчанию 3.

    Returns:
        str: Сгенерированное поздравление или пустая строка при ошибке.
    """

    url = f"http://{Config.QA_HOST}/generate_greeting/"
    data = {"template": template, "user_name": user_name, "holiday_name": holiday_name}

    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data) as resp:
                    logging.info(f"Статус: {resp.status}")
                    if resp.status == 200:
                        resp_json = await resp.json()
                        greeting = resp_json.get("greeting", "")
                        logging.info(f"Получено: {greeting}")
                        return greeting
                    else:
                        error_text = await resp.text()
                        logging.error(f"Ошибка {resp.status}: {error_text}")
        except ClientError as e:
            logging.error(f"Попытка {attempt+1}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2)
        except Exception as e:
            logging.exception(f"Ошибка: {e}")
            break
    logging.error("Поздравление не получено")
    return ""


async def get_vk_user_name(user_id: int) -> str:
    """Получает имя пользователя VK.

    Args:
        user_id (int): ID пользователя VK.

    Returns:
        str: Имя пользователя.
    """

    user_info = await vk_bot.api.users.get(user_ids=user_id)
    if user_info:
        return user_info[0].first_name
    return "пользователь"


async def get_telegram_user_name(user_id: int) -> str:
    """Получает имя пользователя Telegram.

    Args:
        user_id (int): ID пользователя Telegram.

    Returns:
        str: Имя пользователя.
    """

    try:
        user = await tg_bot.get_chat(user_id)
        if user.first_name:
            return user.first_name
    except Exception as e:
        logging.error(f"Ошибка получения имени пользователя Telegram: {e}")
    return "пользователь"


async def get_vk_user_gender(user_id: int) -> str:

    user_info = await vk_bot.api.users.get(user_ids=user_id, fields=["sex"])
    if user_info and user_info[0].sex == 2:
        return "male"
    elif user_info and user_info[0].sex == 1:
        return "female"
    return "unknown"


async def send_vk_message(user_id: int, greeting: str, delay: int = 2):
    try:
        await vk_bot.api.messages.send(user_id=user_id, message=greeting, random_id=0)
        await asyncio.sleep(delay)
    except vk.VKAPIError as e:
        logging.error(f"Ошибка отправки поздравления в VK для {user_id}: {e}")


async def send_tg_message(user_id: int, greeting: str, delay: int = 2):
    try:
        await tg_bot.send_message(chat_id=user_id, text=greeting)
        await asyncio.sleep(delay)
    except Exception as e:
        logging.error(f"Ошибка отправки поздравления в Telegram для {user_id}: {e}")


async def send_holiday_greetings():
    """Отправляет поздравления пользователям VK и Telegram в праздничные дни."""
    holidays = get_today_holidays(engine)
    if not holidays:
        return

    vk_users, tg_users = get_subscribed_users(engine)
    logging.info(f"Subscribed Telegram users: {tg_users}")

    DELAY_BETWEEN_MESSAGES = 2
    DELAY_BETWEEN_PLATFORMS = 5

    for user_id in vk_users:
        gender = await get_vk_user_gender(user_id)
        user_name = await get_vk_user_name(user_id)

        for holiday in holidays:
            if holiday.vk and (
                (holiday.male_holiday and gender == "male")
                or (holiday.female_holiday and gender == "female")
                or (holiday.male_holiday and holiday.female_holiday)
            ):
                greeting = await get_greeting(
                    holiday.template_llm,
                    user_name,
                    holiday.name,
                )
                await send_vk_message(user_id, greeting, DELAY_BETWEEN_MESSAGES)

    await asyncio.sleep(DELAY_BETWEEN_PLATFORMS)

    for user_id in tg_users:
        user_name = await get_telegram_user_name(user_id)

        for holiday in holidays:
            if holiday.tg and not (holiday.male_holiday ^ holiday.female_holiday):
                greeting = await get_greeting(
                    holiday.template_llm,
                    user_name,
                    holiday.name,
                )
                await send_tg_message(user_id, greeting, DELAY_BETWEEN_MESSAGES)


async def check_and_send_greetings():
    """Функция запуска модуля отправки поздравлений"""

    while True:
        now = datetime.now()
        today_run_time = now.replace(hour=13, minute=25, second=0, microsecond=0)
        if now >= today_run_time:
            holidays = get_today_holidays(engine)
            if holidays:
                logging.info("Triggering holiday greetings send (late start).")
                await send_holiday_greetings()
            else:
                logging.info("No holidays today. Skipping greetings for today.")
            next_run = today_run_time + timedelta(days=1)
        else:
            logging.info(
                f"Waiting for next run at {today_run_time.strftime('%H:%M:%S')}. Current time: {now.strftime('%H:%M:%S')}"
            )
            next_run = today_run_time

        sleep_duration = (next_run - now).total_seconds()
        logging.info(f"Sleeping for {sleep_duration} seconds until next schedule.")
        await asyncio.sleep(sleep_duration)


def launch_vk_bot():
    """Функция начала работы чат-бота ВКонтакте"""

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    vk_bot.run_forever()


async def launch_telegram_bot():
    """Функция начала работы чат-бота Telegram"""

    await dispatcher.start_polling(tg_bot)


def run_telegram_process():
    """Запуск Telegram-бота в новом цикле событий в отдельном процессе"""

    asyncio.run(launch_telegram_bot())


def run_web_app():
    """Функция запуска сервера для принятия запроса на рассылку"""

    app = web.Application()
    app.add_routes(routes)
    web.run_app(app, port=5000)


def launch_greeting_service():
    """Функция запуска модуля отправки поздравлений"""

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(check_and_send_greetings())
    loop.close()


if __name__ == "__main__":
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    for logger in loggers:
        logger.setLevel(logging.WARNING)
    web_process = Process(target=run_web_app)
    vk_process = Process(target=launch_vk_bot)
    greeting_process = Process(target=launch_greeting_service)
    tg_process = Process(target=run_telegram_process)
    web_process.start()
    vk_process.start()
    tg_process.start()
    greeting_process.start()
    web_process.join()
    vk_process.join()
    tg_process.join()
    greeting_process.join()
