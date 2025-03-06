import asyncio
import json
import logging
import math
from multiprocessing import Process
import aiogram as tg
from aiohttp import web, ClientError
from sqlalchemy import create_engine
import vkbottle as vk
from vkbottle.bot import Message as VKMessage
from vkbottle.http import aiohttp
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
)
from strings import Strings
from datetime import datetime, timedelta


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
dispatcher = tg.Dispatcher(tg_bot)


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

    keyboard = tg.types.ReplyKeyboardMarkup(resize_keyboard=True)
    keyboard.add(tg.types.KeyboardButton(Strings.ConfluenceButton))
    keyboard.add(tg.types.KeyboardButton(notify_text))
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

    inline_keyboard = tg.types.InlineKeyboardMarkup()
    for i in question_types:
        inline_keyboard.add(
            tg.types.InlineKeyboardButton(
                text=i["content"]["title"], callback_data=f"conf_id{i['content']['id']}"
            )
        )
    await message.answer(text=Strings.WhichInfoDoYouWant, reply_markup=inline_keyboard)


@vk_bot.on.message(text=[Strings.ConfluenceButton])
async def vk_handler(message: VKMessage):
    """Обработчик события (для чат-бота ВКонтакте), при котором пользователь запрашивает
    справочную информацию

    Args:
        message (VKMessage): сообщение, отправленное пользователем при запросе справочной информации
    """

    question_types = make_markup_by_confluence()
    await vk_send_confluence_keyboard(message, question_types)


@dispatcher.message_handler(text=[Strings.ConfluenceButton])
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


@dispatcher.callback_query_handler(lambda c: c.data.startswith("conf_id"))
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
    if rate_answer(engine, payload_data["question_answer_id"], payload_data["score"]):
        await message.answer(message=Strings.ThanksForFeedback, random_id=0)


@dispatcher.callback_query_handler()
async def tg_rate(callback_query: tg.types.CallbackQuery):
    """Обработчик события (для чат-бота Telegram), при котором пользователь оценивает
    ответ на вопрос

    Args:
        callback_query (tg.types.CallbackQuery): запрос при нажатии на inline-кнопку
    """

    score, question_answer_id = map(int, callback_query.data.split())
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


@dispatcher.message_handler(text=[Strings.Subscribe, Strings.Unsubscribe])
async def tg_subscribe(message: tg.types.Message):
    """Обработчик события (для чат-бота Telegram), при котором пользователь оформляет
    или снимает подписку на рассылку

    Args:
        message (tg.types.Message): сообщение пользователя
    """

    user_id = get_user_id(engine, telegram_id=message["from"]["id"])
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


async def get_answer(question: str) -> tuple[str, str | None]:
    """Получение ответа на вопрос с использованием микросервиса

    Args:
        question (str): вопрос пользователя

    Returns:
        tuple[str, str | None]: ответ на вопрос и ссылка на страницу в вики-системе
    """

    question = question.strip().lower()
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"http://{Config.QA_HOST}/qa/", json={"question": question}
        ) as response:
            if response.status == 200:
                resp = await response.json()
                return resp["answer"], resp["confluence_url"]
            else:
                return ("", None)


@vk_bot.on.message()
async def vk_answer(message: VKMessage):
    """Обработчик события (для чат-бота ВКонтакте), при котором пользователь задаёт
    вопрос чат-боту

    После отображения ответа на вопрос чат-бот отправляет inline-кнопки для оценивания
    ответа

    Args:
        message (VKMessage): сообщение пользователя с вопросом
    """

    is_user_added, user_id = add_user(engine, vk_id=message.from_id)
    notify_text = (
        Strings.Unsubscribe if check_subscribing(engine, user_id) else Strings.Subscribe
    )
    if (
        is_user_added
        or Strings.Start in message.text.lower()
        or Strings.StartEnglish in message.text.lower()
    ):
        await message.answer(
            message=Strings.FirstMessage,
            keyboard=vk_keyboard_choice(notify_text),
            random_id=0,
        )
        return
    if len(message.text) < 4:
        await message.answer(message=Strings.Less4Symbols, random_id=0)
        return
    if check_spam(engine, user_id):
        await message.answer(message=Strings.SpamWarning, random_id=0)
        return
    processing = await message.answer(message=Strings.TryFindAnswer, random_id=0)
    answer, confluence_url = await get_answer(message.text)
    question_answer_id = add_question_answer(
        engine, message.text, answer, confluence_url, user_id
    )
    if processing.message_id is not None:
        await vk_bot.api.messages.delete(
            message_ids=[processing.message_id],
            peer_id=message.peer_id,
            delete_for_all=True,
        )
    if confluence_url is None:
        await message.answer(
            message=Strings.NotFound,
            keyboard=vk_keyboard_choice(notify_text),
            random_id=0,
        )
        return
    if len(answer) == 0:
        answer = Strings.NotAnswer
    await message.answer(
        message=f"{answer}\n\n{Strings.SourceURL} {confluence_url}",
        dont_parse_links=True,
        keyboard=(
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
        ),
        random_id=0,
    )


@dispatcher.message_handler(commands=["start"])
async def tg_start(message: tg.types.Message):
    """Обработчик события (для чат-бота Telegram), при котором пользователь отправляет
    команду /start

    Args:
        message (tg.types.Message): сообщение пользователя
    """

    is_user_added, user_id = add_user(engine, telegram_id=message["from"]["id"])
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


@dispatcher.message_handler()
async def tg_answer(message: tg.types.Message):
    """Обработчик события (для чат-бота Telegram), при котором пользователь задаёт
    вопрос чат-боту

    После отображения ответа на вопрос чат-бот отправляет inline-кнопки для оценивания
    ответа

    Args:
        message (tg.types.Message): сообщение с вопросом пользователя
    """

    if len(message["text"]) < 4:
        await message.answer(text=Strings.Less4Symbols)
        return
    user_id = get_user_id(engine, telegram_id=message["from"]["id"])
    if user_id is None:
        await message.answer(text=Strings.NoneUserTelegram)
        return
    if check_spam(engine, user_id):
        await message.answer(text=Strings.SpamWarning)
        return
    processing = await message.answer(Strings.TryFindAnswer)
    answer, confluence_url = await get_answer(message.text)
    question_answer_id = add_question_answer(
        engine, message.text, answer, confluence_url, user_id
    )
    await tg_bot.delete_message(message["chat"]["id"], processing["message_id"])
    if confluence_url is None:
        await message.answer(text=Strings.NotFound)
        return
    if len(answer) == 0:
        answer = Strings.NotAnswer
    await message.answer(
        text=f"{answer}\n\n{Strings.SourceURL} {confluence_url}",
        reply_markup=tg.types.InlineKeyboardMarkup().add(
            tg.types.InlineKeyboardButton(
                text="👎", callback_data=f"1 {question_answer_id}"
            ),
            tg.types.InlineKeyboardButton(
                text="❤", callback_data=f"5 {question_answer_id}"
            ),
        ),
    )


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
                except tg.utils.exceptions.BotBlocked:
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
        next_run = now.replace(hour=14, minute=0, second=0, microsecond=0)

        if now.hour >= 14:
            await send_holiday_greetings()
            next_run = next_run + timedelta(days=1)

        await asyncio.sleep((next_run - now).total_seconds())
        await send_holiday_greetings()


def launch_vk_bot():
    """Функция начала работы чат-бота ВКонтакте"""

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    vk_bot.run_forever()


def launch_telegram_bot():
    """Функция начала работы чат-бота Telegram"""

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tg.executor.start_polling(dispatcher, skip_updates=True)


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
    tg_process = Process(target=launch_telegram_bot)
    greeting_process = Process(target=launch_greeting_service)
    web_process.start()
    vk_process.start()
    tg_process.start()
    greeting_process.start()
    web_process.join()
    vk_process.join()
    tg_process.join()
    greeting_process.join()
