import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from aiohttp import web, ClientSession, ClientError
from sqlalchemy import create_engine

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


routes = web.RouteTableDef()
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)

# --- Max messenger settings ---
MAX_API_BASE = Config.MAX_API_BASE
MAX_ACCESS_TOKEN = Config.MAX_ACCESS_TOKEN


def _auth_header() -> Dict[str, str]:
    token = MAX_ACCESS_TOKEN or ""
    return {"Authorization": token}


async def send_max_message(
    recipient_id: int | str,
    text: str,
    *,
    attachments: Optional[List[Dict[str, Any]]] = None,
    format: Optional[str] = None,
    disable_links_preview: bool = True,
) -> bool:
    """Отправляет сообщение пользователю Max через POST /messages."""
    url = f"{MAX_API_BASE}/messages"
    headers = {"Content-Type": "application/json", **_auth_header()}
    payload: Dict[str, Any] = {"recipient_id": str(recipient_id), "text": text}
    if format:
        payload["format"] = format
    if attachments:
        payload["attachments"] = attachments
    if disable_links_preview:
        payload["disable_links_preview"] = True
    try:
        async with ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=15) as r:
                if r.status >= 400:
                    body = await r.text()
                    logging.error("Max send failed: %s %s", r.status, body)
                    return False
                return True
    except ClientError as e:
        logging.error("HTTP error sending to Max: %s", e)
        return False


def max_inline_keyboard(button_rows: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return [
        {
            "type": "inline_keyboard",
            "payload": {"buttons": button_rows},
        }
    ]


def help_keyboard_rows() -> List[List[Dict[str, Any]]]:
    return [
        [{"type": "message", "text": Strings.ConfluenceButton}],
        [{"type": "message", "text": Strings.NewDialog}],
    ]


def confluence_keyboard(question_types: list) -> List[Dict[str, Any]]:
    rows: List[List[Dict[str, Any]]] = []
    for item in question_types:
        title = item["content"]["title"]
        pid = item["content"]["id"]
        rows.append([
            {"type": "callback", "text": title, "payload": f"conf_id:{pid}"}
        ])
    return max_inline_keyboard(rows)


def rating_keyboard(qa_id: int) -> List[Dict[str, Any]]:
    rows = [[
        {"type": "callback", "text": "👎", "payload": f"rate:1:{qa_id}"},
        {"type": "callback", "text": "❤", "payload": f"rate:5:{qa_id}"},
    ]]
    return max_inline_keyboard(rows)


async def get_answer(question: str, user_id: int) -> tuple[str, str | None]:
    chat_history = get_history_of_chat(engine, user_id)
    answered_pairs, recent_unanswered = filter_chat_history(chat_history)
    question_norm = question.strip().lower()
    dialog_context: List[str] = []
    for qa in answered_pairs:
        dialog_context.append(f"Q: {qa.question}")
        dialog_context.append(f"A: {qa.answer}")
    for unanswered in recent_unanswered:
        dialog_context.append(f"Q: {unanswered.question}")
    url = f"http://{Config.QA_HOST}/qa/"
    try:
        async with ClientSession() as session:
            async with session.post(url, json={"question": question_norm, "dialog_context": dialog_context}, timeout=20) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("answer", ""), data.get("confluence_url")
                logging.error("QA service returned %s", resp.status)
                return "", None
    except Exception as e:
        logging.exception("QA request failed: %s", e)
        return "", None


async def handle_help(recipient_id: int | str):
    question_types = make_markup_by_confluence()
    attachments = confluence_keyboard(question_types)
    await send_max_message(recipient_id, Strings.WhichInfoDoYouWant, attachments=attachments)


async def handle_conf_callback(recipient_id: int | str, payload: str):
    conf_id = payload.split(":", 1)[1] if ":" in payload else payload
    parse = parse_confluence_by_page_id(conf_id)
    if isinstance(parse, list):
        attachments = confluence_keyboard(parse)
        await send_max_message(recipient_id, Strings.WhichInfoDoYouWant, attachments=attachments)
    elif isinstance(parse, str):
        await send_max_message(recipient_id, parse)


def _extract_sender_and_text(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    ev_type = payload.get("type") or payload.get("event", {}).get("type")
    data = payload.get("data") or payload.get("event", {}).get("data") or payload
    sender_id = None
    text = None
    cb_payload = None
    if ev_type in ("message", "message_new", "message.new"):
        sender_id = str(data.get("from") or data.get("sender_id") or data.get("user_id") or data.get("chat_id") or "")
        text = (data.get("text") or "").strip()
    elif ev_type in ("message_callback", "message.callback"):
        sender_id = str(data.get("user_id") or data.get("from") or data.get("sender_id") or "")
        cb_payload = data.get("payload") or data.get("callback_payload")
    else:
        msg = data.get("message") or {}
        sender_id = str(msg.get("from") or msg.get("sender_id") or msg.get("user_id") or data.get("user_id") or "")
        text = (msg.get("text") or data.get("text") or "").strip()
    if sender_id == "":
        sender_id = None
    return sender_id, text, cb_payload


async def handle_rate_payload(recipient_id: str, payload: str):
    try:
        _, score_s, qa_id_s = payload.split(":", 2)
        score = int(score_s)
        qa_id = int(qa_id_s)
        if score == 5:
            try:
                async with ClientSession() as session:
                    await session.post(
                        f"http://{Config.QA_HOST}/answer_embed/",
                        json={"answer_id": qa_id},
                        timeout=10,
                    )
            except Exception:
                logging.exception("answer_embed post failed")
        if rate_answer(engine, qa_id, score):
            await send_max_message(recipient_id, Strings.ThanksForFeedback)
    except Exception:
        logging.exception("bad rate payload: %s", payload)


async def process_text_message(recipient_id: str, text: str):
    try:
        numeric_id = int(recipient_id)
    except Exception:
        logging.error("Non-numeric Max user id: %s (DB expects int).", recipient_id)
        return
    is_added, user_db_id = add_user(engine, vk_id=numeric_id)
    lowered = text.lower()
    if lowered in (Strings.Start, Strings.StartEnglish):
        await send_max_message(recipient_id, Strings.FirstMessage, attachments=max_inline_keyboard(help_keyboard_rows()))
        return
    if lowered == Strings.ConfluenceButton.lower():
        await handle_help(recipient_id)
        return
    if lowered in (Strings.Subscribe.lower(), Strings.Unsubscribe.lower()):
        is_subscribed = subscribe_user(engine, user_db_id)
        msg = Strings.SubscribeMessage if is_subscribed else Strings.UnsubscribeMessage
        await send_max_message(recipient_id, msg, attachments=max_inline_keyboard(help_keyboard_rows()))
        return
    if lowered == Strings.NewDialog.lower():
        set_stop_point(engine, user_db_id, True)
        await send_max_message(recipient_id, Strings.DialogReset, attachments=max_inline_keyboard(help_keyboard_rows()))
        return
    if lowered.startswith("conf_id") or lowered.isdigit():
        conf_id = lowered.replace("conf_id", "").replace(":", "").strip()
        parse = parse_confluence_by_page_id(conf_id)
        if isinstance(parse, list):
            attachments = confluence_keyboard(parse)
            await send_max_message(recipient_id, Strings.WhichInfoDoYouWant, attachments=attachments)
        elif isinstance(parse, str):
            await send_max_message(recipient_id, parse)
        return
    if lowered.startswith("rate"):
        parts = lowered.split()
        if len(parts) >= 3:
            try:
                score = int(parts[1]); qa_id = int(parts[2])
                if score == 5:
                    try:
                        async with ClientSession() as session:
                            await session.post(f"http://{Config.QA_HOST}/answer_embed/", json={"answer_id": qa_id}, timeout=10)
                    except Exception:
                        logging.exception("answer_embed post failed")
                if rate_answer(engine, qa_id, score):
                    await send_max_message(recipient_id, Strings.ThanksForFeedback)
            except Exception:
                pass
        return
    if len(text) < 4:
        await send_max_message(recipient_id, Strings.Less4Symbols, attachments=max_inline_keyboard(help_keyboard_rows()))
        return
    if check_spam(engine, user_db_id):
        await send_max_message(recipient_id, Strings.SpamWarning, attachments=max_inline_keyboard(help_keyboard_rows()))
        return
    await send_max_message(recipient_id, Strings.TryFindAnswer)
    answer, confluence_url = await get_answer(text, user_db_id)
    qa_id = add_question_answer(engine, text, answer, confluence_url, user_db_id)
    if not confluence_url:
        await send_max_message(recipient_id, Strings.NotFound, attachments=max_inline_keyboard(help_keyboard_rows()))
        return
    if not answer:
        answer = Strings.NotAnswer
    msg_text = f"{answer}\n\n{Strings.SourceURL} {confluence_url}\n\nОцените ответ: ❤ или 👎 или 'rate 5 {qa_id}'"
    await send_max_message(recipient_id, msg_text, attachments=rating_keyboard(qa_id))


@routes.post("/webhook/")
async def max_webhook(request: web.Request) -> web.Response:
    try:
        body = await request.json()
    except Exception:
        return web.Response(status=400, text="bad json")
    events = body if isinstance(body, list) else [body]
    for ev in events:
        sender_id, text, cb = _extract_sender_and_text(ev)
        if not sender_id:
            continue
        if cb:
            if isinstance(cb, str) and cb.startswith("conf_id"):
                await handle_conf_callback(sender_id, cb)
            elif isinstance(cb, str) and cb.startswith("rate:"):
                await handle_rate_payload(sender_id, cb)
            else:
                await process_text_message(sender_id, cb)
        elif text:
            await process_text_message(sender_id, text)
    return web.Response(status=200)


async def get_max_user_name(user_id: int | str) -> str:
    try:
        url = f"{MAX_API_BASE}/me"; headers = _auth_header()
        async with ClientSession() as session:
            async with session.get(url, headers=headers, timeout=8) as resp:
                if resp.status == 200:
                    j = await resp.json(); return j.get("name") or "пользователь"
    except Exception:
        pass
    return "пользователь"


async def on_cleanup(app: web.Application):
    task = app.get("greetings_task")
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    app = web.Application()
    app.add_routes(routes)
    app.on_cleanup.append(on_cleanup)
    web.run_app(app, port=5000)
