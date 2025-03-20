from contextlib import redirect_stderr
import io
import logging
import numpy as np
from aiohttp import web
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from config import Config
from database import Chunk, set_embedding, get_answer_by_id, get_the_highest_question
from confluence_retrieving import get_chunk, reindex_confluence

routes = web.RouteTableDef()
engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4096,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
encoder_model = SentenceTransformer(
    "saved_models/multilingual-e5-large-wikiutmn", device="cpu"
)


def get_answer(context: str, question: str) -> str:
    """Отправляет запрос к LLM и возвращает ответ."""
    try:
        headers = Config.get_mistral_headers()
        prompt = Config.get_default_prompt(context, question)
        response = requests.post(Config.MISTRAL_API_URL, json=prompt, headers=headers)
        if response.status_code == 200:
            data = response.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
        else:
            logging.error(f"Ошибка {response.status_code}: {response.text}")
            return ""
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return ""


@routes.post("/answer_embed/")
async def set_answer_embedding(
    request: web.Request, encoder_model: SentenceTransformer
):
    qa_id = (await request.json())["answer_id"]
    answer_text = get_answer_by_id(engine=engine, question_answer_id=qa_id)
    if not answer_text:
        return
    embedding = encoder_model.encode(answer_text)
    set_embedding(engine=engine, question_answer_id=qa_id, embed=embedding)


async def find_similar_question(encoder_model: SentenceTransformer, question: str):
    answer = get_the_highest_question(
        engine=engine, encoder_model=encoder_model, question=question
    )


@routes.post("/qa/")
async def qa(request: web.Request) -> web.Response:
    """Возвращает ответ на вопрос пользователя и ссылку на источник

    Args:
        request (web.Request): запрос, содержащий `question`

    Returns:
        web.Response: ответ
    """

    question = (await request.json())["question"]
    # сюда обработку вопроса
    chunk = get_chunk(engine=engine, encoder_model=encoder_model, question=question)
    if chunk is None:
        return web.Response(text="Chunk not found", status=404)
    alt_stream = io.StringIO()
    with redirect_stderr(alt_stream):
        answer = get_answer(chunk.text, question)
    warnings = alt_stream.getvalue()
    if len(warnings) > 0:
        logging.warning(warnings)
    if "stopped" in warnings or "ответ не найден" in answer.lower():
        return web.Response(text="Answer not found", status=404)
    return web.json_response({"answer": answer, "confluence_url": chunk.confluence_url})


@routes.post("/generate_greeting/")
async def generate_greeting(request: web.Request) -> web.Response:
    """Генерация поздравления с использованием LLM.

    Args:
        request (web.Request): запрос, содержащий `template`, `user_name`, `holiday_name`

    Returns:
        web.Response: ответ с поздравлением
    """
    data = await request.json()
    template = data["template"]
    user_name = data["user_name"]
    holiday_name = data["holiday_name"]

    try:
        headers = Config.get_mistral_headers()
        prompt = Config.get_greeting_prompt(template, user_name, holiday_name)

        logging.info(f"Запрос к Mistral API: {prompt}")

        response = requests.post(Config.MISTRAL_API_URL, json=prompt, headers=headers)

        logging.info(f"Ответ Mistral API: {response.status_code} {response.text}")

        if response.status_code == 200:
            data = response.json()
            greeting = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            return web.json_response({"greeting": greeting})
        else:
            logging.error(f"Ошибка {response.status_code}: {response.text}")
            return web.Response(text="Ошибка генерации поздравления", status=500)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return web.Response(text="Ошибка генерации поздравления", status=500)


@routes.post("/reindex/")
async def reindex(request: web.Request) -> web.Response:
    """Пересоздаёт векторный индекс текстов для ответов на вопросы

    Args:
        request (web.Request): запрос

    Returns:
        web.Response: ответ
    """

    try:
        reindex_confluence(
            engine=engine, text_splitter=text_splitter, encoder_model=encoder_model
        )
        return web.Response(status=200)
    except Exception as e:
        return web.Response(text=str(e), status=500)


if __name__ == "__main__":
    with Session(engine) as session:
        questions = session.scalars(select(Chunk)).first()
        if questions is None:
            reindex_confluence(
                engine=engine, text_splitter=text_splitter, encoder_model=encoder_model
            )
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app)
