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
from database import Chunk, set_embedding, get_answer_by_id, get_all_questions_with_high
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
async def set_question_embedding(request: web.Request):
    """Задает векторное представление вопросу

    Args:
        request (web.Request): запрос, содержащий текст вопроса

    Returns:
        web.Response: результат векторизации
    """
    qa_id = (await request.json())["answer_id"]
    answer_text = get_answer_by_id(engine=engine, question_answer_id=qa_id)
    if not answer_text:
        return web.json_response({"error": "Answer not found"}, status=404)
    embedding = encoder_model.encode(answer_text)
    set_embedding(engine=engine, question_answer_id=qa_id, embed=embedding)
    return web.json_response({"message": "Embedding set successfully"}, status=200)


async def find_similar_question(
    encoder_model: SentenceTransformer, question: str
) -> tuple[str, str] | None:
    """Ищет самый похожий вопрос с оценкой 5 и возвращает его ответ и URL источника

    Args:
        encoder_model (SentenceTransformer): модель получения векторных представлений SentenceTransformer
        question (str): текст заданного вопроса

    Returns:
        tuple[str, str] | None: кортеж, содержащий наиболее близкий ответ и ссылку на документ в Confluence, иначе None.
    """

    questions = get_all_questions_with_high(engine=engine)
    question_embedding = encoder_model.encode(question)

    best_match = None
    best_cosine_distance = float("inf")
    best_qa_id = None

    for q in questions:
        qa_id, answer_text, embedding, url = (
            q["id"],
            q["answer"],
            q["embedding"],
            q["url"],
        )

        if embedding is None:
            continue

        cosine_distance = 1 - np.dot(question_embedding, embedding) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(embedding)
        )

        if cosine_distance < best_cosine_distance:
            best_cosine_distance = cosine_distance
            best_match = answer_text
            best_qa_id = int(qa_id)
            best_qa_url = url
    if best_cosine_distance < 0.8 and best_match and best_qa_id and best_qa_url:
        url = best_qa_url
        return best_match, url or ""

    return None


@routes.post("/qa/")
async def qa(request: web.Request) -> web.Response:
    """Возвращает ответ на вопрос пользователя и ссылку на источник"""
    data = await request.json()
    question = data.get("question")
    result = await find_similar_question(encoder_model=encoder_model, question=question)
    if result:
        db_answer, url = result
        return web.json_response({"answer": db_answer, "confluence_url": url})
    chunk = get_chunk(engine=engine, encoder_model=encoder_model, question=question)
    if chunk is None:
        return web.Response(text="Chunk not found", status=404)
    alt_stream = io.StringIO()
    with redirect_stderr(alt_stream):
        answer = get_answer(chunk.text, question)
    warnings = alt_stream.getvalue()
    if warnings:
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
