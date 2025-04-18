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
from database import (
    Chunk,
    set_embedding,
    get_answer_by_id,
    get_all_questions_with_score,
    get_document_by_url,
)
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
    """Отправляет запрос к LLM и возвращает ответ.

    Args:
        context (str): фрагмент документа, содержащий ответ на вопрос
        question (str): текст вопроса

    Returns:
        str: ответ на вопрос
    """
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


def assess_answer(
    question: str, answer: str, content: str = "", generation: bool = False
) -> bool:
    """Оценивает релевантность подобранного ответа на вопрос с использованием LLM модели-судьи.

    Args:
        question (str): текст вопроса
        answer (str): текст ответа
    """
    try:
        headers = Config.get_judge_headers()
        prompt = Config.get_judge_prompt(
            question_text=question,
            answer_text=answer,
            content=content,
            generation=generation,
        )
        response = requests.post(Config.MISTRAL_API_URL, json=prompt, headers=headers)
        if response.status_code == 200:
            data = response.json()
            solution = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                .lower()
            )
            logging.info(f"Solution is {solution}")
            if solution == "yes":
                return True
            else:
                return False
        else:
            logging.error(f"Ошибка {response.status_code}: {response.text}")
            return False
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return False


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
    logging.info(f"Answer text: {answer_text}")

    if not answer_text:
        logging.error(f"Answer not found for QA ID: {qa_id}")
        return web.json_response({"error": "Answer not found"}, status=404)

    embedding = encoder_model.encode(answer_text)

    # Логируем векторное представление ответа
    logging.info(f"Embedding for QA ID {qa_id}: {embedding}")

    set_embedding(engine=engine, question_answer_id=qa_id, embed=embedding)
    logging.info(f"Embedding set successfully for QA ID: {qa_id}")
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

    questions = get_all_questions_with_score(engine=engine)
    question_embedding = encoder_model.encode(question)

    # Логируем векторное представление вопроса
    logging.info(f"Question: '{question}' => Embedding: {question_embedding}")

    best_match = None
    best_cosine_distance = float("inf")
    best_qa_id = None
    best_qa_url = None

    for q in questions:
        qa_id, answer_text, embedding, url = (
            q["id"],
            q["answer"],
            q["embedding"],
            q["url"],
        )

        if embedding is None or len(embedding) == 0:
            logging.warning(f"Skipping empty embedding for QA ID {qa_id}")
            continue

        # Проверка на размерности векторов
        if question_embedding.shape[0] != embedding.shape[0]:
            logging.error(
                f"Shape mismatch for QA ID {qa_id}: question {question_embedding.shape}, embedding {embedding.shape}"
            )
            continue

        # Косинусное расстояние
        cosine_distance = 1 - np.dot(question_embedding, embedding) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(embedding)
        )

        # Логируем совпадение и расстояние
        logging.info(f"Cosine distance for QA ID {qa_id}: {cosine_distance}")

        if cosine_distance < best_cosine_distance:
            best_cosine_distance = cosine_distance
            best_match = answer_text
            best_qa_id = int(qa_id)
            best_qa_url = url

    if best_cosine_distance <= 0.6 and best_match and best_qa_id and best_qa_url:
        logging.info(
            f"Best match found for QA ID {best_qa_id} with cosine distance {best_cosine_distance}"
        )
        return best_match, best_qa_url

    logging.info(
        f"No suitable match found for the question with cosine distance {best_cosine_distance}"
    )
    return None


@routes.post("/qa/")
async def qa(request: web.Request) -> web.Response:
    """Возвращает ответ на вопрос пользователя и ссылку на источник

    Args:
        request (web.Request): запрос

    Returns:
        web.Response: ответ
    """
    data = await request.json()
    question = data.get("question")

    logging.info(f"Received question: '{question}'")

    result = await find_similar_question(encoder_model=encoder_model, question=question)
    if result:
        db_answer, url = result
        logging.info(f"Answer found in database for question '{question}'")
        verdict = assess_answer(question=question, answer=db_answer)
        if verdict:
            return web.json_response({"answer": db_answer, "confluence_url": url})
        else:
            logging.error(f"Answer {db_answer} were banned for question {question}")

    logging.info(
        f"No match found in database for question '{question}', processing with LLM."
    )

    chunk = get_chunk(engine=engine, encoder_model=encoder_model, question=question)
    if chunk is None:
        logging.error(f"Chunk not found for question: {question}")
        return web.Response(text="Chunk not found", status=404)

    alt_stream = io.StringIO()
    with redirect_stderr(alt_stream):
        answer = get_answer(chunk.text, question)

    warnings = alt_stream.getvalue()

    verdict = assess_answer(
        question=question, answer=answer, content=chunk.text, generation=True
    )
    if verdict:
        logging.info(f"Answer generated by LLM for question '{question}': {answer}")
        return web.json_response(
            {"answer": answer, "confluence_url": chunk.confluence_url}
        )

    if not verdict:
        logging.error(f"Answer {answer} were banned for question {question}")
        return web.Response(text="Answer banned", status=404)

    if warnings:
        logging.warning(warnings)

    if "stopped" in warnings or "ответ не найден" in answer.lower():
        logging.warning(f"Answer not found for question: '{question}'")
        return web.Response(text="Answer not found", status=404)


@routes.post("/generate_greeting/")
async def generate_greeting(request: web.Request) -> web.Response:
    """Генерирует поздравления с использованием LLM.

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


@routes.post("/check-score/")
async def check_score(request: web.Request) -> web.Response:
    """_summary_

    Args:
        request (web.Request): _description_

    Returns:
        web.Response: _description_
    """
    try:
        questions = get_all_questions_with_score(engine, highscore=False)
        for question in questions:
            q_text = question["question"]
            q_answer = question["answer"]
            q_content = get_document_by_url(engine, question["url"])
            if q_content is None:
                q_content = ""

            logging.info(f"Question: {q_text}")
            logging.info(f"Answer: {q_answer}")
            logging.info(f"Content: {q_content[:50]}")
        headers = Config.get_judge_headers()
        prompt = Config.get_judge_prompt(
            question_text=q_text, answer_text=q_answer, content=q_content, scorer=True
        )
        response = requests.post(Config.MISTRAL_API_URL, json=prompt, headers=headers)
        if response.status_code == 200:
            data = response.json()
            solution = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                .lower()
            )
            logging.info(f"Solution is {solution}")
            if solution == "yes":
                return True
    except Exception as e:
        return web.Response(text=str(e), status=500)


@routes.post("/reembed/")
async def reembed(request: web.Request) -> web.Response:
    """Пересоздаёт векторные представления для вопросов

    Args:
        request (web.Request): запрос на пересоздание векторных представлений

    Returns:
        web.Response: ответ
    """
    logging.warning("START CREATE EMBEDDING")
    try:
        questions = get_all_questions_with_score(engine)
        for question in questions:
            q_text = question["question"]
            q_id = question["id"]
            embed = encoder_model.encode(q_text)
            set_embedding(engine, q_id, embed)
        logging.warning("SUCCESSFULLY CREATED EMBEDDING")
        return web.Response(status=200)
    except Exception as e:
        logging.error("FAILED TO CREATED EMBEDDING, ERROR: " + str(e))
        return web.Response(text=str(e), status=500)


if __name__ == "__main__":
    with Session(engine) as session:
        questions = session.scalars(select(Chunk)).first()
        if questions is None:
            reindex_confluence(
                engine=engine, text_splitter=text_splitter, encoder_model=encoder_model
            )

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],  # Логи выводятся в консоль
    )
    app = web.Application()
    app.add_routes(routes)
    web.run_app(app)
