"""Клиент LLM-судьи для оценки качества ответов.

Использует DeepSeek API с поддержкой retry логики.
"""

import logging
import os
import json
import re
import time
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)


def _parse_json_payload(text: str) -> Dict[str, Any]:
    """Распарсить JSON-ответ модели или plain number с учётом markdown и экранированных скобок."""
    cleaned = text.strip()

    if not cleaned:
        logger.warning("Пустой ответ от модели при парсинге JSON")
        return {"score": 3.0}

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    cleaned = cleaned.strip()
    if not cleaned:
        logger.warning("Пустой ответ после очистки markdown")
        return {"score": 3.0}

    # Убрать экранированные двойные скобки {{ }} -> { }
    cleaned = cleaned.replace("{{", "{").replace("}}", "}")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"Невалидный JSON: {e}, text: {cleaned[:200]}")
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        plain_match = re.search(r"\d+", cleaned)
        if plain_match:
            score = int(plain_match.group(0))
            if 1 <= score <= 5:
                logger.info(f"Извлечено plain number: {score}")
                return {"score": float(score)}

        logger.warning("Не найден JSON объект или число в ответе")
        return {"score": 3.0}


def _get_judge_candidates() -> List[tuple[str, str, str]]:
    """Собрать кандидатов подключения к judge API из окружения."""
    candidates: List[tuple[str, str, str]] = []

    benchmarks_api = os.getenv("BENCHMARKS_JUDGE_API_KEY")
    benchmarks_base = os.getenv("BENCHMARKS_JUDGE_BASE_URL", "https://api.deepseek.com")
    benchmarks_model = os.getenv("BENCHMARKS_JUDGE_MODEL", "deepseek-chat")
    if benchmarks_api:
        candidates.append((benchmarks_api, benchmarks_base, benchmarks_model))

    project_api = os.getenv("JUDGE_API")
    project_base = os.getenv("JUDGE_BASE_URL", "https://api.mistral.ai/v1")
    project_model = os.getenv("JUDGE_MODEL", "mistral-small-latest")
    if project_api:
        candidates.append((project_api, project_base, project_model))

    if not candidates:
        raise ValueError(
            "Не задан ключ judge API. Укажите BENCHMARKS_JUDGE_API_KEY "
            "или JUDGE_API в окружении."
        )

    unique_candidates = []
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)
    return unique_candidates


class LLMJudge:
    """LLM-судья для оценки качества ответов.

    Attributes:
        client: OpenAI клиент для взаимодействия с API
        model: Название модели для использования
        request_delay: Задержка между запросами в секундах
        request_timeout: Таймаут запроса в секундах
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        request_delay: float = 2.0,
        request_timeout: float = 120.0,
        evaluation_mode: str = "direct",
    ):
        """Инициализировать LLM-судью.

        Args:
            api_key: API ключ (по умолчанию из BENCHMARKS_JUDGE_API_KEY)
            base_url: Базовый URL (по умолчанию из BENCHMARKS_JUDGE_BASE_URL или https://api.deepseek.com)
            model: Название модели (по умолчанию из BENCHMARKS_JUDGE_MODEL или deepseek-chat)
            request_delay: Задержка между запросами в секундах (по умолчанию 2.0)
            request_timeout: Таймаут запроса в секундах (по умолчанию 120.0)

        Raises:
            ValueError: Если не удалось подключиться к API
        """
        if evaluation_mode not in {"direct", "reasoned"}:
            raise ValueError("evaluation_mode должен быть direct или reasoned")
        self.evaluation_mode = evaluation_mode

        candidates: List[tuple[str, str, str]] = []
        if api_key and base_url and model:
            candidates.append((api_key, base_url, model))
        candidates.extend(_get_judge_candidates())

        last_error: Optional[Exception] = None
        for candidate_api, candidate_base, candidate_model in candidates:
            self.model = candidate_model
            self.client = OpenAI(
                api_key=candidate_api, base_url=candidate_base, timeout=request_timeout
            )
            self.request_delay = request_delay
            logger.info(
                "LLMJudge инициализация: model=%s, base_url=%s, timeout=%s",
                candidate_model,
                candidate_base,
                request_timeout,
            )
            time.sleep(1.0)
            try:
                self._check_connection()
                return
            except Exception as error:
                last_error = error
                logger.warning(
                    "Не удалось подключиться к judge API (%s, %s), пробуем следующий источник.",
                    candidate_model,
                    candidate_base,
                )

        raise ConnectionError(f"Не удалось инициализировать LLMJudge: {last_error}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=5, max=120),
        reraise=True,
    )
    def _check_connection(self):
        """Проверить подключение к API.

        Raises:
            ConnectionError: Если не удалось подключиться
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=1,
                timeout=30,
            )

            if response.choices and len(response.choices) > 0:
                logger.info("✅ Подключение к API успешно проверено")
            else:
                raise ConnectionError("API вернул пустой ответ")

        except Exception as e:
            error_msg = f"Ошибка подключения к API: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=5, max=120),
        reraise=True,
    )
    def generate_question_from_chunk(self, chunk_text: str) -> Dict[str, str]:
        """Сгенерировать вопрос и идеальный ответ из текста чанка.

        Args:
            chunk_text: Текст чанка

        Returns:
            Словарь с question и ground_truth_answer

        Raises:
            RuntimeError: Если не удалось сгенерировать вопрос
        """
        prompt = """Тебе показали фрагмент внутренней документации университета.

Сгенерируй ОДИН осмысленный вопрос в поддержку и краткий идеальный ответ.

Требования к вопросу:
- Вопрос обязан напрямую опираться на факты из этого конкретного чанка.
- Вопрос должен быть понятен без дополнительного контекста.
- Стиль: живой студент ТюмГУ пишет в чат поддержки (естественный русский язык).
- Используй неформальные обороты: "чё это", "как так-то", "чё за", "блин", "халява", "препод", "стипуха", "пересдача", "отработка", "вышка", "сессия", "зачёт", "хвосты".
- Можно использовать названия факультетов/институтов: ШКН, ФЭИ, СоцГУ, ИПиП, Биофак, ИНЖИН, ИНХИМ, ФТИ, ИНБИО, ИФК, ЕГШ, ШПИ, ШЕН, УИОТ, Гуманитарный институт и т.д.
- Можно использовать студенческий сленг: "когда сессия", "когда пересдача", "чё сдавать", "как закрыть хвост", "дадут ли стипуху", "когда вывесят расписание", "чё с оценками", "можно ли отработать", "как перевестись", "чё делать если...".
- Избегай канцелярита и шаблонов вроде "Каков регламент...", "Просьба разъяснить...", "Каким образом...".
- Не придумывай факты, которых нет в тексте чанка.

Требования к ответу:
- Кратко и точно.
- Только по информации из чанка.

Текст чанка:
{chunk_text}

Верни ТОЛЬКО JSON:
{{"question": "...", "ground_truth_answer": "..."}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ты формируешь synthetic dataset для RAG-бенчмарков. "
                            "Каждый вопрос должен быть привязан к конкретному чанку, "
                            "звучать как живой вопрос студента и оставаться смысловым. "
                            "Запрещено использовать канцелярские формулировки и выдумывать факты."
                        ),
                    },
                    {"role": "user", "content": prompt.format(chunk_text=chunk_text)},
                ],
                temperature=0.7,
                max_tokens=500,
            )

            content = (response.choices[0].message.content or "").strip()
            result = _parse_json_payload(content)

            if "question" not in result or "ground_truth_answer" not in result:
                raise ValueError("Некорректный формат ответа от API")

            logger.info(f"Сгенерирован вопрос: {result['question'][:50]}...")

            time.sleep(self.request_delay)

            return result

        except Exception as e:
            error_msg = f"Ошибка генерации вопроса из чанка: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=5, max=120),
        reraise=True,
    )
    def evaluate_faithfulness(
        self,
        question: str,
        context: str,
        answer: str,
    ) -> float:
        """Оценить точность ответа (faithfulness).

        Проверяет, соответствует ли ответ контексту и нет ли галлюцинаций.

        Args:
            question: Вопрос пользователя
            context: Контекст (текст чанка)
            answer: Ответ системы

        Returns:
            Оценка от 1 до 5

        Raises:
            RuntimeError: Если не удалось получить оценку
        """
        prompt = f"""Оцени точность ответа по шкале от 1 до 5.

Критерии:
- 5: Ответ полностью соответствует контексту, нет галлюцинаций
- 4: Ответ соответствует контексту, есть незначительные неточности
- 3: Ответ частично соответствует контексту, есть некоторые неточности
- 2: Ответ слабо соответствует контексту, есть существенные неточности
- 1: Ответ не соответствует контексту, есть галлюцинации

Вопрос: {question}
Контекст: {context}
Ответ системы: {answer}

Верни ТОЛЬКО JSON в формате: {{"score": 1}}
Где score — число от 1 до 5.
"""

        if self.evaluation_mode == "reasoned":
            prompt += (
                "\nСначала рассуждай пошагово про себя, но в ответе верни только "
                "JSON с оценкой."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты эксперт по оценке качества ответов чат-ботов.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )

            content = (response.choices[0].message.content or "").strip()
            payload = _parse_json_payload(content)

            score = float(payload.get("score", 3.0))
            score = max(1.0, min(5.0, score))

            logger.info(f"Faithfulness оценка: {score}")

            time.sleep(self.request_delay)

            return score

        except Exception as e:
            error_msg = f"Ошибка оценки faithfulness: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=5, max=120),
        reraise=True,
    )
    def evaluate_answer_relevance(
        self,
        question: str,
        answer: str,
    ) -> float:
        """Оценить релевантность ответа.

        Проверяет, отвечает ли система на вопрос по существу.

        Args:
            question: Вопрос пользователя
            answer: Ответ системы

        Returns:
            Оценка от 1 до 5

        Raises:
            RuntimeError: Если не удалось получить оценку
        """
        prompt = f"""Оцени релевантность ответа на вопрос по шкале от 1 до 5.

Критерии:
- 5: Ответ полностью соответствует вопросу, отвечает по существу
- 4: Ответ соответствует вопросу, но есть мелкие неточности
- 3: Ответ частично соответствует вопросу
- 2: Ответ слабо соответствует вопросу
- 1: Ответ не соответствует вопросу

Вопрос: {question}
Ответ: {answer}

Верни ТОЛЬКО JSON в формате: {{"score": 1}}
Где score — число от 1 до 5.
"""

        if self.evaluation_mode == "reasoned":
            prompt += (
                "\nСначала рассуждай пошагово про себя, но в ответе верни только "
                "JSON с оценкой."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты эксперт по оценке качества ответов чат-ботов.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )

            content = (response.choices[0].message.content or "").strip()
            payload = _parse_json_payload(content)

            score = float(payload.get("score", 3.0))
            score = max(1.0, min(5.0, score))

            logger.info(f"Answer Relevance оценка: {score}")

            time.sleep(self.request_delay)

            return score

        except Exception as e:
            error_msg = f"Ошибка оценки answer relevance: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=5, max=120),
        reraise=True,
    )
    def evaluate_e2e_quality(
        self,
        question: str,
        system_answer: str,
        ground_truth_answer: str,
    ) -> float:
        """Оценить качество ответа end-to-end.

        Сравнивает ответ системы с идеальным ответом.

        Args:
            question: Вопрос пользователя
            system_answer: Ответ системы
            ground_truth_answer: Идеальный ответ

        Returns:
            Оценка от 1 до 5

        Raises:
            RuntimeError: Если не удалось получить оценку
        """
        prompt = """Оцени качество ответа системы по сравнению с идеальным ответом по шкале от 1 до 5.

Критерии:
- 5: Ответ систем близок к идеальному, полностью отвечает на вопрос
- 4: Ответ системы хорош, но есть отличия от идеального
- 3: Ответ системы приемлемый, но есть заметные отличия
- 2: Ответ системы слабый, существенно отличается от идеального
- 1: Ответ системы некорректный

Вопрос: {question}
Ответ системы: {system_answer}
Идеальный ответ: {ground_truth_answer}

Верни ТОЛЬКО JSON в формате:
{{"score": 1}}
Где score — число от 1 до 5.
"""

        if self.evaluation_mode == "reasoned":
            prompt += (
                "\nСначала рассуждай пошагово про себя, но в ответе верни только "
                "JSON с оценкой."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты эксперт по оценке качества ответов чат-ботов.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )

            content = (response.choices[0].message.content or "").strip()
            payload = _parse_json_payload(content)

            score = float(payload.get("score", 3.0))
            score = max(1.0, min(5.0, score))

            logger.info(f"E2E Quality оценка: {score}")

            time.sleep(self.request_delay)

            return score

        except Exception as e:
            error_msg = f"Ошибка оценки E2E качества: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=5, max=120),
        reraise=True,
    )
    def evaluate_answer_correctness(
        self,
        question: str,
        system_answer: str,
        ground_truth_answer: str,
    ) -> float:
        """Оценить корректность ответа относительно эталона по шкале 1-5."""
        prompt = f"""Оцени корректность ответа системы по отношению к эталонному ответу.

Критерии:
- 5: ответ полностью корректен и покрывает ключевые факты эталона
- 4: в целом корректен, есть небольшие пропуски
- 3: частично корректен, значимые пробелы
- 2: много неточностей или искажений
- 1: в основном некорректен

Вопрос: {question}
Ответ системы: {system_answer}
Эталонный ответ: {ground_truth_answer}

Верни ТОЛЬКО JSON в формате: {{"score": 1}}
Где score — число от 1 до 5.
"""
        if self.evaluation_mode == "reasoned":
            prompt += (
                "\nСначала рассуждай пошагово про себя, но в ответе верни только "
                "JSON с оценкой."
            )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты эксперт по проверке фактической корректности ответов.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=200,
            )

            content = (response.choices[0].message.content or "").strip()
            payload = _parse_json_payload(content)

            score = float(payload.get("score", 3.0))
            score = max(1.0, min(5.0, score))

            logger.info(f"Answer Correctness оценка: {score}")

            time.sleep(self.request_delay)
            return score

        except Exception as e:
            error_msg = f"Ошибка оценки answer correctness: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=5, max=120),
        reraise=True,
    )
    def compare_answers_pairwise(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        allow_tie: bool = True,
    ) -> Dict[str, str]:
        """Сравнить два ответа и вернуть победителя (A/B/Tie)."""
        tie_option = "Tie" if allow_tie else "A or B"
        prompt = f"""Сравни два ответа на один вопрос и выбери лучший.

Вопрос: {question}
Ответ A: {answer_a}
Ответ B: {answer_b}

Критерии: релевантность, корректность, полезность.
Разрешённый формат ответа: {tie_option}.
Если ответы равноценны, выбери Tie.

Верни строго JSON вида:
{{"winner": "A|B|Tie", "reason": "краткое объяснение"}}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "Ты сравниваешь ответы для pairwise-evaluation.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=200,
        )

        content = (response.choices[0].message.content or "").strip()
        payload = _parse_json_payload(content)
        winner = str(payload.get("winner", "Tie"))
        if winner not in {"A", "B", "Tie"}:
            winner = "Tie"
        if not allow_tie and winner == "Tie":
            winner = "A"
        return {
            "winner": winner,
            "reason": str(payload.get("reason", "")),
        }
