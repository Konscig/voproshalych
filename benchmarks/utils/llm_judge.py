"""Клиент LLM-судьи для оценки качества ответов.

Использует DeepSeek API с поддержкой retry логики.
"""

import logging
import os
import json
import re
from typing import Dict, List, Optional

from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)


def _parse_json_payload(text: str) -> Dict[str, str]:
    """Распарсить JSON-ответ модели с учётом markdown-обёрток."""
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


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
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Инициализировать LLM-судью.

        Args:
            api_key: API ключ (по умолчанию из BENCHMARKS_JUDGE_API_KEY)
            base_url: Базовый URL (по умолчанию из BENCHMARKS_JUDGE_BASE_URL или https://api.deepseek.com)
            model: Название модели (по умолчанию из BENCHMARKS_JUDGE_MODEL или deepseek-chat)

        Raises:
            ValueError: Если не удалось подключиться к API
        """
        candidates: List[tuple[str, str, str]] = []
        if api_key and base_url and model:
            candidates.append((api_key, base_url, model))
        candidates.extend(_get_judge_candidates())

        last_error: Optional[Exception] = None
        for candidate_api, candidate_base, candidate_model in candidates:
            self.model = candidate_model
            self.client = OpenAI(api_key=candidate_api, base_url=candidate_base)
            logger.info(
                "LLMJudge инициализация: model=%s, base_url=%s",
                candidate_model,
                candidate_base,
            )
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
        wait=wait_random_exponential(min=1, max=60),
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
        wait=wait_random_exponential(min=1, max=60),
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
- Стиль: живой студент пишет в чат поддержки (естественный русский язык).
- Разрешён умеренный сленг (например: препод, стипуха, пересдача), если уместно.
- Избегай канцелярита и шаблонов вроде "Каков регламент...".
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

            return result

        except Exception as e:
            error_msg = f"Ошибка генерации вопроса из чанка: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=60),
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
        prompt = """Оцени точность ответа по шкале от 1 до 5.

Критерии:
- 5: Ответ полностью соответствует контексту, нет галлюцинаций
- 4: Ответ соответствует контексту, есть незначительные неточности
- 3: Ответ частично соответствует контексту, есть некоторые неточности
- 2: Ответ слабо соответствует контексту, есть существенные неточности
- 1: Ответ не соответствует контексту, есть галлюцинации

Вопрос: {question}
Контекст: {context}
Ответ системы: {answer}

Ответ только числом от 1 до 5.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты эксперт по оценке качества ответов чат-ботов.",
                    },
                    {"role": "user", "content": prompt.format(**locals())},
                ],
                temperature=0.3,
                max_tokens=10,
            )

            score_str = (response.choices[0].message.content or "").strip()

            try:
                score = float(score_str)
                score = max(1.0, min(5.0, score))

                logger.info(f"Faithfulness оценка: {score}")
                return score

            except ValueError:
                raise ValueError(f"Некорректный формат оценки: {score_str}")

        except Exception as e:
            error_msg = f"Ошибка оценки faithfulness: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=60),
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
        prompt = """Оцени релевантность ответа на вопрос по шкале от 1 до 5.

Критерии:
- 5: Ответ полностью соответствует вопросу, отвечает по существу
- 4: Ответ соответствует вопросу, но есть мелкие неточности
- 3: Ответ частично соответствует вопросу
- 2: Ответ слабо соответствует вопросу
- 1: Ответ не соответствует вопросу

Вопрос: {question}
Ответ: {answer}

Ответ только числом от 1 до 5.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты эксперт по оценке релевантности ответов.",
                    },
                    {"role": "user", "content": prompt.format(**locals())},
                ],
                temperature=0.3,
                max_tokens=10,
            )

            score_str = (response.choices[0].message.content or "").strip()

            try:
                score = float(score_str)
                score = max(1.0, min(5.0, score))

                logger.info(f"Answer Relevance оценка: {score}")
                return score

            except ValueError:
                raise ValueError(f"Некорректный формат оценки: {score_str}")

        except Exception as e:
            error_msg = f"Ошибка оценки answer relevance: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=1, max=60),
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
- 5: Ответ системы близок к идеальному, полностью отвечает на вопрос
- 4: Ответ системы хорош, но есть отличия от идеального
- 3: Ответ системы приемлемый, но есть заметные отличия
- 2: Ответ системы слабый, существенно отличается от идеального
- 1: Ответ системы некорректный

Вопрос: {question}
Ответ системы: {system_answer}
Идеальный ответ: {ground_truth_answer}

Ответ только числом от 1 до 5.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Ты эксперт по оценке качества ответов чат-ботов.",
                    },
                    {"role": "user", "content": prompt.format(**locals())},
                ],
                temperature=0.3,
                max_tokens=10,
            )

            score_str = (response.choices[0].message.content or "").strip()

            try:
                score = float(score_str)
                score = max(1.0, min(5.0, score))

                logger.info(f"E2E Quality оценка: {score}")
                return score

            except ValueError:
                raise ValueError(f"Некорректный формат оценки: {score_str}")

        except Exception as e:
            error_msg = f"Ошибка оценки E2E качества: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
