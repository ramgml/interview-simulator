"""Интервьюер: промпты PLAN/TURN, STYLE_PROMPTS, state-machine хода (ARCHITECTURE §Интервьюер)."""

import json
import logging
from string import Template

from openai import OpenAI

from app.config import settings as env
from app.errors import InterviewError
from app.llm import json_chat
from app.models import Session

logger = logging.getLogger(__name__)

# Точные формулировки стилей — docs/ARCHITECTURE.md §Интервьюер (константы модуля, AGENTS.md).
STYLE_PROMPTS: dict[str, str] = {
    "friendly": "дружелюбный, поддерживающий",
    "strict": "сухой тон, давит follow-ups, вскрывает слабости, стресс-интервью",
    "realistic": "как живое собеседование в средней продуктовой компании",
}

PLAN_SYSTEM_TEMPLATE = Template(
    "Опытный технический интервьюер. По вакансии составь план интервью. Верни ТОЛЬКО JSON: "
    '{"position_title": str, "competencies": [str], "rounds": [{"type": "technical"|"behavioral"'
    '|"algorithms"|"system_design", "questions": [{"topic", "question", "competency"}]}]}. '
    "Раунды и вопросы зависят от вакансии; покрой ключевые требования; "
    "уровень кандидата: $seniority; язык: $language"
)

TURN_SYSTEM_TEMPLATE = Template(
    "Ты ведёшь собеседование один на один. Стиль: $style_prompt. "
    "Решение: ответ неполный — один follow-up по теме; тема исчерпана — следующий вопрос из "
    "плана (не заданный ранее); все темы покрыты — заверши. Верни ТОЛЬКО JSON: "
    '{"action": "followup"|"next_question"|"finish", "text": str, "covered_topic": str|null}'
)

# Полный транскрипт в TURN-промпт и авто-fin — до 24 ходов (ARCHITECTURE §Интервьюер/§API).
MAX_TURNS = 24

# Вакансия в PLAN-промпте обрезается до 8000 символов (ARCHITECTURE §Интервьюер).
VACANCY_MAX_CHARS = 8000

FINISH_FALLBACK_TEXT = "Спасибо, интервью завершено"


def _plan(session: Session) -> dict:
    """plan_json сессии → dict; пустой/битый план → пустой dict (fallback: незаданных нет)."""
    if not session.plan_json:
        return {}
    try:
        return json.loads(session.plan_json)
    except ValueError:
        logger.warning("plan_json is not valid JSON for session %s", session.id)
        return {}


def build_plan(client: OpenAI, session: Session, model: str) -> dict:
    """PLAN: вакансия → план интервью через json_chat. Ошибка LLM — наружу как есть."""
    system = PLAN_SYSTEM_TEMPLATE.substitute(
        seniority=session.seniority or "не указан", language=session.language
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": session.vacancy_text[:VACANCY_MAX_CHARS]},
    ]
    return json_chat(client, model, messages, temperature=0.2, max_tokens=env.plan_max_tokens)


def first_question(plan: dict) -> str | None:
    """Первый вопрос плана (порядок раундов) или None, если план без вопросов."""
    for round_ in plan.get("rounds", []):
        for question in round_.get("questions", []):
            text = question.get("question")
            if text:
                return text
    return None


def _unasked_question(plan: dict, transcript_turns: list[dict]) -> str | None:
    """Первый незаданный вопрос плана; заданные — тексты interviewer-ходов транскрипта."""
    asked = {t.get("text") for t in transcript_turns if t.get("role") == "interviewer"}
    for round_ in plan.get("rounds", []):
        for question in round_.get("questions", []):
            text = question.get("question")
            if text and text not in asked:
                return text
    return None


def conduct_turn(
    client: OpenAI, session: Session, transcript_turns: list[dict], model: str
) -> dict:
    """TURN: план+транскрипт (до 24 ходов) → решение followup/next_question/finish.

    Деградация: InterviewError от LLM → следующий незаданный вопрос из plan_json;
    незаданных нет → action=finish. Сессия не падает.
    """
    style_prompt = STYLE_PROMPTS.get(session.style, STYLE_PROMPTS["realistic"])
    system = TURN_SYSTEM_TEMPLATE.substitute(style_prompt=style_prompt)
    messages = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(
                {"plan": _plan(session), "transcript": transcript_turns[:MAX_TURNS]},
                ensure_ascii=False,
            ),
        },
    ]
    try:
        return json_chat(client, model, messages, temperature=0.2, max_tokens=env.turn_max_tokens)
    except InterviewError:
        logger.warning("conduct_turn fallback for session %s", session.id)
        question = _unasked_question(_plan(session), transcript_turns)
        if question is None:
            return {"action": "finish", "text": FINISH_FALLBACK_TEXT, "covered_topic": None}
        return {"action": "next_question", "text": question, "covered_topic": None}
