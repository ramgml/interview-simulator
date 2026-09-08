"""Оценщик: промпт EVAL, финальный отчёт по схеме, degraded-fallback (ARCHITECTURE §Оценщик)."""

import json
import logging

from openai import OpenAI
from sqlalchemy import select

from app import interviewer
from app.config import settings as env
from app.db import SessionFactory
from app.errors import InterviewError
from app.llm import json_chat
from app.models import Session, Turn

logger = logging.getLogger(__name__)

EVAL_SYSTEM_PROMPT = (
    "Опытный технический интервьюер. Оцени прошедшее интервью по плану и полному транскрипту. "
    "Верни ТОЛЬКО JSON по схеме: "
    '{"overall_score": 0-10, '
    '"competencies": [{"name": str, "score": 0-10, "comment": str}], '
    '"turn_feedback": [{"turn_idx": int, "question": str, "answer": str, "score": 0-10, '
    '"good": str, "missed": str, "strong_answer": str}], '
    '"strengths": [str], "weaknesses": [str], '
    '"plan": [{"topic": str, "action": str, "resources_hint": str}], '
    '"verdict": str, '
    '"hire_recommendation": "strong_yes"|"yes"|"no"|"strong_no"}. '
    "overall_score — итоговая оценка кандидата; turn_feedback — разбор каждого ответа "
    "кандидата (turn_idx — из транскрипта); plan — план подготовки по слабым местам. "
    "Оценки честные и обоснованные, тексты — на русском"
)

# Нарратив degraded-отчёта: двойная ошибка JSON — текст вместо структурированного разбора.
DEGRADED_VERDICT = (
    "Автоматический разбор интервью недоступен: модель вернула ответ, который не удалось "
    "преобразовать в отчёт. Интервью завершено и сохранено — транскрипт доступен в сессии, "
    "повторите финальную оценку позже или разберите ответы вручную."
)


def degraded_report() -> dict:
    """Отчёт-деградация: нарратив в verdict, пустые списки, null-баллы (ARCHITECTURE §Оценщик)."""
    return {
        "degraded": True,
        "overall_score": None,
        "competencies": [],
        "turn_feedback": [],
        "strengths": [],
        "weaknesses": [],
        "plan": [],
        "verdict": DEGRADED_VERDICT,
        "hire_recommendation": None,
    }


def _transcript(session: Session) -> list[dict]:
    """Транскрипт сессии из БД (turns по idx) — как в tracing, через SessionFactory."""
    stmt = select(Turn).where(Turn.session_id == session.id).order_by(Turn.idx)
    with SessionFactory() as db:
        rows = db.execute(stmt).scalars().all()
    return [{"turn_idx": t.idx, "role": t.role, "text": t.text} for t in rows]


def evaluate(client: OpenAI, session: Session, model: str) -> dict:
    """EVAL: план+транскрипт → отчёт по схеме через json_chat.

    Деградация: InterviewError (двойная ошибка JSON/недоступность LLM) → degraded-отчёт,
    сессия всё равно завершается отчётом (ARCHITECTURE §Оценщик).
    """
    messages = [
        {"role": "system", "content": EVAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {"plan": interviewer._plan(session), "transcript": _transcript(session)},
                ensure_ascii=False,
            ),
        },
    ]
    try:
        return json_chat(client, model, messages, temperature=0.2, max_tokens=env.eval_max_tokens)
    except InterviewError:
        logger.warning("evaluate degraded fallback for session %s", session.id)
        return degraded_report()
