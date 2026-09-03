"""Роутер сессий: /api/sessions — create/start/answer/finish, GET-ы (ARCHITECTURE §API)."""

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession
from app import interviewer, llm, stt, tracing
from app.db import get_db
from app.errors import EmptyTranscript, InterviewError
from app.interviewer import MAX_TURNS
from app.models import Session, Settings, Turn
from app.schemas import AnswerOut, SessionCreate, SessionCreated

logger = logging.getLogger(__name__)

router = APIRouter()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@router.post("/api/sessions", response_model=SessionCreated)
def create_session(req: SessionCreate, db: DbSession = Depends(get_db)) -> SessionCreated:
    """Создать сессию: {vacancy_text, ...} → {id}; status=created, план позже (start)."""
    row = Session(
        vacancy_text=req.vacancy_text,
        seniority=req.seniority,
        language=req.language,
        style=req.style,
        planned_questions=req.planned_questions,
    )
    db.add(row)
    db.commit()
    return SessionCreated(id=row.id)


@router.post("/api/sessions/{session_id}/start")
def start_session(session_id: str, db: DbSession = Depends(get_db)) -> dict:
    """PLAN: build_plan → plan_json; первый вопрос → turns; status=in_progress, started_at."""
    session = _get_session(db, session_id)
    if session.status != "created":
        raise _http_error(409, f"Сессия уже начата (status={session.status})")
    try:
        client = llm.get_client(_settings(db))
        plan = interviewer.build_plan(client, session)
    except InterviewError as exc:
        _fail(db, session, exc)
        raise _http_error(502, str(exc)) from exc
    session.plan_json = json.dumps(plan, ensure_ascii=False)
    session.position_title = str(plan.get("position_title") or session.position_title or "")
    question = interviewer.first_question(plan)
    if question:
        _add_turn(db, session, role="interviewer", text=question)
    session.status = "in_progress"
    session.started_at = utcnow()
    db.commit()
    return _session_state(db, session)


@router.post("/api/sessions/{session_id}/answer", response_model=AnswerOut)
async def answer_session(
    session_id: str,
    request: Request,
    text: str | None = Form(None),
    audio: UploadFile | None = File(None),
    db: DbSession = Depends(get_db),
) -> AnswerOut:
    """Ход кандидата: multipart audio (STT) ИЛИ json {text}; далее TURN-решение интервьюера.

    Пустой STT → 422 «Речь не распознана», LLM не дёргается. Авто-fin при 24 ходах.
    """
    session = _get_session(db, session_id)
    if session.status != "in_progress":
        raise _http_error(409, f"Сессия не идёт (status={session.status})")
    if text is None and audio is None:
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("application/json"):
            body = await request.json()
            text = body.get("text") if isinstance(body, dict) else None
    transcript: str
    confidence: float | None = None
    if audio is not None:
        data = await audio.read()
        position_title = _position_title(session)
        try:
            transcript, confidence = await asyncio.get_running_loop().run_in_executor(
                None, stt.transcribe, data, session.language, position_title
            )
        except EmptyTranscript:
            raise HTTPException(status_code=422, detail="Речь не распознана") from None
        except InterviewError as exc:
            _fail(db, session, exc)
            raise _http_error(502, str(exc)) from exc
    else:
        if not text or not text.strip():
            raise _http_error(422, "Пустой текст ответа")
        transcript = text.strip()

    _add_turn(db, session, role="candidate", text=transcript, stt_confidence=confidence)
    turns = _turns_of(db, session)
    try:
        client = llm.get_client(_settings(db))
        decision = interviewer.conduct_turn(client, session, _transcript(turns))
        trace_id = tracing.get_last_trace_id()
    except InterviewError as exc:
        _fail(db, session, exc)
        raise _http_error(502, str(exc)) from exc

    action = decision.get("action", "next_question")
    done = action == "finish"
    if len(turns) >= MAX_TURNS:
        action = "finish"
        done = True
        decision["text"] = interviewer.FINISH_FALLBACK_TEXT
    _add_turn(
        db,
        session,
        role="interviewer",
        text=decision.get("text") or "",
        llm_trace_id=trace_id,
    )
    db.commit()

    if done:
        _complete(db, session)
    return AnswerOut(
        transcript=transcript,
        question_text=decision.get("text") if not done else None,
        done=done,
        action=action,
    )


@router.post("/api/sessions/{session_id}/finish")
def finish_session(session_id: str, db: DbSession = Depends(get_db)) -> dict:
    """Досрочный finish: status=completed + duration_sec (без отчёта)."""
    session = _get_session(db, session_id)
    if session.status not in ("in_progress", "created"):
        raise _http_error(409, f"Сессия не идёт (status={session.status})")
    _complete(db, session)
    return _session_state(db, session)


@router.get("/api/sessions")
def list_sessions(db: DbSession = Depends(get_db)) -> list[dict]:
    """История: список сессий (без turns)."""
    rows = db.execute(select(Session).order_by(Session.created_at.desc())).scalars().all()
    return [_session_brief(row) for row in rows]


@router.get("/api/sessions/{session_id}")
def get_session(session_id: str, db: DbSession = Depends(get_db)) -> dict:
    """Состояние сессии + turns."""
    session = _get_session(db, session_id)
    return _session_state(db, session)


@router.get("/api/sessions/{session_id}/report")
def get_report(session_id: str, db: DbSession = Depends(get_db)) -> dict:
    """report_json сессии; отчёта ещё нет → 404."""
    session = _get_session(db, session_id)
    if not session.report_json:
        raise _http_error(404, "Отчёт ещё не готов")
    return json.loads(session.report_json)


# --- helpers --------------------------------------------------------------------


def _get_session(db: DbSession, session_id: str) -> Session:
    row = db.get(Session, session_id)
    if row is None:
        raise _http_error(404, "Сессия не найдена")
    return row


def _settings(db: DbSession) -> Settings:
    """Singleton-строка настроек (id=1) для get_client."""
    return db.get(Settings, 1)


def _position_title(session: Session) -> str:
    """Заголовок позиции для STT initial_prompt: из плана, иначе пустая строка."""
    return str(interviewer._plan(session).get("position_title") or "")


def _transcript(turns: list[Turn]) -> list[dict]:
    return [{"role": t.role, "text": t.text} for t in turns]


def _turns_of(db: DbSession, session: Session) -> list[Turn]:
    stmt = select(Turn).where(Turn.session_id == session.id).order_by(Turn.idx)
    return list(db.execute(stmt).scalars().all())


def _add_turn(
    db: DbSession,
    session: Session,
    *,
    role: str,
    text: str,
    stt_confidence: float | None = None,
    llm_trace_id: str | None = None,
) -> Turn:
    last_idx = db.execute(
        select(Turn.idx).where(Turn.session_id == session.id).order_by(Turn.idx.desc()).limit(1)
    ).scalar()
    turn = Turn(
        session_id=session.id,
        idx=(last_idx or 0) + 1,
        role=role,
        text=text,
        stt_confidence=stt_confidence,
        llm_trace_id=llm_trace_id,
    )
    db.add(turn)
    return turn


def _complete(db: DbSession, session: Session) -> None:
    """Финал сессии: completed + completed_at + duration_sec от started_at.

    SQLite отдаёт naive datetime — нормализуем к UTC перед вычитанием.
    """
    completed = utcnow()
    session.status = "completed"
    session.completed_at = completed
    started = session.started_at
    if started is not None:
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        session.duration_sec = (completed - started).total_seconds()
    db.commit()


def _fail(db: DbSession, session: Session, exc: Exception) -> None:
    """Необработанная ошибка start/answer: status=failed, error (ARCHITECTURE §API)."""
    session.status = "failed"
    session.error = str(exc)
    db.commit()


def _http_error(status: int, detail: str) -> HTTPException:
    logger.warning("sessions: %d — %s", status, detail)
    return HTTPException(status_code=status, detail=detail)


def _session_brief(session: Session) -> dict:
    return {
        "id": session.id,
        "created_at": session.created_at,
        "status": session.status,
        "position_title": session.position_title,
        "seniority": session.seniority,
        "language": session.language,
        "style": session.style,
        "planned_questions": session.planned_questions,
        "overall_score": session.overall_score,
    }


def _session_state(db: DbSession, session: Session) -> dict:
    state = _session_brief(session)
    state.update(
        {
            "vacancy_text": session.vacancy_text,
            "plan_json": json.loads(session.plan_json) if session.plan_json else None,
            "error": session.error,
            "started_at": session.started_at,
            "completed_at": session.completed_at,
            "duration_sec": session.duration_sec,
            "turns": [
                {
                    "idx": t.idx,
                    "role": t.role,
                    "text": t.text,
                    "stt_confidence": t.stt_confidence,
                    "llm_trace_id": t.llm_trace_id,
                }
                for t in _turns_of(db, session)
            ],
        }
    )
    return state
