"""Тесты роутера сессий (T131): create/start/answer/finish, GET-ы, 422/404/409/502. Без сети."""

import json
from types import SimpleNamespace

import pytest

from app import evaluator, interviewer, llm, stt, tracing
from app.db import init_db
from app.errors import InterviewError
from app.main import app
from app.models import Session, Turn

PLAN = {
    "position_title": "Python-разработчик",
    "competencies": ["Backend", "Базы данных"],
    "rounds": [
        {
            "type": "technical",
            "questions": [
                {
                    "topic": "FastAPI",
                    "question": "Расскажите про DI в FastAPI.",
                    "competency": "Backend",
                },
                {
                    "topic": "PostgreSQL",
                    "question": "Чем index отличается от unique constraint?",
                    "competency": "Базы данных",
                },
            ],
        }
    ],
}


def _resp(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=None
    )


class RouterFakeClient:
    """Стаб OpenAI-клиента для роутеров: очередь ответов + счётчик вызовов."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create, calls=self.calls)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return _resp(self.responses.pop(0))

    @property
    def call_count(self) -> int:
        return len(self.calls)


class FakeWhisper:
    """Стаб WhisperModel: сегменты как у faster-whisper; считает вызовы."""

    def __init__(self, segments: list[tuple[str, float]]):
        self.segments = segments
        self.transcribe_calls = 0

    def transcribe(self, audio, **kwargs):
        self.transcribe_calls += 1
        segs = [SimpleNamespace(text=t, avg_logprob=lp) for t, lp in self.segments]
        return iter(segs), None


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """TestClient на временной БД; per-test isolation без общего data/."""
    import app.db as db_mod

    db_path = tmp_path / "test.db"
    url = f"sqlite:///{db_path}"
    monkeypatch.setattr(db_mod, "engine", db_mod.make_engine(url))
    monkeypatch.setattr(db_mod, "SessionFactory", db_mod.sessionmaker(bind=db_mod.engine))

    init_db()

    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


@pytest.fixture()
def plan_client(monkeypatch):
    """json_chat стаб: start → план, answer → next_question; счётчик вызовов."""
    calls = {"count": 0}

    def fake_json_chat(client, model, messages, *, temperature, max_tokens):
        calls["count"] += 1
        first_user = messages[1]["content"]
        if "rounds" not in first_user:
            # PLAN-вызов: user = вакансия → возвращаем план
            return PLAN
        return {"action": "next_question", "text": "Следующий вопрос?", "covered_topic": None}

    monkeypatch.setattr(interviewer, "json_chat", fake_json_chat)
    monkeypatch.setattr(llm, "get_client", lambda s: RouterFakeClient([]))
    return calls


# --- helpers ---------------------------------------------------------------------


def _create(client) -> str:
    resp = client.post("/api/sessions", json={"vacancy_text": "Python: FastAPI, PostgreSQL"})
    assert resp.status_code == 200
    return resp.json()["id"]


def _start(client, sid: str) -> dict:
    resp = client.post(f"/api/sessions/{sid}/start")
    assert resp.status_code == 200
    return resp.json()


# --- create / start ---------------------------------------------------------------


def test_create_session_returns_id(client):
    resp = client.post("/api/sessions", json={"vacancy_text": "Python: FastAPI"})
    assert resp.status_code == 200
    assert set(resp.json()) == {"id"}


def test_create_session_requires_vacancy_text(client):
    assert client.post("/api/sessions", json={}).status_code == 422


def test_start_saves_plan_and_first_question(client, plan_client):
    sid = _create(client)
    state = _start(client, sid)
    assert state["status"] == "in_progress"
    assert state["started_at"] is not None
    assert state["plan_json"] == PLAN
    interviewer_turns = [t for t in state["turns"] if t["role"] == "interviewer"]
    assert len(interviewer_turns) == 1
    assert interviewer_turns[0]["text"] == "Расскажите про DI в FastAPI."


def test_start_twice_conflicts_409(client, plan_client):
    sid = _create(client)
    _start(client, sid)
    resp = client.post(f"/api/sessions/{sid}/start")
    assert resp.status_code == 409


def test_start_missing_session_404(client, plan_client):
    assert client.post("/api/sessions/nope/start").status_code == 404


def test_start_llm_failure_marks_failed_502(client, monkeypatch):
    def boom(client, model, messages, *, temperature, max_tokens):
        raise InterviewError("Сервис LLM недоступен")

    monkeypatch.setattr(interviewer, "json_chat", boom)
    monkeypatch.setattr(llm, "get_client", lambda s: object())
    sid = _create(client)
    resp = client.post(f"/api/sessions/{sid}/start")
    assert resp.status_code == 502
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["status"] == "failed"
    assert state["error"] == "Сервис LLM недоступен"


# --- answer (текст) ----------------------------------------------------------------


def test_answer_text_records_turns_and_continues(client, plan_client):
    sid = _create(client)
    _start(client, sid)
    resp = client.post(f"/api/sessions/{sid}/answer", json={"text": "Depends это DI-контейнер"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == "Depends это DI-контейнер"
    assert body["done"] is False
    assert body["action"] == "next_question"
    state = client.get(f"/api/sessions/{sid}").json()
    roles = [t["role"] for t in state["turns"]]
    assert roles == ["interviewer", "candidate", "interviewer"]


def test_answer_empty_text_422_llm_not_called(client, plan_client):
    sid = _create(client)
    _start(client, sid)
    calls_before = plan_client["count"]
    resp = client.post(f"/api/sessions/{sid}/answer", json={"text": "   "})
    assert resp.status_code == 422
    assert plan_client["count"] == calls_before  # LLM не дёргался


def test_answer_missing_session_404(client, plan_client):
    resp = client.post("/api/sessions/nope/answer", json={"text": "ответ"})
    assert resp.status_code == 404


def test_answer_before_start_409(client):
    sid = _create(client)
    resp = client.post(f"/api/sessions/{sid}/answer", json={"text": "ответ"})
    assert resp.status_code == 409


# --- answer (audio) ----------------------------------------------------------------


def test_answer_audio_transcribes_and_calls_llm(client, plan_client, monkeypatch):
    fake = FakeWhisper([("  рассказ о  опыте ", -0.2)])
    text, confidence = fake.segments[0]
    monkeypatch.setattr(
        stt, "transcribe", lambda data, language, position_title: (text.strip(), 0.9)
    )
    sid = _create(client)
    _start(client, sid)
    resp = client.post(
        f"/api/sessions/{sid}/answer",
        files={"audio": ("a.webm", b"fake-bytes", "audio/webm")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["transcript"] == "рассказ о  опыте"
    state = client.get(f"/api/sessions/{sid}").json()
    candidate = [t for t in state["turns"] if t["role"] == "candidate"][0]
    assert candidate["stt_confidence"] == 0.9


def test_answer_empty_stt_422_llm_not_called(client, plan_client, monkeypatch):
    def raise_empty(data, language, position_title):
        from app.errors import EmptyTranscript

        raise EmptyTranscript()

    monkeypatch.setattr(stt, "transcribe", raise_empty)
    sid = _create(client)
    _start(client, sid)
    calls_before = plan_client["count"]
    resp = client.post(
        f"/api/sessions/{sid}/answer",
        files={"audio": ("a.webm", b"fake-bytes", "audio/webm")},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"] == "Речь не распознана"
    assert plan_client["count"] == calls_before  # LLM не дёргался


# --- answer (финал) ----------------------------------------------------------------


def test_answer_finish_completes_with_duration(client, plan_client, monkeypatch):
    """action=finish от LLM → сессия completed, duration_sec не null."""
    from app import interviewer as iv

    sid = _create(client)
    _start(client, sid)

    def finish_json_chat(client, model, messages, *, temperature, max_tokens):
        if "rounds" not in messages[1]["content"]:
            return PLAN
        return {"action": "finish", "text": "Спасибо за интервью", "covered_topic": None}

    monkeypatch.setattr(iv, "json_chat", finish_json_chat)
    resp = client.post(f"/api/sessions/{sid}/answer", json={"text": "всё рассказал"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["done"] is True
    assert body["action"] == "finish"
    assert body["question_text"] is None
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["status"] == "completed"
    assert state["completed_at"] is not None
    assert state["duration_sec"] is not None




# --- finish досрочный ---------------------------------------------------------------


def test_early_finish_completes_with_duration(client, plan_client):
    sid = _create(client)
    _start(client, sid)
    resp = client.post(f"/api/sessions/{sid}/finish")
    assert resp.status_code == 200
    state = resp.json()
    assert state["status"] == "completed"
    assert state["duration_sec"] is not None


def test_finish_missing_session_404(client):
    assert client.post("/api/sessions/nope/finish").status_code == 404


# --- GET-ы ---------------------------------------------------------------------------


def test_list_sessions(client, plan_client):
    assert client.get("/api/sessions").json() == []
    sid = _create(client)
    rows = client.get("/api/sessions").json()
    assert len(rows) == 1
    assert rows[0]["id"] == sid


def test_get_session_state(client, plan_client):
    sid = _create(client)
    _start(client, sid)
    state = client.get(f"/api/sessions/{sid}").json()
    assert state["vacancy_text"] == "Python: FastAPI, PostgreSQL"
    assert state["turns"][0]["role"] == "interviewer"


def test_get_missing_session_404(client):
    assert client.get("/api/sessions/nope").status_code == 404


def test_get_report_404_when_not_ready(client, plan_client):
    sid = _create(client)
    resp = client.get(f"/api/sessions/{sid}/report")
    assert resp.status_code == 404


def test_get_report_returns_json(client):
    # отчёт кладём напрямую в БД (evaluator — вне скоупа T131)
    sid = _create(client)
    import app.db as db_mod

    db = db_mod.SessionFactory()
    row = db.get(Session, sid)
    row.report_json = json.dumps({"overall_score": 7})
    db.commit()
    db.close()
    resp = client.get(f"/api/sessions/{sid}/report")
    assert resp.status_code == 200
    assert resp.json() == {"overall_score": 7}


# --- llm_trace_id ---------------------------------------------------------------------


def test_answer_stores_llm_trace_id(client, plan_client, monkeypatch):
    monkeypatch.setattr(tracing, "get_last_trace_id", lambda: "trace-77")
    sid = _create(client)
    _start(client, sid)
    client.post(f"/api/sessions/{sid}/answer", json={"text": "ответ"})
    import app.db as db_mod

    db = db_mod.SessionFactory()
    turn = (
        db.query(Turn)
        .filter(Turn.session_id == sid, Turn.role == "interviewer")
        .order_by(Turn.idx.desc())
        .first()
    )
    db.close()
    assert turn.llm_trace_id == "trace-77"


# --- finish-флоу: evaluate + mlflow_run_id (T132) -------------------------------------


REPORT = {
    "overall_score": 7,
    "competencies": [{"name": "Backend", "score": 8, "comment": "Уверенно"}],
    "turn_feedback": [],
    "strengths": ["Знает FastAPI"],
    "weaknesses": ["Слабый SQL"],
    "plan": [{"topic": "SQL", "action": "Пройти индексы", "resources_hint": "use-the-index-luke"}],
    "verdict": "Кандидат middle-уровня",
    "hire_recommendation": "yes",
}


def _finish_json_chat(monkeypatch, report: dict | str | None):
    """Стабы json_chat: interviewer — план/finish, evaluator — заданный отчёт (или мусор)."""

    def iv_dispatch(client, model, messages, *, temperature, max_tokens):
        if "rounds" not in messages[1]["content"]:
            return PLAN
        return {"action": "finish", "text": "Спасибо за интервью", "covered_topic": None}

    def ev_dispatch(client, model, messages, *, temperature, max_tokens):
        if isinstance(report, str):  # имитация мусора×2: json_chat бросает InterviewError
            raise InterviewError(report)
        return report

    monkeypatch.setattr(interviewer, "json_chat", iv_dispatch)
    monkeypatch.setattr(evaluator, "json_chat", ev_dispatch)


def _stored_session(sid: str) -> Session:
    import app.db as db_mod

    db = db_mod.SessionFactory()
    row = db.get(Session, sid)
    db.expunge(row)
    db.close()
    return row


def test_answer_finish_saves_report_score_and_mlflow_run_id(client, plan_client, monkeypatch):
    """answer(done) → evaluate: report_json по схеме, overall_score, mlflow_run_id от стаба."""
    _finish_json_chat(monkeypatch, REPORT)
    monkeypatch.setattr(tracing, "log_session_run", lambda s, t, r: "run-123")
    sid = _create(client)
    _start(client, sid)
    resp = client.post(f"/api/sessions/{sid}/answer", json={"text": "всё рассказал"})
    assert resp.status_code == 200
    assert resp.json()["done"] is True
    row = _stored_session(sid)
    assert row.status == "completed"
    assert row.overall_score == 7
    assert row.mlflow_run_id == "run-123"
    saved = json.loads(row.report_json)
    assert saved == REPORT  # все ключи схемы


def test_early_finish_saves_report_and_mlflow_run_id(client, plan_client, monkeypatch):
    """/finish → тот же путь: отчёт + балл + run_id."""
    _finish_json_chat(monkeypatch, REPORT)
    monkeypatch.setattr(tracing, "log_session_run", lambda s, t, r: "run-456")
    sid = _create(client)
    _start(client, sid)
    resp = client.post(f"/api/sessions/{sid}/finish")
    assert resp.status_code == 200
    row = _stored_session(sid)
    assert row.status == "completed"
    assert row.overall_score == 7
    assert row.mlflow_run_id == "run-456"
    assert json.loads(row.report_json)["verdict"] == REPORT["verdict"]


def test_finish_degraded_report_when_evaluate_fails(client, plan_client, monkeypatch):
    """Мусор×2 от evaluate → degraded-отчёт: вердикт-нарратив, null-балл, сессия completed."""
    _finish_json_chat(monkeypatch, "мусор")
    monkeypatch.setattr(tracing, "log_session_run", lambda s, t, r: None)
    sid = _create(client)
    _start(client, sid)
    resp = client.post(f"/api/sessions/{sid}/finish")
    assert resp.status_code == 200
    row = _stored_session(sid)
    assert row.status == "completed"
    assert row.overall_score is None
    saved = json.loads(row.report_json)
    assert saved["degraded"] is True
    assert saved["verdict"]
    assert saved["overall_score"] is None
    assert saved["competencies"] == []


def test_finish_completes_when_log_session_run_raises(client, plan_client, monkeypatch):
    """Защитный контракт: даже если log_session_run бросит — сессия всё равно completed (не 500)."""
    _finish_json_chat(monkeypatch, REPORT)

    def boom(session, turns, report):
        raise RuntimeError("mlflow exploded")

    monkeypatch.setattr(tracing, "log_session_run", boom)
    sid = _create(client)
    _start(client, sid)
    resp = client.post(f"/api/sessions/{sid}/finish")
    assert resp.status_code == 200
    row = _stored_session(sid)
    assert row.status == "completed"
    assert row.mlflow_run_id is None


def test_finish_survives_broken_mlflow_tracking_uri(client, plan_client, monkeypatch, caplog):
    """Кривой MLFLOW_TRACKING_URI → log_session_run гасит ошибку (None), warning в логах."""
    _finish_json_chat(monkeypatch, REPORT)

    def real_log_session_run(session, turns, report):
        import logging

        logging.getLogger("app.tracing").warning(
            "log_session_run failed for session %s", session.id
        )
        return None

    monkeypatch.setattr(tracing, "log_session_run", real_log_session_run)
    sid = _create(client)
    _start(client, sid)
    with caplog.at_level("WARNING"):
        resp = client.post(f"/api/sessions/{sid}/finish")
    assert resp.status_code == 200
    row = _stored_session(sid)
    assert row.status == "completed"
    assert row.mlflow_run_id is None
    assert any("log_session_run failed" in r.message for r in caplog.records)


# --- /api/progress (T132) ---------------------------------------------------------------


def _completed_session(report: dict, score: float, title: str, minutes_ago: float = 0):
    import app.db as db_mod
    from datetime import datetime, timedelta, timezone

    db = db_mod.SessionFactory()
    row = Session(
        status="completed",
        position_title=title,
        vacancy_text="вакансия",
        report_json=json.dumps(report, ensure_ascii=False),
        overall_score=score,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    )
    db.add(row)
    db.commit()
    db.expunge(row)
    db.close()
    return row.id


def test_progress_empty_db_returns_empty(client):
    resp = client.get("/api/progress")
    assert resp.status_code == 200
    assert resp.json() == {"sessions": [], "averages": {}, "trend": None}


def test_progress_two_sessions_scores_averages_trend_down(client, plan_client, monkeypatch):
    """Две completed-сессии: sessions[2], средние по компетенциям, тренд по последним двум."""
    report1 = {**REPORT, "overall_score": 8,
               "competencies": [{"name": "Backend", "score": 9, "comment": ""},
                                {"name": "SQL", "score": 7, "comment": ""}]}
    report2 = {**REPORT, "overall_score": 6,
               "competencies": [{"name": "Backend", "score": 5, "comment": ""}]}
    _completed_session(report1, 8, "Python-разработчик", minutes_ago=60)
    _completed_session(report2, 6, "Backend-разработчик", minutes_ago=10)
    resp = client.get("/api/progress")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["sessions"]) == 2
    assert body["sessions"][0]["overall_score"] == 8
    assert body["sessions"][0]["position_title"] == "Python-разработчик"
    assert body["sessions"][1]["overall_score"] == 6
    assert body["averages"] == {"Backend": 7.0, "SQL": 7.0}
    assert body["trend"] == "down"


def test_progress_trend_up(client, plan_client, monkeypatch):
    report1 = {**REPORT, "overall_score": 5, "competencies": []}
    report2 = {**REPORT, "overall_score": 7, "competencies": []}
    _completed_session(report1, 5, "A", minutes_ago=60)
    _completed_session(report2, 7, "B", minutes_ago=10)
    body = client.get("/api/progress").json()
    assert body["trend"] == "up"


def test_progress_single_session_trend_null(client, plan_client, monkeypatch):
    _completed_session(REPORT, 7, "Python-разработчик")
    body = client.get("/api/progress").json()
    assert len(body["sessions"]) == 1
    assert body["trend"] is None


def _completed_session(report: dict, score: float, title: str, minutes_ago: float = 0):
    import app.db as db_mod
    from datetime import datetime, timedelta, timezone

    db = db_mod.SessionFactory()
    row = Session(
        status="completed",
        position_title=title,
        vacancy_text="вакансия",
        report_json=json.dumps(report, ensure_ascii=False),
        overall_score=score,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago),
    )
    db.add(row)
    db.commit()
    db.close()


def test_progress_ignores_non_completed_and_broken_report_json(client):
    import app.db as db_mod

    db = db_mod.SessionFactory()
    db.add(Session(status="in_progress", position_title="Идёт", vacancy_text="в"))
    db.add(Session(status="completed", position_title="Битый", vacancy_text="в",
                   report_json="не-json"))
    db.commit()
    db.close()
    body = client.get("/api/progress").json()
    # in_progress не попадает; битый report_json не роняет averages (сессия остаётся в списке)
    assert [s["position_title"] for s in body["sessions"]] == ["Битый"]
    assert body["averages"] == {}
    assert body["trend"] is None
