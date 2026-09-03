"""Тесты SQLite-слоя (T155): WAL + busy_timeout на движке, ранний commit хода кандидата.

Гонка: LLM-вызов внутри /answer держал write-транзакцию (autoflush INSERT) на всё время
вызова — параллельная запись падала `OperationalError: database is locked`. После фикса
ход кандидата коммитится ДО conduct_turn, а прагмы движка — WAL + busy_timeout=10000.
"""

import threading
import time
from types import SimpleNamespace

import pytest

from app import interviewer, llm
from app.db import init_db
from app.errors import InterviewError
from app.main import app

PLAN = {
    "position_title": "Python-разработчик",
    "competencies": ["Backend"],
    "rounds": [
        {
            "type": "technical",
            "questions": [
                {
                    "topic": "FastAPI",
                    "question": "Расскажите про DI в FastAPI.",
                    "competency": "Backend",
                },
            ],
        }
    ],
}


def _resp(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=None
    )


class SlowRouterFakeClient:
    """Стаб OpenAI-клиента: create держит задержку — окно write-транзакции для гонки."""

    def __init__(self, delay: float):
        self.delay = delay

    @property
    def chat(self):
        return self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        time.sleep(self.delay)
        return _resp('{"action": "next_question", "text": "Следующий?", "covered_topic": null}')


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


def _create(client) -> str:
    resp = client.post("/api/sessions", json={"vacancy_text": "Python: FastAPI, PostgreSQL"})
    assert resp.status_code == 200
    return resp.json()["id"]


def _start(client, sid: str) -> None:
    resp = client.post(f"/api/sessions/{sid}/start")
    assert resp.status_code == 200


def _slow_turn_stubs(monkeypatch, delay: float) -> dict:
    """Стабы LLM: PLAN без задержки, TURN-вызов с задержкой delay (окно гонки)."""
    calls = {"count": 0}

    def fake_json_chat(client, model, messages, *, temperature, max_tokens):
        calls["count"] += 1
        first_user = messages[1]["content"]
        if "rounds" not in first_user:
            return PLAN
        time.sleep(delay)
        return {"action": "next_question", "text": "Следующий?", "covered_topic": None}

    monkeypatch.setattr(interviewer, "json_chat", fake_json_chat)
    monkeypatch.setattr(llm, "get_client", lambda s: SlowRouterFakeClient(delay))
    return calls


# --- engine: прагмы -----------------------------------------------------------------


def test_make_engine_sqlite_sets_wal_and_busy_timeout(tmp_path):
    """make_engine на sqlite-URL: journal_mode=wal и busy_timeout на соединении движка."""
    import app.db as db_mod
    from sqlalchemy import text

    engine = db_mod.make_engine(f"sqlite:///{tmp_path / 'pragma.db'}")
    try:
        with engine.connect() as conn:
            assert conn.execute(text("PRAGMA journal_mode")).scalar() == "wal"
            busy_ms = conn.execute(text("PRAGMA busy_timeout")).scalar()
        assert isinstance(busy_ms, int) and busy_ms >= 10000
    finally:
        engine.dispose()


def test_make_engine_non_sqlite_no_sqlite_pragmas():
    """Не-sqlite ветка не трогается: прагмы ставятся только для sqlite-URL.

    Драйвер pymysql не установлен: ImportError при создании движка доказывает, что
    make_engine для не-sqlite URL дошёл до create_engine и не выполнял sqlite-прагм.
    """
    import pytest

    import app.db as db_mod

    with pytest.raises(ImportError, match="pymysql"):
        db_mod.make_engine("mysql+pymysql://u:p@localhost/db")


def test_journal_mode_persists_after_init_db(tmp_path):
    """init_db на tmp-БД: journal_mode файла — wal (отдельное raw-соединение sqlite3)."""
    import sqlite3

    import app.db as db_mod

    db_path = tmp_path / "journal.db"
    engine = db_mod.make_engine(f"sqlite:///{db_path}")
    db_mod.Base.metadata.create_all(engine)
    engine.dispose()

    raw = sqlite3.connect(str(db_path))
    try:
        assert raw.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    finally:
        raw.close()


# --- конкурентная запись -------------------------------------------------------------


def test_parallel_write_during_llm_call_no_database_is_locked(client, monkeypatch):
    """Пока /answer ждёт LLM, PUT /api/settings с другой portal-нити пишет в БД.

    Задержка стаба (7с) больше дефолтного busy_timeout python-sqlite3 (5с): до фикса
    INSERT кандидата открывал write-транзакцию (autoflush) и держал её до конца LLM-
    вызова — PUT выжидал 5с и падал OperationalError: database is locked. Второй
    TestClient нужен для реальной параллельности: у каждого свой portal-поток.
    """
    from fastapi.testclient import TestClient

    _slow_turn_stubs(monkeypatch, delay=7.0)
    sid = _create(client)
    _start(client, sid)

    with TestClient(app) as settings_client:
        answer_result: dict = {}

        def _answer():
            answer_result["resp"] = client.post(
                f"/api/sessions/{sid}/answer", json={"text": "ответ"}
            )

        answer_thread = threading.Thread(target=_answer)
        answer_thread.start()
        time.sleep(0.5)  # answer уже в conduct_turn (внутри write-транзакции до фикса)
        try:
            resp = settings_client.put("/api/settings", json={"model": "glm/glm-5.3-flash"})
            assert resp.status_code == 200, resp.json()
        finally:
            answer_thread.join(timeout=15)
    assert answer_result["resp"].status_code == 200, answer_result["resp"].json()

    # Оба хода на месте: interviewer-вопрос, candidate-ответ, interviewer-следующий
    state = client.get(f"/api/sessions/{sid}").json()
    assert [t["role"] for t in state["turns"]] == ["interviewer", "candidate", "interviewer"]


# --- 502-путь: ход кандидата не теряется ----------------------------------------------


def test_answer_llm_failure_keeps_candidate_turn_and_502(client, monkeypatch):
    """InterviewError от LLM-слоя → 502, но ход кандидата остаётся в БД (status=failed)."""

    state = {"fail_turn": False}

    def get_client_or_boom(s):
        if state["fail_turn"]:
            raise InterviewError("Сервис LLM недоступен")
        return object()

    monkeypatch.setattr(llm, "get_client", get_client_or_boom)

    def plan_only(client, model, messages, *, temperature, max_tokens):
        return PLAN

    monkeypatch.setattr(interviewer, "json_chat", plan_only)
    sid = _create(client)
    _start(client, sid)
    state["fail_turn"] = True  # PLAN прошёл; теперь ломаем TURN-путь (answer)
    resp = client.post(f"/api/sessions/{sid}/answer", json={"text": "важный ответ"})
    assert resp.status_code == 502
    assert resp.json()["detail"] == "Сервис LLM недоступен"

    state = client.get(f"/api/sessions/{sid}").json()
    assert state["status"] == "failed"
    assert [t["role"] for t in state["turns"]] == ["interviewer", "candidate"]
    assert state["turns"][1]["text"] == "важный ответ"


# --- регресс обычного пути -------------------------------------------------------------


def test_answer_normal_flow_unchanged(client, monkeypatch):
    """После раннего commit кандидата: оба хода записаны, ответ 200, вопрос интервьюера."""
    calls = _slow_turn_stubs(monkeypatch, delay=0)
    sid = _create(client)
    _start(client, sid)

    resp = client.post(f"/api/sessions/{sid}/answer", json={"text": "мой ответ"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["done"] is False
    assert body["question_text"] == "Следующий?"
    assert calls["count"] == 2  # PLAN + TURN

    state = client.get(f"/api/sessions/{sid}").json()
    assert [t["role"] for t in state["turns"]] == ["interviewer", "candidate", "interviewer"]
    assert state["turns"][1]["text"] == "мой ответ"
