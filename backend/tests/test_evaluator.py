"""Тесты оценщика (T132): evaluate — схема отчёта, degraded-fallback. Без сети."""

import json
import pytest
from types import SimpleNamespace

from app import evaluator
from app.errors import InterviewError
from app.evaluator import EVAL_SYSTEM_PROMPT, degraded_report, evaluate

REPORT = {
    "overall_score": 7,
    "competencies": [{"name": "Backend", "score": 8, "comment": "Уверенно"}],
    "turn_feedback": [
        {
            "turn_idx": 2,
            "question": "Расскажите про DI в FastAPI.",
            "answer": "Depends внедряет зависимости.",
            "score": 7,
            "good": "Понимание DI",
            "missed": "scopes",
            "strong_answer": "Пример с yield-зависимостью",
        }
    ],
    "strengths": ["Знает FastAPI"],
    "weaknesses": ["Слабый SQL"],
    "plan": [{"topic": "SQL", "action": "Пройти индексы", "resources_hint": "use-the-index-luke"}],
    "verdict": "Кандидат middle-уровня",
    "hire_recommendation": "yes",
}


@pytest.fixture(autouse=True)
def _transcript_db_on_tmp_path(tmp_path, monkeypatch):
    """Транскрипт в evaluate читает глобальный SessionFactory — якорим его на tmp_path."""
    import app.db as db_mod

    engine = db_mod.make_engine(f"sqlite:///{tmp_path / 'transcript.db'}")
    db_mod.Base.metadata.create_all(engine)
    monkeypatch.setattr(evaluator, "SessionFactory", db_mod.sessionmaker(bind=engine))


class FakeClient:
    """Стаб OpenAI-клиента: очередь ответов create() + счётчик вызовов."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create, calls=self.calls)
        )

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.responses.pop(0)))],
            usage=None,
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)


class FailingClient:
    """Стаб недоступного endpoint: create() всегда бросает заданное исключение."""

    def __init__(self, exc: Exception):
        self.exc = exc
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._fail, calls=self.calls)
        )

    def _fail(self, **kwargs):
        self.calls.append(kwargs)
        raise self.exc


def make_session(**overrides):
    from app.models import Session

    defaults = dict(
        id="abc123",
        vacancy_text="Python-разработчик: FastAPI",
        seniority="middle",
        language="ru",
        style="friendly",
        plan_json=json.dumps(
            {
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
                            }
                        ],
                    }
                ],
            }
        ),
    )
    defaults.update(overrides)
    return Session(**defaults)


REPORT_JSON = json.dumps(REPORT, ensure_ascii=False)


# --- evaluate: валидный отчёт -------------------------------------------------------


def test_evaluate_valid_json_returns_report_by_schema():
    client = FakeClient([REPORT_JSON])
    report = evaluate(client, make_session(), model="test-model")
    assert set(report) == {
        "overall_score",
        "competencies",
        "turn_feedback",
        "strengths",
        "weaknesses",
        "plan",
        "verdict",
        "hire_recommendation",
    }
    assert report["overall_score"] == 7
    assert report["hire_recommendation"] == "yes"
    assert report["competencies"][0]["name"] == "Backend"


def test_evaluate_sends_plan_and_transcript():
    client = FakeClient([REPORT_JSON])
    evaluate(client, make_session(), model="test-model")
    assert client.calls[0]["model"] == "test-model"
    system = client.calls[0]["messages"][0]["content"]
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert "TOЛЬКО JSON" in system or "ТОЛЬКО JSON" in system
    assert payload["plan"]["position_title"] == "Python-разработчик"
    assert payload["transcript"] == []


def test_evaluate_strips_json_fences():
    fenced = f"```json\n{REPORT_JSON}\n```"
    client = FakeClient([fenced])
    report = evaluate(client, make_session(), model="test-model")
    assert report["overall_score"] == 7


def test_evaluate_temperature_is_02():
    client = FakeClient([REPORT_JSON])
    evaluate(client, make_session(), model="test-model")
    assert client.calls[0]["temperature"] == 0.2


def test_evaluate_wraps_in_json_fence_with_single_retry_and_parses():
    """Ретрай внутри json_chat: второй ответ в заборах — всё ещё валидный отчёт."""
    client = FakeClient(["мусор", f"```json\n{REPORT_JSON}\n```"])
    report = evaluate(client, make_session(), model="test-model")
    assert report["overall_score"] == 7
    assert client.call_count == 2


# --- evaluate: degraded-fallback ------------------------------------------------------


def test_evaluate_sends_model_not_session_style():
    """EVAL: model — из параметра, не стиль сессии (T152)."""
    client = FakeClient([REPORT_JSON])
    evaluate(client, make_session(style="friendly"), model="glm/glm-5.3-flash")
    assert client.calls[0]["model"] == "glm/glm-5.3-flash"
    assert client.calls[0]["model"] not in {"friendly", "strict", "realistic"}


def test_evaluate_garbage_twice_returns_degraded_with_verdict_and_nulls(caplog):
    client = FakeClient(["совсем не json", "опять мусор"])
    with caplog.at_level("WARNING"):
        report = evaluate(client, make_session(), model="test-model")
    assert report["degraded"] is True
    assert report["verdict"] == degraded_report()["verdict"]
    assert report["overall_score"] is None
    assert report["hire_recommendation"] is None
    for key in ("competencies", "turn_feedback", "strengths", "weaknesses", "plan"):
        assert report[key] == []
    assert client.call_count == 2
    assert any("degraded" in r.message for r in caplog.records)


def test_evaluate_llm_down_returns_degraded_not_raises():
    client = FailingClient(ConnectionError("down"))
    report = evaluate(client, make_session(), model="test-model")
    assert report["degraded"] is True
    assert report["overall_score"] is None
    assert report["verdict"]


def test_degraded_report_keys_match_schema():
    report = degraded_report()
    assert set(report) == {
        "degraded",
        "overall_score",
        "competencies",
        "turn_feedback",
        "strengths",
        "weaknesses",
        "plan",
        "verdict",
        "hire_recommendation",
    }


def test_evaluate_json_chat_error_mocked_is_degraded(monkeypatch):
    """Стаб json_chat, бросающий InterviewError — деградация без роняния (постановка T132)."""

    def boom(client, model, messages, *, temperature, max_tokens):
        raise InterviewError("LLM returned invalid JSON")

    monkeypatch.setattr(evaluator, "json_chat", boom)
    report = evaluate(FailingClient(ConnectionError("down")), make_session(), model="test-model")
    assert report["degraded"] is True


# --- prompts (константы модуля) ------------------------------------------------------


def test_eval_system_prompt_mentions_schema_fields():
    for key in ("overall_score", "competencies", "turn_feedback", "plan", "verdict",
                "hire_recommendation"):
        assert key in EVAL_SYSTEM_PROMPT


def test_evaluate_transcript_reads_turns_from_db(tmp_path, monkeypatch):
    """Транскрипт берётся из БД (turns по idx) — как в tracing, через SessionFactory."""
    import app.db as db_mod
    from app.models import Turn

    db_path = tmp_path / "t.db"
    engine = db_mod.make_engine(f"sqlite:///{db_path}")
    db_mod.Base.metadata.create_all(engine)
    factory = db_mod.sessionmaker(bind=engine)
    monkeypatch.setattr(evaluator, "SessionFactory", factory)

    with factory() as db:
        db.add(make_session())
        db.add(Turn(session_id="abc123", idx=1, role="interviewer", text="Вопрос?"))
        db.add(Turn(session_id="abc123", idx=2, role="candidate", text="Ответ."))
        db.commit()

    client = FakeClient([REPORT_JSON])
    evaluate(client, make_session(), model="test-model")
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert payload["transcript"] == [
        {"turn_idx": 1, "role": "interviewer", "text": "Вопрос?"},
        {"turn_idx": 2, "role": "candidate", "text": "Ответ."},
    ]
