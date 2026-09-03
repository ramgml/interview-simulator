"""Тесты интервьюера (T131): STYLE_PROMPTS, build_plan, conduct_turn, fallback. Без сети."""

import json

import pytest

from app.errors import InterviewError
from app.interviewer import (
    FINISH_FALLBACK_TEXT,
    MAX_TURNS,
    VACANCY_MAX_CHARS,
    STYLE_PROMPTS,
    build_plan,
    conduct_turn,
    first_question,
)


class FakeCompletions:
    """Стаб client.chat.completions: помнит вызовы, отдаёт заготовленный ответ."""

    def __init__(self, client):
        self._client = client
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._client.response


class FakeChat:
    def __init__(self, client):
        self.completions = FakeCompletions(client)


class FakeClient:
    """Стаб OpenAI-клиента: response — заготовленный ответ create(); счётчик вызовов."""

    def __init__(self, response: str):
        self.response = response
        self.calls: list[dict] = []
        self.chat = FakeChat(self)
        self.chat.completions.calls = self.calls

    @property
    def call_count(self) -> int:
        return len(self.calls)


class FakeFailingClient:
    """Стаб недоступного endpoint: create() всегда бросает заданное исключение."""

    def __init__(self, exc: Exception):
        self.chat = FakeChat(self)
        self.chat.completions.create = lambda **kwargs: (_ for _ in ()).throw(exc)


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]
        self.usage = None


def make_session(**overrides):
    from app.models import Session

    defaults = dict(
        id="abc123",
        vacancy_text="Python-разработчик: FastAPI, PostgreSQL, Docker",
        seniority="middle",
        language="ru",
        style="friendly",
    )
    defaults.update(overrides)
    return Session(**defaults)


PLAN_JSON = json.dumps(
    {
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
    },
    ensure_ascii=False,
)

TURN_FOLLOWUP = (
    '{"action": "followup", "text": "Уточните про Depends.", "covered_topic": "FastAPI"}'
)
TURN_NEXT = (
    '{"action": "next_question", "text": "Чем index отличается от unique constraint?",'
    ' "covered_topic": "FastAPI"}'
)
TURN_FINISH = '{"action": "finish", "text": "Спасибо, интервью завершено", "covered_topic": null}'


# --- STYLE_PROMPTS ---------------------------------------------------------------


def test_style_prompts_has_all_three_keys():
    assert set(STYLE_PROMPTS) == {"friendly", "strict", "realistic"}
    assert all(isinstance(v, str) and v for v in STYLE_PROMPTS.values())


# --- build_plan ------------------------------------------------------------------


def test_build_plan_sends_vacancy_seniority_language_and_returns_plan():
    client = FakeClient(FakeResponse(PLAN_JSON))
    plan = build_plan(client, make_session(), model="test-model")
    assert plan["position_title"] == "Python-разработчик"
    assert client.calls[0]["model"] == "test-model"
    system = client.calls[0]["messages"][0]["content"]
    assert "Python-разработчик: FastAPI" in client.calls[0]["messages"][1]["content"]
    assert "уровень кандидата: middle" in system
    assert "язык: ru" in system


def test_build_plan_truncates_long_vacancy_to_8000():
    client = FakeClient(FakeResponse(PLAN_JSON))
    vacancy = "x" * (VACANCY_MAX_CHARS + 500)
    build_plan(client, make_session(vacancy_text=vacancy), model="test-model")
    sent = client.calls[0]["messages"][1]["content"]
    assert len(sent) == VACANCY_MAX_CHARS


def test_build_plan_without_seniority_says_not_specified():
    client = FakeClient(FakeResponse(PLAN_JSON))
    build_plan(client, make_session(seniority=None), model="test-model")
    assert "уровень кандидата: не указан" in client.calls[0]["messages"][0]["content"]


def test_build_plan_error_propagates():
    client = FakeFailingClient(ConnectionError("down"))
    with pytest.raises(InterviewError):
        build_plan(client, make_session(), model="test-model")


# --- conduct_turn ----------------------------------------------------------------


def test_conduct_turn_followup_branch():
    client = FakeClient(FakeResponse(TURN_FOLLOWUP))
    decision = conduct_turn(client, make_session(plan_json=PLAN_JSON), [], model="test-model")
    assert decision["action"] == "followup"
    assert client.calls[0]["model"] == "test-model"
    system = client.calls[0]["messages"][0]["content"]
    assert STYLE_PROMPTS["friendly"] in system


def test_conduct_turn_next_question_branch():
    client = FakeClient(FakeResponse(TURN_NEXT))
    decision = conduct_turn(client, make_session(plan_json=PLAN_JSON), [], model="test-model")
    assert decision["action"] == "next_question"


def test_conduct_turn_finish_branch():
    client = FakeClient(FakeResponse(TURN_FINISH))
    decision = conduct_turn(client, make_session(plan_json=PLAN_JSON), [], model="test-model")
    assert decision["action"] == "finish"


def test_conduct_turn_sends_plan_and_transcript_capped_at_24():
    client = FakeClient(FakeResponse(TURN_NEXT))
    transcript = [{"role": "candidate", "text": f"ответ {i}"} for i in range(30)]
    conduct_turn(client, make_session(plan_json=PLAN_JSON), transcript, model="test-model")
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert len(payload["transcript"]) == MAX_TURNS
    assert payload["plan"]["position_title"] == "Python-разработчик"


def test_build_plan_sends_model_not_session_style():
    """Модель из параметра уходит в LLM; стиль сессии именем модели не является (T152)."""
    client = FakeClient(FakeResponse(PLAN_JSON))
    build_plan(client, make_session(style="friendly"), model="glm/glm-5.3-flash")
    assert client.calls[0]["model"] == "glm/glm-5.3-flash"
    assert client.calls[0]["model"] not in {"friendly", "strict", "realistic"}


def test_conduct_turn_sends_model_not_style_but_keeps_style_in_prompt():
    """TURN: model — из параметра, стиль — остаётся в system-промпте (T152)."""
    client = FakeClient(FakeResponse(TURN_FOLLOWUP))
    decision = conduct_turn(
        client, make_session(style="friendly", plan_json=PLAN_JSON), [], model="glm/glm-5.3-flash"
    )
    assert decision["action"] == "followup"
    assert client.calls[0]["model"] == "glm/glm-5.3-flash"
    assert client.calls[0]["model"] not in {"friendly", "strict", "realistic"}
    assert STYLE_PROMPTS["friendly"] in client.calls[0]["messages"][0]["content"]


# --- conduct_turn fallback -------------------------------------------------------


def test_conduct_turn_llm_error_falls_back_to_unasked_question():
    client = FakeFailingClient(ConnectionError("down"))
    session = make_session(plan_json=PLAN_JSON)
    transcript = [{"role": "interviewer", "text": "Расскажите про DI в FastAPI."}]
    decision = conduct_turn(client, session, transcript, model="test-model")
    assert decision == {
        "action": "next_question",
        "text": "Чем index отличается от unique constraint?",
        "covered_topic": None,
    }


def test_conduct_turn_llm_error_all_asked_finishes():
    client = FakeFailingClient(ConnectionError("down"))
    session = make_session(plan_json=PLAN_JSON)
    transcript = [
        {"role": "interviewer", "text": "Расскажите про DI в FastAPI."},
        {"role": "interviewer", "text": "Чем index отличается от unique constraint?"},
    ]
    decision = conduct_turn(client, session, transcript, model="test-model")
    assert decision["action"] == "finish"
    assert decision["text"] == FINISH_FALLBACK_TEXT


def test_conduct_turn_llm_error_without_plan_finishes():
    client = FakeFailingClient(ConnectionError("down"))
    decision = conduct_turn(client, make_session(), [], model="test-model")
    assert decision["action"] == "finish"
    assert decision["text"] == FINISH_FALLBACK_TEXT


# --- first_question ----------------------------------------------------------------


def test_first_question_returns_first_of_first_round():
    plan = json.loads(PLAN_JSON)
    assert first_question(plan) == "Расскажите про DI в FastAPI."


def test_first_question_empty_plan_returns_none():
    assert first_question({}) is None
