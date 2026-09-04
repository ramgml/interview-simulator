"""Тесты LLM-слоя (T130): get_client local/cloud, chat/json_chat. Детерминированно, без сети."""

import pytest
from openai import OpenAI

from app import llm
from app.config import settings as env
from app.errors import InterviewError
from app.llm import (
    RETRY_SYSTEM_PROMPT,
    _strip_json_fences,
    chat,
    get_client,
    json_chat,
    resolve_model,
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
    """Стаб OpenAI-клиента: response — заготовленный ответ create()."""

    def __init__(self, response):
        self.response = response
        self.chat = FakeChat(self)


class FakeResponse:
    """Стаб ответа create(): content + finish_reason (usage с токенами)."""

    def __init__(self, content: str, finish_reason: str | None = "stop"):
        self.choices = [
            type(
                "C",
                (),
                {
                    "message": type("M", (), {"content": content})(),
                    "finish_reason": finish_reason,
                },
            )()
        ]
        self.usage = type(
            "U", (), {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        )()


class FakeFailingClient:
    """Стаб недоступного endpoint: create() всегда падает."""

    def __init__(self, exc: Exception):
        self.chat = type("Chat", (), {})()
        self.chat.completions = type("Completions", (), {})()
        self.chat.completions.create = lambda **kwargs: (_ for _ in ()).throw(exc)


def make_settings(provider: str, base_url: str = "", api_key: str | None = None):
    from app.models import Settings

    return Settings(
        id=1,
        provider=provider,
        base_url=base_url,
        api_key=api_key,
        model="test-model",
        whisper_model="",
        tts_voice="",
    )


# --- get_client ----------------------------------------------------------------


def test_get_client_local_uses_env_url():
    client = get_client(make_settings("local"))
    assert isinstance(client, OpenAI)
    assert str(client.base_url).rstrip("/") == env.local_llm_base_url.rstrip("/")
    assert client.api_key == "sk-local"


def test_get_client_cloud_uses_db_row():
    client = get_client(
        make_settings("cloud", base_url="https://api.example.com/v1", api_key="sk-test")
    )
    assert str(client.base_url).rstrip("/") == "https://api.example.com/v1"
    assert client.api_key == "sk-test"


@pytest.mark.parametrize("base_url,api_key", [("", "sk-test"), ("https://x", ""), (None, None)])
def test_get_client_cloud_incomplete_raises(base_url, api_key):
    with pytest.raises(InterviewError, match="Заполните облачный провайдер"):
        get_client(make_settings("cloud", base_url=base_url, api_key=api_key))



# --- resolve_model ---------------------------------------------------------------


def test_resolve_model_local_uses_env_model(monkeypatch):
    monkeypatch.setattr(llm.env, "local_llm_model", "test-local-model")
    assert resolve_model(make_settings("local")) == "test-local-model"


def test_resolve_model_cloud_uses_db_model():
    s = make_settings("cloud", base_url="https://x", api_key="sk-test")
    s.model = "cloud-model-x"
    assert resolve_model(s) == "cloud-model-x"


def test_resolve_model_cloud_empty_model_raises():
    s = make_settings("cloud", base_url="https://x", api_key="sk-test")
    s.model = ""
    with pytest.raises(InterviewError, match="Заполните модель"):
        resolve_model(s)


def test_resolve_model_unknown_provider_raises():
    with pytest.raises(InterviewError, match="Unknown provider"):
        resolve_model(make_settings("hybrid"))


# --- chat ----------------------------------------------------------------------

def test_chat_returns_content_and_usage():
    client = FakeClient(FakeResponse('{"ok": true}'))
    messages = [{"role": "user", "content": "ping"}]
    content, usage = chat(client, "m", messages, temperature=0.2, max_tokens=100)
    assert content == '{"ok": true}'
    assert usage == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "finish_reason": "stop",
    }
    call = client.chat.completions.calls[0]
    assert call["model"] == "m"
    assert call["messages"] == messages
    assert call["temperature"] == 0.2
    assert call["max_tokens"] == 100


def test_chat_endpoint_down_raises_interview_error():
    client = FakeFailingClient(ConnectionError("connection refused"))
    with pytest.raises(InterviewError):
        chat(client, "m", [{"role": "user", "content": "x"}], temperature=0.2, max_tokens=10)


# --- json_chat -----------------------------------------------------------------


def test_json_chat_plain_json():
    client = FakeClient(FakeResponse('{"action": "next_question"}'))
    result = json_chat(client, "m", [], temperature=0.2, max_tokens=100)
    assert result == {"action": "next_question"}
    assert len(client.chat.completions.calls) == 1


def test_json_chat_strips_json_fence():
    client = FakeClient(FakeResponse('```json\n{"a": 1}\n```'))
    assert json_chat(client, "m", [], temperature=0.2, max_tokens=10) == {"a": 1}


def test_json_chat_strips_bare_fence():
    client = FakeClient(FakeResponse('```\n{"a": 2}\n```'))
    assert json_chat(client, "m", [], temperature=0.2, max_tokens=10) == {"a": 2}


def test_json_chat_retry_adds_system_prompt_and_succeeds():
    """Первый ответ — мусор: ровно один ретрай с доп. system, второй ответ — валидный JSON."""
    client = FakeClient(FakeResponse("мусор"))
    responses = [FakeResponse("мусор без json"), FakeResponse('{"recovered": true}')]

    def create(**kwargs):
        client.chat.completions.calls.append(kwargs)
        return responses.pop(0)

    client.chat.completions.create = create
    result = json_chat(client, "m", [{"role": "user", "content": "q"}],
                       temperature=0.2, max_tokens=10)
    assert result == {"recovered": True}
    calls = client.chat.completions.calls
    assert len(calls) == 2  # один ретрай
    assert calls[0]["messages"] == [{"role": "user", "content": "q"}]
    retry_messages = calls[1]["messages"]
    assert retry_messages[0] == {"role": "user", "content": "q"}
    assert retry_messages[-1] == {"role": "system", "content": RETRY_SYSTEM_PROMPT}


def test_json_chat_both_invalid_raises():
    client = FakeClient(FakeResponse("совсем не json"))
    with pytest.raises(InterviewError):
        json_chat(client, "m", [], temperature=0.2, max_tokens=10)
    assert len(client.chat.completions.calls) == 2  # ретрай ровно один


def test_json_chat_length_retries_with_doubled_budget():
    """Обрыв по length + мусор: ровно один ретрай с удвоенным бюджетом и без доп. system."""
    client = FakeClient(FakeResponse('{"обрыв', finish_reason="length"))
    responses = [
        FakeResponse('{"обрыв', finish_reason="length"),
        FakeResponse('{"recovered": true}', finish_reason="stop"),
    ]

    def create(**kwargs):
        client.chat.completions.calls.append(kwargs)
        return responses.pop(0)

    client.chat.completions.create = create
    result = json_chat(client, "m", [{"role": "user", "content": "q"}],
                       temperature=0.2, max_tokens=10)
    assert result == {"recovered": True}
    calls = client.chat.completions.calls
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 10
    assert calls[1]["max_tokens"] == 20  # budget*2 от первой попытки
    assert calls[1]["messages"] == [{"role": "user", "content": "q"}]


def test_json_chat_length_twice_raises():
    """Обрыв по length дважды (второй бюджет 2×) → InterviewError, ретрай ровно один."""
    client = FakeClient(FakeResponse('{"обрыв', finish_reason="length"))
    with pytest.raises(InterviewError):
        json_chat(client, "m", [], temperature=0.2, max_tokens=10)
    calls = client.chat.completions.calls
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 10
    assert calls[1]["max_tokens"] == 20


def test_json_chat_valid_json_with_length_no_retry():
    """Валидный JSON при finish_reason=length → возврат сразу, без ретрая."""
    client = FakeClient(FakeResponse('{"ok": true}', finish_reason="length"))
    assert json_chat(client, "m", [], temperature=0.2, max_tokens=10) == {"ok": True}
    assert len(client.chat.completions.calls) == 1


def test_strip_json_fences_variants():
    assert _strip_json_fences('{"a": 1}') == '{"a": 1}'
    assert _strip_json_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_json_fences('```\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_json_fences('  {"a": 1}  ') == '{"a": 1}'
