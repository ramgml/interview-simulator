"""Тесты /api/models (T159): список id моделей провайдера для datalist настроек.

Стаб app.llm через monkeypatch sys.modules (как test_settings.py): без сети, sleep
и живого LLM. Покрывают: cloud заполнен → 200 c id-списком и снапшотом настроек,
cloud без base_url/api_key → 422 и get_client не вызван, ошибка соединения → 502,
InterviewError → 502, local → 200 с локальной строкой настроек.
"""

import sys
import types

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.models import Base

CLOUD_PUT = {
    "provider": "cloud",
    "base_url": "https://api.example.com/v1",
    "api_key": "sk-live",
    "model": "glm-4",
}


# --- фикстуры (по образцу test_settings.py) ---


@pytest.fixture()
def client(monkeypatch, tmp_path):
    """TestClient c изолированной sqlite в tmp и сидом settings id=1 (init_db из lifespan)."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'app.db'}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    monkeypatch.setattr("app.db.engine", engine)
    monkeypatch.setattr("app.db.SessionFactory", sessionmaker(bind=engine, expire_on_commit=False))
    with TestClient(app) as c:
        yield c


class FakeModelsPage:
    """Стаб страницы client.models.list(): .data — объекты с .id."""

    def __init__(self, ids):
        self.data = [types.SimpleNamespace(id=model_id) for model_id in ids]


class FakeModels:
    """Стаб client.models: list() возвращает страницу-стаб, считает вызовы."""

    def __init__(self, ids):
        self._ids = ids
        self.list_calls = 0

    def list(self):
        self.list_calls += 1
        return FakeModelsPage(self._ids)


class FakeClient:
    """Стаб openai.OpenAI: with_options(timeout=...) → клиент с models.list(); счётчики."""

    def __init__(self, ids, on_create=None):
        self.models = FakeModels(ids)
        self._on_create = on_create
        self.with_options_calls = 0
        self.with_options_timeout = None

    def with_options(self, **kwargs):
        self.with_options_calls += 1
        self.with_options_timeout = kwargs.get("timeout")
        return self


@pytest.fixture()
def llm_stub(monkeypatch):
    """Подменяет sys.modules['app.llm']: get_client возвращает FakeClient со списком id."""
    holder: dict[str, object] = {}
    module = types.ModuleType("app.llm")

    def get_client(s):
        fake = FakeClient(["b", "a", "b", "c"])
        holder["client"] = fake
        holder["settings_snapshot"] = {
            "provider": s.provider,
            "base_url": s.base_url,
            "api_key": s.api_key,
            "model": s.model,
        }
        return fake

    module.get_client = get_client
    monkeypatch.setitem(sys.modules, "app.llm", module)
    return holder


def llm_module():
    return sys.modules["app.llm"]


# --- GET /api/models ---


def test_models_cloud_filled_ok(client, llm_stub):
    """Cloud заполнен → 200, отсортированные уникальные id; get_client получил строку из БД."""
    client.put("/api/settings", json=CLOUD_PUT)
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert resp.json() == {"models": ["a", "b", "c"]}  # уникальные, отсортированные
    assert llm_stub["client"].models.list_calls == 1
    assert llm_stub["client"].with_options_calls == 1
    assert llm_stub["client"].with_options_timeout == 10  # короткий таймаут списка
    snap = llm_stub["settings_snapshot"]
    assert snap["provider"] == "cloud"
    assert snap["base_url"] == "https://api.example.com/v1"
    assert snap["api_key"] == "sk-live"  # в get_client уходит настоящий ключ, не маска
    assert snap["model"] == "glm-4"


def test_models_cloud_requires_base_url_and_key(client, llm_stub):
    """Cloud без base_url/api_key → 422; get_client не вызывался."""
    client.put("/api/settings", json={"provider": "cloud"})
    resp = client.get("/api/models")
    assert resp.status_code == 422
    assert "base_url" in resp.json()["detail"]
    assert "api_key" in resp.json()["detail"]
    assert "client" not in llm_stub  # get_client не вызывался


def test_models_connection_error_502(client, llm_stub, monkeypatch):
    """Ошибка соединения (timeout/refused/dns) → 502 «Провайдер недоступен: …»."""

    def boom(s):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(llm_module(), "get_client", boom)
    client.put("/api/settings", json=CLOUD_PUT)
    resp = client.get("/api/models")
    assert resp.status_code == 502
    assert resp.json()["detail"] == "Провайдер недоступен: connection refused"


def test_models_interview_error_502(client, llm_stub, monkeypatch):
    """InterviewError из llm-слоя → 502 с текстом исключения."""
    from app.errors import InterviewError

    def boom(s):
        raise InterviewError("boom")

    monkeypatch.setattr(llm_module(), "get_client", boom)
    client.put("/api/settings", json=CLOUD_PUT)
    resp = client.get("/api/models")
    assert resp.status_code == 502
    assert resp.json()["detail"] == "boom"


def test_models_local_ok_with_stub(client, llm_stub):
    """provider local → 200 (стаб); get_client получил локальную строку настроек."""
    resp = client.get("/api/models")
    assert resp.status_code == 200
    assert resp.json() == {"models": ["a", "b", "c"]}
    snap = llm_stub["settings_snapshot"]
    assert snap["provider"] == "local"
    assert snap["base_url"] == "http://localhost:20128/v1"  # локальная строка из БД
