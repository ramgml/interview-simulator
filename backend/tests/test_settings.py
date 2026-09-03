"""Тесты настроек (T133): GET/PUT /api/settings, маскирование api_key, GET /api/settings/test.

Без сети, без живого LLM, без sleep (AGENTS.md): app.llm стабится monkeypatch-ом на уровне
sys.modules (модуля ещё нет в develop — мержится в T130), OpenAI-клиент — FakeClient.
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

# --- фикстуры ---


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


class FakeModels:
    """Стаб client.models: считает вызовы list()."""

    def __init__(self):
        self.list_calls = 0

    def list(self):
        self.list_calls += 1
        return []


class FakeClient:
    """Стаб openai.OpenAI: models.list() проходит, счётчик для assert-ов."""

    def __init__(self, on_create=None):
        self.models = FakeModels()
        self._on_create = on_create

    def __getattr__(self, name):
        if self._on_create is not None:
            self._on_create(name)
        raise AssertionError(f"неожиданное обращение к client.{name}")


@pytest.fixture()
def llm_stub(monkeypatch):
    """Подменяет sys.modules['app.llm'] до мержа T130: get_client возвращает FakeClient."""
    holder: dict[str, object] = {}
    module = types.ModuleType("app.llm")

    def get_client(s):
        fake = FakeClient()
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


# --- GET ---


def test_get_returns_seed_from_env_with_masked_api_key(client):
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["provider"] == "local"
    assert body["api_key"] is None  # сид: ключ не задан → null, не пустая строка
    assert body["model"] == "glm/glm-5.3-flash"
    assert set(body) == {
        "provider",
        "base_url",
        "api_key",
        "model",
        "whisper_model",
        "tts_voice",
        "updated_at",
    }


# --- PUT ---


def test_put_updates_fields(client):
    resp = client.put("/api/settings", json={"model": "gpt-4o", "tts_voice": "baya"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "gpt-4o"
    assert body["tts_voice"] == "baya"
    # нетронутые поля сохраняются
    assert body["provider"] == "local"
    assert client.get("/api/settings").json()["model"] == "gpt-4o"


def test_put_round_trip_masked_key_survives(client):
    """Round-trip: задали ключ → GET вернул '***'; PUT с '***' и пустым не перетирает."""
    assert client.put("/api/settings", json={"api_key": "sk-secret-123"}).status_code == 200
    assert client.get("/api/settings").json()["api_key"] == "***"

    client.put("/api/settings", json={"api_key": "***", "model": "m1"})
    client.put("/api/settings", json={"api_key": "", "model": "m2"})

    body = client.get("/api/settings").json()
    assert body["model"] == "m2"  # оба PUT прошли, поля применились
    assert body["api_key"] == "***"  # ключ всё ещё '***' снаружи — перезатирки не было
    row = client.get("/api/settings")  # хранимое значение проверяем напрямую в БД
    stored = _stored_api_key()
    assert stored == "sk-secret-123"
    assert row.json()["api_key"] == "***"


def _stored_api_key():
    """Прямой доступ к singleton-строке: наружу ключ не виден, проверяем хранилище."""
    from app.db import SessionFactory
    from app.models import Settings

    with SessionFactory() as db:
        return db.get(Settings, 1).api_key


# --- PUT: валидация ---


def test_put_invalid_provider_422(client):
    resp = client.put("/api/settings", json={"provider": "hybrid"})
    assert resp.status_code == 422
    # ничего не изменилось
    assert client.get("/api/settings").json()["provider"] == "local"


def test_put_no_fields_keeps_row(client):
    resp = client.put("/api/settings", json={})
    assert resp.status_code == 200
    assert resp.json()["provider"] == "local"


# --- GET /api/settings/test ---


def test_test_local_ok_with_stub(client, llm_stub):
    resp = client.get("/api/settings/test")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    assert llm_stub["client"].models.list_calls == 1
    snap = llm_stub["settings_snapshot"]
    assert snap["provider"] == "local"
    assert snap["base_url"] == "http://localhost:20128/v1"  # get_client получил строку из БД


def test_test_cloud_requires_filled_fields(client, llm_stub):
    client.put("/api/settings", json={"provider": "cloud"})
    resp = client.get("/api/settings/test")
    assert resp.status_code == 422
    assert "base_url" in resp.json()["detail"]
    assert "client" not in llm_stub  # get_client не вызывался


def test_test_cloud_filled_ok(client, llm_stub):
    client.put(
        "/api/settings",
        json={"provider": "cloud", "base_url": "https://api.example.com/v1",
              "api_key": "sk-live", "model": "glm-4"},
    )
    resp = client.get("/api/settings/test")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True}
    snap = llm_stub["settings_snapshot"]
    assert snap["provider"] == "cloud"
    assert snap["base_url"] == "https://api.example.com/v1"
    assert snap["api_key"] == "sk-live"  # в get_client уходит настоящий ключ, не маска
    assert snap["model"] == "glm-4"


def test_test_cloud_connection_error_502(client, llm_stub, monkeypatch):
    def boom(s):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(llm_module(), "get_client", boom)
    client.put(
        "/api/settings",
        json={"provider": "cloud", "base_url": "https://api.example.com/v1",
              "api_key": "sk-live", "model": "glm-4"},
    )
    resp = client.get("/api/settings/test")
    assert resp.status_code == 502
    assert "connection refused" in resp.json()["detail"]


def llm_module():
    return sys.modules["app.llm"]
