"""Тесты CORS (T177): allow_origins берётся из конфига (env CORS_ORIGINS, CSV-формат).

Без сети и живых сервисов (AGENTS.md): preflight OPTIONS замыкается в CORSMiddleware,
lifespan не запускается (TestClient без with). app.main пересобирается importlib.reload-ом
с подменённым app.config.settings (Settings(_env_file=None) — тест не зависит от .env);
после теста модуль восстанавливается с исходным settings.
"""

import importlib

import pytest
from fastapi.testclient import TestClient

from app import config as config_module
from app import main as main_module
from app.config import Settings

FOREIGN_ORIGIN = "http://evil.example"


@pytest.fixture()
def rebuild_main(monkeypatch):
    """Пересобирает app.main с заданным CORS_ORIGINS (None — без env); восстанавливает модуль."""
    real_settings = config_module.settings

    def _rebuild(cors_env: str | None) -> TestClient:
        if cors_env is None:
            monkeypatch.delenv("CORS_ORIGINS", raising=False)
        else:
            monkeypatch.setenv("CORS_ORIGINS", cors_env)
        monkeypatch.setattr(config_module, "settings", Settings(_env_file=None))
        importlib.reload(main_module)
        return TestClient(main_module.app)

    yield _rebuild

    monkeypatch.setattr(config_module, "settings", real_settings)
    importlib.reload(main_module)


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/health",
        headers={"Origin": origin, "Access-Control-Request-Method": "GET"},
    )


# --- env CORS_ORIGINS ---


def test_preflight_allows_origins_from_env(rebuild_main):
    client = rebuild_main("http://localhost:3000,http://localhost:5173")
    for origin in ("http://localhost:3000", "http://localhost:5173"):
        resp = _preflight(client, origin)
        assert resp.status_code == 200
        assert resp.headers["access-control-allow-origin"] == origin


def test_preflight_rejects_foreign_origin_without_header(rebuild_main):
    client = rebuild_main("http://localhost:3000,http://localhost:5173")
    resp = _preflight(client, FOREIGN_ORIGIN)
    assert resp.status_code == 400
    assert "access-control-allow-origin" not in resp.headers


# --- дефолт без env ---


def test_preflight_without_env_uses_default_origin(rebuild_main):
    client = rebuild_main(None)
    resp = _preflight(client, "http://localhost:3000")
    assert resp.status_code == 200
    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"


# --- разбор CSV в конфиге ---


def test_cors_origins_parses_csv_and_strips_spaces(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", " http://a.example , http://b.example ")
    parsed = Settings(_env_file=None).cors_origins
    assert parsed == ["http://a.example", "http://b.example"]
