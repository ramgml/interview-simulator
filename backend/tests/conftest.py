"""Pytest: пакет app импортируется из backend/ (uv run pytest запускается из backend/)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import tracing  # noqa: E402


@pytest.fixture(autouse=True)
def fake_mlflow_module(monkeypatch):
    """Ни один тест не касается реального mlflow/диска: стабы вместо модулей в app.tracing."""

    class FakeMlflow:
        def __init__(self):
            self.tracking_uris: list[str] = []
            self.experiments: list[str] = []

        def set_tracking_uri(self, uri):
            self.tracking_uris.append(uri)

        def set_experiment(self, name):
            self.experiments.append(name)

    class FakeOpenai:
        def __init__(self):
            self.autolog_calls: list[dict] = []

        def autolog(self, **kwargs):
            self.autolog_calls.append(kwargs)

    monkeypatch.setattr(tracing, "mlflow", FakeMlflow())
    monkeypatch.setattr(tracing, "mlflow_openai", FakeOpenai())
