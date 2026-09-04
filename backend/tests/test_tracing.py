"""Тесты MLflow-трейсинга (T130): init, get_last_trace_id, log_session_run. Стаб mlflow."""

import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main as main_app_module
from app import tracing
from app.models import Session, Settings, Turn

class FakeMlflow:
    """Стаб mlflow-модуля: ловит init-вызовы, start_run, log_*; артефакты/трейсы — в атрибутах."""

    def __init__(self):
        self.init_calls: list[tuple] = []
        self.autolog_calls: list[dict] = []
        self.log_params_calls: list[dict] = []
        self.log_metrics_calls: list[dict] = []
        self.log_dict_calls: list[tuple] = []
        self.log_artifact_calls: list[str] = []
        self.transcript_contents: dict[str, str] = {}  # путь → содержимое на момент log_artifact
        self.start_run_calls: list[dict] = []
        self.last_trace_id: str | None = None
        self.fail_on = None  # имя метода, на котором бросать исключение

    def set_tracking_uri(self, uri):
        self.init_calls.append(("set_tracking_uri", uri))

    def set_experiment(self, name):
        self.init_calls.append(("set_experiment", name))

    def get_last_active_trace_id(self):
        if self.fail_on == "get_last_active_trace_id":
            raise RuntimeError("mlflow down")
        return self.last_trace_id

    def start_run(self, **kwargs):
        if self.fail_on == "start_run":
            raise RuntimeError("mlflow down")
        self.start_run_calls.append(kwargs)

        class Info:
            run_id = "fake-run-id-123"

        class Run:
            info = Info()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return Run()

    def log_params(self, params):
        if self.fail_on == "log_params":
            raise RuntimeError("mlflow down")
        self.log_params_calls.append(params)

    def log_artifact(self, local_path):
        self.log_artifact_calls.append(local_path)
        try:
            with open(local_path, encoding="utf-8") as f:
                self.transcript_contents[local_path] = f.read()
        except OSError:
            self.transcript_contents[local_path] = ""

    def log_metrics(self, metrics):
        if self.fail_on == "log_metrics":
            raise RuntimeError("mlflow down")
        self.log_metrics_calls.append(metrics)

    def log_dict(self, data, artifact_file):
        self.log_dict_calls.append((data, artifact_file))



class FakeOpenaiModule:
    """Стаб mlflow.openai: autolog ловится отдельно (в реальном модуле это submodule)."""

    def __init__(self):
        self.autolog_calls: list[dict] = []

    def autolog(self, **kwargs):
        self.autolog_calls.append(kwargs)


@pytest.fixture
def fake_settings_row():
    """Singleton-строка настроек для log_session_run (provider/model)."""
    return Settings(id=1, provider="local", model="glm/glm-5.3-flash")


@pytest.fixture
def fake_db(monkeypatch, fake_settings_row):
    """SessionFactory → стаб-сессия с db.get(Settings, 1) → строка настроек."""

    class FakeDb:
        def get(self, model, pk):
            assert model is Settings and pk == 1
            return fake_settings_row

    class FakeFactory:
        def __call__(self):
            return self

        def __enter__(self):
            return FakeDb()

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(tracing, "SessionFactory", FakeFactory())



@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = FakeMlflow()
    fake.openai = FakeOpenaiModule()
    fake.autolog_calls = fake.openai.autolog_calls  # один список для обоих стабов
    monkeypatch.setattr(tracing, "mlflow", fake)
    monkeypatch.setattr(tracing, "mlflow_openai", fake.openai)
    return fake


def make_session(**overrides) -> Session:
    defaults = dict(
        id="abc123",
        status="completed",
        position_title="Python-разработчик",
        seniority="middle",
        style="friendly",
        planned_questions=8,
        vacancy_text="вакансия",
        overall_score=7.5,
        duration_sec=120.5,
    )
    defaults.update(overrides)
    return Session(**defaults)


def make_turn(idx: int, role: str, text: str, trace_id: str | None = None) -> Turn:
    return Turn(session_id="abc123", idx=idx, role=role, text=text, llm_trace_id=trace_id)


# --- init_mlflow ---------------------------------------------------------------


def test_init_mlflow_makes_three_calls(fake_mlflow):
    tracing.init_mlflow()
    assert len(fake_mlflow.init_calls) == 2  # set_tracking_uri + set_experiment
    assert len(fake_mlflow.autolog_calls) == 1  # mlflow.openai.autolog
    kinds = [name for name, _ in fake_mlflow.init_calls]
    assert kinds == ["set_tracking_uri", "set_experiment"]
    assert fake_mlflow.init_calls[0][1] == tracing._repo_relative_sqlite_url(
        tracing.env.mlflow_tracking_uri
    )
    assert fake_mlflow.init_calls[1][1] == "interview-simulator"
    assert fake_mlflow.autolog_calls[0] == {"log_traces": True}


def test_init_mlflow_total_calls_is_three(fake_mlflow):
    """Контракт ARCHITECTURE §MLflow: ровно 3 вызова (tracking_uri, experiment, autolog)."""
    tracing.init_mlflow()
    total = len(fake_mlflow.init_calls) + len(fake_mlflow.autolog_calls)
    assert total == 3



# --- get_last_trace_id ---------------------------------------------------------



def test_get_last_trace_id_returns_id(fake_mlflow):
    fake_mlflow.last_trace_id = "trace-42"
    assert tracing.get_last_trace_id() == "trace-42"


def test_get_last_trace_id_none_is_not_error(fake_mlflow):
    fake_mlflow.last_trace_id = None
    assert tracing.get_last_trace_id() is None


def test_get_last_trace_id_swallows_mlflow_failure(fake_mlflow, caplog):
    fake_mlflow.fail_on = "get_last_active_trace_id"
    with caplog.at_level(logging.WARNING):
        assert tracing.get_last_trace_id() is None
    assert any("get_last_active_trace_id" in r.message for r in caplog.records)


# --- log_session_run -----------------------------------------------------------


def test_log_session_run_run_name_template(fake_mlflow):
    session = make_session(id="deadbeef", position_title="Python Backend Developer")
    run_id = tracing.log_session_run(session, [], {"competencies": []})
    assert run_id == "fake-run-id-123"
    assert fake_mlflow.start_run_calls[0]["run_name"] == "session-deadbeef-Python Backend Developer"


def test_log_session_run_run_name_position_truncated_to_30(fake_mlflow):
    title = "Очень длинное название вакансии превышающее лимит"
    session = make_session(id="sid", position_title=title)
    tracing.log_session_run(session, [], {"competencies": []})
    name = fake_mlflow.start_run_calls[0]["run_name"]
    assert name.startswith("session-sid-")
    assert name == f"session-sid-{session.position_title[:30]}"


def test_log_session_run_metrics_and_sanitize(fake_mlflow):
    session = make_session(overall_score=8.0)
    report = {
        "competencies": [
            {"name": "Алгоритмы", "score": 7},
            {"name": "Docker / Kubernetes", "score": 9},
            {"name": "Без оценки", "score": None},
        ]
    }
    tracing.log_session_run(session, [], report)
    metrics = fake_mlflow.log_metrics_calls[0]
    assert metrics["overall_score"] == 8.0
    assert metrics["score_алгоритмы"] == 7
    assert metrics["score_docker___kubernetes"] == 9
    assert "score_без_оценки" not in metrics  # null-балл не логируется


def test_log_session_run_params(fake_mlflow):
    session = make_session()
    turns = [make_turn(0, "interviewer", "Расскажите о себе")]
    tracing.log_session_run(session, turns, {"competencies": []})
    params = fake_mlflow.log_params_calls[0]
    assert params["planned_questions"] == 8
    assert params["seniority"] == "middle"
    assert params["style"] == "friendly"
    assert params["turns_count"] == 1
    assert params["duration_sec"] == 120.5
    # provider/model читаются из singleton-настроек, не из сессии
    assert params["provider"] != "unknown"
    assert params["model"] != "unknown"


def test_log_session_run_artifacts(fake_mlflow, tmp_path):
    session = make_session()
    report = {"overall_score": 7.5, "competencies": [{"name": "X", "score": 5}]}
    turns = [make_turn(0, "interviewer", "В1"), make_turn(1, "candidate", "О1")]
    run_id = tracing.log_session_run(session, turns, report)
    assert run_id == "fake-run-id-123"

    assert len(fake_mlflow.log_dict_calls) == 1
    data, artifact_file = fake_mlflow.log_dict_calls[0]
    assert artifact_file == "report.json"
    assert data == report
    assert len(fake_mlflow.log_artifact_calls) == 1
    transcript_path = fake_mlflow.log_artifact_calls[0]
    text = fake_mlflow.transcript_contents[transcript_path]
    assert "Интервьюер" in text and "В1" in text
    assert "Кандидат" in text and "О1" in text


def test_log_session_run_tags_session_id_and_trace_ids_json_list(fake_mlflow):
    session = make_session()
    turns = [
        make_turn(0, "interviewer", "В1", trace_id="trace-a"),
        make_turn(1, "candidate", "О1", trace_id=None),
        make_turn(2, "interviewer", "В2", trace_id="trace-b"),
    ]
    tracing.log_session_run(session, turns, {"competencies": []})
    tags = fake_mlflow.start_run_calls[0]["tags"]
    assert tags["session_id"] == "abc123"
    assert json.loads(tags["llm_trace_ids"]) == ["trace-a", "trace-b"]


def test_log_session_run_error_returns_none_without_raising(fake_mlflow, fake_db):
    fake_mlflow.fail_on = "log_metrics"
    result = tracing.log_session_run(make_session(), [], {"competencies": []})
    assert result is None


def test_log_session_run_db_unavailable_returns_none(fake_mlflow, monkeypatch):
    def boom():
        raise RuntimeError("db gone")

    monkeypatch.setattr(tracing, "SessionFactory", boom)
    result = tracing.log_session_run(make_session(), [], {"competencies": []})
    assert result is None


# --- repo-relative sqlite URI --------------------------------------------------


def test_init_mlflow_anchors_relative_sqlite_uri_to_repo_root(fake_mlflow):
    """Относительный sqlite-URI из env якорится к корню репо, а не к cwd процесса."""
    tracing.init_mlflow()
    repo_root = Path(__file__).resolve().parents[2]
    expected = "sqlite:///" + str((repo_root / "data/mlflow.db").resolve())
    assert fake_mlflow.init_calls[0] == ("set_tracking_uri", expected)
    uri = fake_mlflow.init_calls[0][1]
    assert uri.startswith("sqlite:////")  # абсолютный путь: 4 слэша
    assert uri.endswith("/data/mlflow.db")


def test_repo_relative_sqlite_url_absolute_and_non_sqlite_pass_through():
    """Абсолютный sqlite-URI и не-sqlite URI проходят через хелпер без изменений."""
    assert tracing._repo_relative_sqlite_url("sqlite:////tmp/x.db") == "sqlite:////tmp/x.db"
    assert tracing._repo_relative_sqlite_url("http://host:5100") == "http://host:5100"


# --- lifespan ------------------------------------------------------------------


def test_lifespan_calls_init_db_then_init_mlflow(monkeypatch):
    """Порядок старта: init_db, затем init_mlflow."""
    calls: list[str] = []
    monkeypatch.setattr(main_app_module, "init_db", lambda: calls.append("init_db"))
    monkeypatch.setattr(main_app_module, "init_mlflow", lambda: calls.append("init_mlflow"))
    with TestClient(main_app_module.app):
        pass
    assert calls == ["init_db", "init_mlflow"]


def test_lifespan_runs_real_init_mlflow_with_stub_mlflow(monkeypatch):
    """Lifespan реально вызывает init_mlflow: якоренный URI + autolog(log_traces=True)."""
    class StubOpenai:
        def __init__(self):
            self.autolog_calls: list[dict] = []

        def autolog(self, **kwargs):
            self.autolog_calls.append(kwargs)

    class StubMlflow:
        def __init__(self):
            self.init_calls: list[tuple] = []

        def set_tracking_uri(self, uri):
            self.init_calls.append(("set_tracking_uri", uri))

        def set_experiment(self, name):
            self.init_calls.append(("set_experiment", name))

    stub = StubMlflow()
    stub.openai = StubOpenai()
    stub.autolog_calls = stub.openai.autolog_calls
    monkeypatch.setattr(tracing, "mlflow", stub)
    monkeypatch.setattr(tracing, "mlflow_openai", stub.openai)
    with TestClient(main_app_module.app):
        pass
    repo_root = Path(__file__).resolve().parents[2]
    expected = "sqlite:///" + str((repo_root / "data/mlflow.db").resolve())
    assert ("set_tracking_uri", expected) in stub.init_calls
    assert stub.autolog_calls == [{"log_traces": True}]
