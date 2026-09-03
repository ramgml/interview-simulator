"""Консолидация make models (T141): whisper → silero, реюз app.download_silero."""

from app import download_models
from app.download_models import main


def test_main_runs_whisper_then_silero(monkeypatch, caplog):
    calls = []
    monkeypatch.setattr(
        download_models, "download_whisper", lambda: calls.append("whisper")
    )
    monkeypatch.setattr(
        download_models, "download_silero_model", lambda: calls.append("silero")
    )
    with caplog.at_level("INFO"):
        main()
    assert calls == ["whisper", "silero"]
    assert calls[-1] == "silero"


def test_silero_step_reuses_download_silero(monkeypatch):
    """download_silero_model реюзнит download() из app.download_silero."""
    seen = {}

    def fake_download():
        seen["called"] = True
        seen["path"] = __import__("pathlib").Path("/tmp/fake-v4_ru.pt")
        return seen["path"]

    monkeypatch.setattr(download_models, "download", fake_download)
    result = download_models.download_silero_model()
    assert seen["called"] is True
    assert result == seen["path"]
