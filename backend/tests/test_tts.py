"""Тесты TTS (T135): singleton, резка по предложениям, роут /api/tts, CPU-гигиена.

Живая модель silero не грузится — torch.hub/PackageImporter стабятся (AGENTS.md «Тесты»).
"""

import io

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import tts
from app.errors import InterviewError, TtsEmptyText
from app.main import app
from app.schemas import TtsRequest


class FakeSilero:
    """Стаб silero: помнит вызовы apply_tts, отдаёт волны разной длины."""

    def __init__(self):
        self.calls: list[dict] = []
        self.counter = 0

    def apply_tts(self, text, speaker, sample_rate):
        self.calls.append({"text": text, "speaker": speaker, "sample_rate": sample_rate})
        import torch

        self.counter += 1
        return torch.zeros(2400 * self.counter, dtype=torch.float32)  # 0.1 c × n


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    fake = FakeSilero()
    monkeypatch.setattr(tts, "_load_fallbacks", lambda: fake)
    tts.reset_model()
    yield fake
    tts.reset_model()


def _read_wav(data: bytes):
    audio, rate = sf.read(io.BytesIO(data))
    return audio, rate


# --- синтез ------------------------------------------------------------------


def test_short_text_single_call(fake_model):
    data = tts.synthesize("Проверка связи.", voice="kseniya")
    assert len(fake_model.calls) == 1
    audio, rate = _read_wav(data)
    assert rate == tts.SAMPLE_RATE == 24000
    assert audio.ndim == 1  # mono


def test_long_text_splits_by_sentences_and_joins(fake_model):
    text = (
        "Первое предложение подробно описывает техническую задачу и контекст проекта целиком. "
        "Второе предложение заметно длиннее первого и добавляет новые детали к описанию. "
        "Третье предложение подводит промежуточный итог обсуждения и закрывает мысль. "
        "Четвёртое нужно, чтобы суммарный размер уверенно превысил границу в триста символов."
    )
    assert len(text) > 300
    data = tts.synthesize(text, voice="baya")
    assert len(fake_model.calls) == 4  # резка по [.!?…]\s — четыре предложения
    assert max(len(call["text"]) for call in fake_model.calls) <= tts.MAX_CHUNK_CHARS
    audio, rate = _read_wav(data)
    assert rate == 24000
    # один файл, длительность ≈ сумме частей (сэмплы конкатенируются)
    total_samples = sum(2400 * (i + 1) for i in range(len(fake_model.calls)))
    assert len(audio) == total_samples


def test_oversized_sentence_splits_within_limit(fake_model):
    sentence = " ".join(["слово"] * 150) + "."  # одно предложение без точки внутри, 900 символов
    assert len(sentence) >= 600
    data = tts.synthesize(sentence, voice="kseniya")
    texts = [call["text"] for call in fake_model.calls]
    assert len(texts) >= 2  # сверхдлинное предложение синтезируется частями
    assert max(len(t) for t in texts) <= tts.MAX_CHUNK_CHARS
    assert " ".join(texts) == sentence  # резка по словам не теряет содержимое
    audio, _ = _read_wav(data)
    total_samples = sum(2400 * (i + 1) for i in range(len(texts)))
    assert len(audio) == total_samples


def test_single_huge_word_hard_split(fake_model):
    word = "а" * 700  # одиночное слово длиннее границы — жёсткая резка по 300
    data = tts.synthesize(word, voice="kseniya")
    texts = [call["text"] for call in fake_model.calls]
    assert len(texts) == 3  # 300 + 300 + 100
    assert max(len(t) for t in texts) <= tts.MAX_CHUNK_CHARS
    assert "".join(texts) == word  # жёсткая резка сохраняет всё слово
    audio, _ = _read_wav(data)
    total_samples = sum(2400 * (i + 1) for i in range(len(texts)))
    assert len(audio) == total_samples


def test_voice_passed_as_speaker(fake_model):
    tts.synthesize("Текст.", voice="eugene")
    assert all(call["speaker"] == "eugene" for call in fake_model.calls)


def test_empty_and_whitespace_text_typed_error():
    with pytest.raises(TtsEmptyText):
        tts.synthesize("")
    with pytest.raises(TtsEmptyText):
        tts.synthesize("   \n\t ")


def test_unknown_voice_typed_error():
    with pytest.raises(InterviewError):
        tts.synthesize("Текст.", voice="no_such_voice")


def test_synthesis_output_stays_cpu(fake_model):
    tts.synthesize("Текст.", voice="kseniya")
    assert all("device" not in call for call in fake_model.calls)  # apply_tts без device
    assert fake_model.calls


# --- singleton ---------------------------------------------------------------


def test_singleton_loads_model_once(monkeypatch):
    loads = {"n": 0}

    def loader():
        loads["n"] += 1
        return FakeSilero()

    monkeypatch.setattr(tts, "_load_fallbacks", loader)
    first = tts.get_model()
    second = tts.get_model()
    assert loads["n"] == 1
    assert first is second


def test_artifact_loader_pins_cpu_map_location(monkeypatch, tmp_path):
    recorded = {}

    class SpyImporter:
        def __init__(self, path):
            recorded["path"] = path

        def load_pickle(self, name, key, map_location=None):
            recorded["map_location"] = map_location
            return FakeSilero()

    hub_dir = tmp_path / "hub" / "silero"
    hub_dir.mkdir(parents=True)
    (hub_dir / "v4_ru.pt").write_bytes(b"stub")
    monkeypatch.setattr(tts.torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
    monkeypatch.setattr("torch.package.PackageImporter", SpyImporter)
    model = tts._load_via_artifact()
    assert recorded["map_location"] == "cpu"  # загрузка строго на CPU
    assert isinstance(model, FakeSilero)


# --- роут --------------------------------------------------------------------


def _tiny_wav_bytes() -> bytes:
    import torch

    buf = io.BytesIO()
    sf.write(buf, torch.zeros(2400).numpy(), 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_route_post_tts_returns_wav(monkeypatch):
    monkeypatch.setattr(tts, "synthesize", lambda text, voice: _tiny_wav_bytes())
    client = TestClient(app)
    resp = client.post("/api/tts", json={"text": "Проверка связи"})
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    assert resp.content[:4] == b"RIFF"


def test_route_empty_text_422_not_500():
    client = TestClient(app)
    # пустая строка отсекается схемой
    assert client.post("/api/tts", json={"text": ""}).status_code == 422
    # whitespace проходит схему, но отсекается типизированной TtsEmptyText → тоже 422
    resp = client.post("/api/tts", json={"text": "   "})
    assert resp.status_code == 422
    assert resp.json()["detail"] == "Пустой текст для озвучки"


def test_route_synthesis_failure_502(monkeypatch):
    def boom(text, voice):
        raise InterviewError("Ошибка синтеза речи (silero): взрыв")

    monkeypatch.setattr(tts, "synthesize", boom)
    client = TestClient(app)
    resp = client.post("/api/tts", json={"text": "Проверка связи"})
    assert resp.status_code == 502


def test_tts_request_schema_validates_voice_and_text():
    with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
        TtsRequest(text="")
    assert TtsRequest(text="ok", voice="random").voice == "random"
    with pytest.raises(Exception):  # noqa: B017 — pydantic ValidationError
        TtsRequest(text="ok", voice="vendor")


# --- резка -------------------------------------------------------------------


def test_split_sentences_edges():
    assert tts.split_sentences("Привет. Мир!") == ["Привет.", "Мир!"]
    assert tts.split_sentences("Без точки в конце") == ["Без точки в конце"]
    assert tts.split_sentences("Вопрос? Восклицание! Многоточие… Конец") == [
        "Вопрос?",
        "Восклицание!",
        "Многоточие…",
        "Конец",
    ]
    assert tts.split_sentences("Тире — внутри. Предложение №2, с запятыми.") == [
        "Тире — внутри.",
        "Предложение №2, с запятыми.",
    ]
