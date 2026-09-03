"""Тесты STT: decode_audio, transcribe (стаб), singleton, роут /api/stt.

Без сети, без живых моделей, без сна (AGENTS.md п. 37): WhisperModel не загружается
с диска — везде стабы/monkeypatch.
"""

import math
from io import BytesIO

import av
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import app.stt as stt
from app.errors import EmptyTranscript
from app.main import app


@pytest.fixture()
def client(monkeypatch):
    """TestClient с чистым singleton-состоянием на каждый тест."""
    monkeypatch.setattr(stt, "_model", None)
    with TestClient(app) as c:
        yield c


def make_wav(seconds: float = 0.3, freq: float = 440.0, rate: int = 44100) -> bytes:
    """Синтетический стерео-wav (тон) — для проверки декода/resample."""
    t = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)
    buf = BytesIO()
    sf.write(buf, stereo, rate, format="WAV", subtype="FLOAT")
    return buf.getvalue()


class FakeSegment:
    """Поля как у faster-whisper Segment (AGENTS.md п. 20 — стаб вместо живой модели)."""

    def __init__(self, text: str, avg_logprob: float):
        self.text = text
        self.avg_logprob = avg_logprob


class FakeModel:
    """Стаб WhisperModel: возвращает заданные сегменты, считает вызовы transcribe."""

    def __init__(self, segments: list[FakeSegment] | None = None):
        self.segments = segments if segments is not None else [FakeSegment("привет", -0.2)]
        self.transcribe_calls = 0
        self.last_audio = None

    def transcribe(self, audio, **kwargs):
        self.transcribe_calls += 1
        self.last_audio = audio
        self.last_kwargs = kwargs
        return iter(self.segments), None


# --- decode_audio ---


def _wav_bytes(seconds: float, freq: float, rate: int) -> bytes:
    t = np.linspace(0.0, seconds, int(rate * seconds), endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)
    buf = BytesIO()
    sf.write(buf, stereo, rate, format="WAV")
    return buf.getvalue()


def test_decode_audio_mono_16k_float32():
    data = _wav_bytes(seconds=0.3, freq=440.0, rate=16000)
    audio = stt.decode_audio(data)
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert len(audio) == 4800  # 0.3 с × 16 кГц


def test_decode_audio_contract_resample_to_mono_16k():
    """Контракт: любой контейнер → mono 16k float32 правильной длительности."""
    audio = stt.decode_audio(_wav_bytes(seconds=1.0, freq=880.0, rate=44100))
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    # Ресемплинг 44.1k → 16k даёт ±50 сэмплов на секунду (границы блоков фильтра).
    assert abs(len(audio) - 16000) <= 100
    # Реальный звук, не тишина: есть энергия.
    assert float(np.abs(audio).max()) > 0.01


def test_decode_audio_webm_opus_container():
    """Синтетический webm/opus декодируется тем же контрактом (MediaRecorder-путь)."""
    buf = BytesIO()
    with av.open(buf, "w", format="webm") as out:
        stream = out.add_stream("libopus", rate=48000, layout="stereo")
        t = np.linspace(0.0, 0.02, 960, endpoint=False)
        tone = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        packed = np.empty(1920, dtype=np.float32)
        packed[0::2] = tone
        packed[1::2] = tone
        frame = av.AudioFrame.from_ndarray(packed.reshape(1, -1), format="flt", layout="stereo")
        frame.sample_rate = 48000
        for packet in stream.encode(frame):
            out.mux(packet)
        for packet in stream.encode(None):
            out.mux(packet)
    audio = stt.decode_audio(buf.getvalue())
    assert audio.dtype == np.float32
    assert audio.ndim == 1
    assert 300 <= len(audio) <= 340  # 960 сэмплов 48k → 16k = 320; opus-праймер ±несколько


def test_transcribe_silence_raises_empty_transcript(monkeypatch):
    """Тишина декодируется, но сегментов нет — EmptyTranscript (LLM не вызывается)."""
    fake = FakeModel([])
    monkeypatch.setattr(stt, "_model", fake)
    t = np.zeros(16000, dtype=np.float32)
    buf = BytesIO()
    sf.write(buf, t, 16000, format="WAV")
    with pytest.raises(EmptyTranscript):
        stt.transcribe(buf.getvalue(), "ru", "Backend")
    assert fake.transcribe_calls == 1  # модель вызвана, но сегментов не вернула


def test_transcribe_returns_text_and_confidence(monkeypatch):
    fake = FakeModel([FakeSegment("  рассказ о  опыте ", -0.2), FakeSegment("с 2019 года", -0.4)])
    monkeypatch.setattr(stt, "_model", fake)
    text, confidence = stt.transcribe(_wav_bytes(0.2, 440.0, 16000), "ru", "Backend")
    assert text == "рассказ о  опыте с 2019 года"
    assert confidence == pytest.approx(math.exp((-0.2 + -0.4) / 2))


def test_transcribe_passes_whisper_params_and_prompt(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(stt, "_model", fake)
    stt.transcribe(_wav_bytes(0.2, 440.0, 16000), "en", "Go разработчик")
    kw = fake.last_kwargs
    assert kw["language"] == "en"
    assert kw["beam_size"] == 5
    assert kw["vad_filter"] is True
    assert kw["initial_prompt"].startswith(
        "Собеседование по программированию. Технологии: Go разработчик"
    )
    assert kw["initial_prompt"].endswith("Big-O.")


def test_transcribe_empty_text_raises_empty_transcript(monkeypatch):
    fake = FakeModel([FakeSegment("", -0.1), FakeSegment("  ", -0.1)])
    monkeypatch.setattr(stt, "_model", fake)
    with pytest.raises(EmptyTranscript):
        stt.transcribe(_wav_bytes(0.2, 440.0, 16000), "ru", "Backend")


# --- singleton ---


def test_get_model_is_singleton(monkeypatch):
    calls = {"count": 0}

    class CountingModel:
        def __init__(self, *args, **kwargs):
            calls["count"] += 1

    monkeypatch.setattr(stt, "WhisperModel", CountingModel)
    monkeypatch.setattr(stt, "_model", None)
    m1 = stt.get_model()
    m2 = stt.get_model()
    assert calls["count"] == 1
    assert m1 is m2


def test_get_model_device_profile(monkeypatch):
    """WHISPER_DEVICE=cpu → cpu+int8; auto без CUDA → cpu+int8 (AGENTS.md п. 34)."""
    seen: list[dict] = []

    class RecordingModel:
        def __init__(self, name, device, compute_type):
            seen.append({"name": name, "device": device, "compute_type": compute_type})

    monkeypatch.setattr(stt, "WhisperModel", RecordingModel)
    monkeypatch.setattr(stt, "_model", None)
    monkeypatch.setattr(stt.settings, "whisper_device", "cpu")
    stt.get_model()
    assert seen == [{"name": "large-v3-turbo", "device": "cpu", "compute_type": "int8"}]

    import app.stt as stt_mod

    monkeypatch.setattr(stt_mod, "_model", None)
    monkeypatch.setattr(stt_mod.settings, "whisper_device", "auto")
    monkeypatch.setattr(stt_mod, "_resolve_device", lambda d: ("cpu", "int8"))
    stt_mod.get_model()
    assert seen[-1] == {"name": "large-v3-turbo", "device": "cpu", "compute_type": "int8"}


# --- роут ---


def test_stt_route_200(client, monkeypatch):
    monkeypatch.setattr(
        stt,
        "transcribe",
        lambda data, language, position_title: ("опыт пять лет", 0.9),
    )
    resp = client.post("/api/stt", files={"audio": ("a.wav", make_wav(), "audio/wav")})
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"text": "опыт пять лет", "confidence": 0.9}


def test_stt_route_empty_transcript_422(client, monkeypatch):
    def raise_empty(data, language, position_title):
        raise EmptyTranscript()

    monkeypatch.setattr(stt, "transcribe", raise_empty)
    resp = client.post("/api/stt", files={"audio": ("a.wav", make_wav(), "audio/wav")})
    assert resp.status_code == 422
    assert resp.json()["detail"] == "Речь не распознана"


def test_stt_route_requires_audio_field(client):
    resp = client.post("/api/stt")
    assert resp.status_code == 422
