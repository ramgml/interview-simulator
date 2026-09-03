"""STT: ленивый singleton faster-whisper, PyAV-декод в mono 16k float32, POST /api/stt.

Контракт — docs/ARCHITECTURE.md «Голос» и «API». Тяжёлые шаги (decode/transcribe)
не блокируют event loop — выполняются в executor-е на уровне роутера (AGENTS.md п. 33).
"""

import asyncio
import logging
import math
from io import BytesIO

import av
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from faster_whisper import WhisperModel

from app.config import settings
from app.errors import EmptyTranscript

logger = logging.getLogger(__name__)

# Точная формулировка из docs/ARCHITECTURE.md «Голос» (константа модуля, AGENTS.md п. 21).
_PROMPT_TECH = (
    "Python, JavaScript, TypeScript, React, Docker, Kubernetes, "
    "PostgreSQL, gRPC, микросервисы, CI/CD, алгоритмы, Big-O."
)

router = APIRouter(prefix="/api")

_model: WhisperModel | None = None


def _resolve_device(device: str) -> tuple[str, str]:
    """WHISPER_DEVICE=auto → (cuda, int8_float16) если CUDA доступен, иначе (cpu, int8).

    Явные значения cpu/cuda проходят как есть. Занятая GPU обнаруживается самой
    faster-whisper при загрузке весов — тогда она сама падает на cpu
    (см. get_model: инициализация с cuda → fallback cpu без проброса ошибки).
    """
    if device == "auto":
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "int8_float16"
        except Exception:  # noqa: BLE001 - отсутствие/поломка CUDA-стека = cpu
            logger.info("cuda check failed, falling back to cpu")
        return "cpu", "int8"
    if device == "cuda":
        return "cuda", "int8_float16"
    return "cpu", "int8"


def get_model() -> WhisperModel:
    """Ленивый singleton WhisperModel; инициализация на каждый запрос запрещена."""
    global _model
    if _model is None:
        device, compute_type = _resolve_device(settings.whisper_device)
        try:
            _model = WhisperModel(settings.whisper_model, device=device, compute_type=compute_type)
        except Exception:
            if device == "cpu":
                raise
            # Занятая/недоступная GPU при auto — корректный fallback на cpu (AGENTS.md п. 34).
            logger.warning("whisper init on %s failed, falling back to cpu+int8", device)
            _model = WhisperModel(settings.whisper_model, device="cpu", compute_type="int8")
        logger.info(
            "whisper model initialized: name=%s device=%s compute_type=%s",
            settings.whisper_model,
            device,
            compute_type,
        )
    return _model


def decode_audio(data: bytes) -> np.ndarray:
    """Байты любого контейнера (wav/webm/opus) → mono 16 kHz float32."""
    with av.open(BytesIO(data)) as container:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=16000)
        chunks: list[np.ndarray] = []
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                arr = out.to_ndarray()
                # flt-планарный mono даёт форму (1, n) — к плоскому 1-D.
                chunks.append(arr.reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)


def transcribe(data: bytes, language: str, position_title: str) -> tuple[str, float]:
    """Аудио → (текст, confidence=exp(avg_logprob)). Пустой текст → EmptyTranscript."""
    audio = decode_audio(data)
    if len(audio) == 0:
        raise EmptyTranscript()
    segments, _info = get_model().transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=True,
        initial_prompt="Собеседование по программированию. Технологии: "
        + position_title
        + ", "
        + _PROMPT_TECH,
    )
    texts: list[str] = []
    logprobs: list[float] = []
    for seg in segments:
        texts.append(seg.text.strip())
        logprobs.append(seg.avg_logprob)
    text = " ".join(t for t in texts if t).strip()
    if not text:
        raise EmptyTranscript()
    confidence = math.exp(sum(logprobs) / len(logprobs))
    return text, confidence


@router.post("/stt")
async def stt(audio: UploadFile = File(...)) -> dict[str, float | str]:
    """Отладочный роут: multipart audio → {text, confidence}."""
    data = await audio.read()
    try:
        text, confidence = await asyncio.get_running_loop().run_in_executor(
            None, transcribe, data, "ru", ""
        )
        return {"text": text, "confidence": confidence}
    except EmptyTranscript:
        raise HTTPException(status_code=422, detail="Речь не распознана") from None
