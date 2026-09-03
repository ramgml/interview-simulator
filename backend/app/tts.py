"""TTS Silero: ленивый singleton, озвучка по предложениям (ARCHITECTURE §Голос).

Модель строго CPU: загрузка выполняется с map_location='cpu', а сигнатура
apply_tts не принимает device — синтез физически не может занять VRAM
(проверяется тестом test_model_stays_on_cpu).
"""

import io
import logging
import pathlib
import re
import threading
from typing import Callable

import soundfile as sf
import torch

from app.errors import InterviewError, TtsEmptyText

logger = logging.getLogger(__name__)

# Контракт синтеза: wav 24 кГц (docs/ARCHITECTURE.md §Голос).
SAMPLE_RATE = 24000

# Свыше 300 символов текст режется на предложения и синтезируется по частям.
MAX_CHUNK_CHARS = 300

# Голоса silero v4_ru; random — случайный голос, генерируется моделью.
VOICES = ("aidar", "baya", "kseniya", "xenia", "eugene", "random")

_HUB_SPEC = ("snakers4/silero-models", "silero_tts")

_model: torch.nn.Module | None = None
_lock = threading.Lock()


def _load_via_artifact() -> torch.nn.Module:
    """Артефакт v4_ru.pt (torch.package) — считается скачанным download_silero."""
    from torch.package import PackageImporter

    path = pathlib.Path(torch.hub.get_dir()) / "silero" / "v4_ru.pt"
    if not path.is_file():
        raise FileNotFoundError(f"silero artifact not found: {path}")
    logger.info("tts: loading silero from artifact %s (cpu)", path)
    return PackageImporter(str(path)).load_pickle("tts_models", "model", map_location="cpu")


def _load_via_hub() -> torch.nn.Module:
    """Публичный путь silero (нужны omegaconf и репо snakers4/silero-models)."""
    logger.info("tts: loading silero via torch.hub %s (cpu)", _HUB_SPEC)
    model, _ = torch.hub.load(
        *_HUB_SPEC, "silero_tts", language="ru", speaker="v4_ru", trust_repo=True
    )
    return model


def _load_via_local() -> torch.nn.Module:
    """Фолбэк W200: репозиторий torch.hub недоступен — source='local' из кэша."""
    logger.info("tts: loading silero via torch.hub source='local' (cpu)")
    model, _ = torch.hub.load(
        _HUB_SPEC[0], "silero_tts", source="local", language="ru", speaker="v4_ru", trust_repo=True
    )
    return model


def _load_fallbacks() -> torch.nn.Module:
    """Цепочка загрузки: артефакт download_silero → torch.hub → source='local'."""
    attempts: tuple[tuple[str, Callable[[], torch.nn.Module]], ...] = (
        ("artifact", _load_via_artifact),
        ("torch.hub", _load_via_hub),
        ("local", _load_via_local),
    )
    errors: list[str] = []
    for name, load in attempts:
        try:
            return load()
        except Exception as exc:  # следующий фолбэк; наружу — только если исчерпаны
            errors.append(f"{name}: {exc}")
            logger.warning("tts: %s load failed: %s", name, exc)
    raise InterviewError("Не удалось загрузить TTS-модель (silero); попробуйте `make models`")


def get_model() -> torch.nn.Module:
    """Ленивый singleton (AGENTS.md «Производительность и GPU»): одна загрузка на процесс."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                _model = _load_fallbacks()
                logger.info("tts: silero ready (device=cpu, singleton)")
    return _model


def reset_model() -> None:
    """Сброс singleton (для тестов)."""
    global _model
    with _lock:
        _model = None


def split_sentences(text: str) -> list[str]:
    """Резка на предложения по [.!?…]+\\s; хвост без пунктуации — отдельная часть."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?…])\s+", text.strip()) if p.strip()]
    return parts or [text.strip()]


def _synthesize_chunk(model: torch.nn.Module, text: str, voice: str) -> torch.Tensor:
    """Один вызов apply_tts: Tensor float32 mono на cpu (24 кГц)."""
    try:
        audio = model.apply_tts(text=text, speaker=voice, sample_rate=SAMPLE_RATE)
    except Exception as exc:
        raise InterviewError(f"Ошибка синтеза речи (silero): {exc}") from exc
    return audio.detach().to("cpu").to(torch.float32).contiguous()


def synthesize(text: str, voice: str = "kseniya") -> bytes:
    """Текст → wav bytes (mono, SAMPLE_RATE). Длинный текст — по предложениям, один wav."""
    if not text or not text.strip():
        raise TtsEmptyText("Пустой текст для озвучки")
    if voice not in VOICES:
        raise InterviewError(f"Неизвестный голос: {voice}")

    model = get_model()
    chunks = split_sentences(text)
    parts = [_synthesize_chunk(model, chunk, voice) for chunk in chunks]
    audio = parts[0] if len(parts) == 1 else torch.cat(parts)

    buf = io.BytesIO()
    sf.write(buf, audio.numpy(), SAMPLE_RATE, format="WAV", subtype="PCM_16")
    logger.info("tts: synthesized %d chunk(s), %.1f sec", len(parts), len(audio) / SAMPLE_RATE)
    return buf.getvalue()
