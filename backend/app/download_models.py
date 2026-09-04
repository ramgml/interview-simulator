"""Предскачивание голосовых моделей: whisper + silero (T134/T135, консолидация T141).

Вызывается make-целью: cd backend && uv run python -m app.download_models
Последовательные шаги: whisper → silero; каждый идемпотентен (already cached —
пропускается), отдельная логика silero живёт в app/download_silero.py.
"""
import logging
from pathlib import Path

from app.download_silero import download
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Имя = значение faster_whisper.utils._MODELS["large-v3-turbo"] (repo Systran/... не существует:
# официальный ct2-конверсии турбо нет у Systran, faster-whisper маппит турбо на mobiuslabsgmbh,
# который редиректит на dropbox-dash — snapshot_download идёт по редиректу).
WHISPER_MODEL_REPO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"


def download_whisper() -> str:
    """Снапшот ct2-whisper large-v3-turbo (~1.6 ГБ) в кэш HF; повтор — already cached."""
    path = snapshot_download(WHISPER_MODEL_REPO)
    logger.info("whisper snapshot ready: %s", path)
    return path


def download_silero_model() -> Path:
    """Silero v4_ru: переиспользование silero-шага из app.download_silero."""
    path = download()
    logger.info("silero artifact ready: %s", path)
    return path


def main() -> None:
    """Оба шага последовательно: whisper → silero; ошибка шага падает наружу."""
    download_whisper()
    download_silero_model()


if __name__ == "__main__":
    main()
