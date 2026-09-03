"""Прогрев silero v4_ru (T135): скачивание артефакта в torch-hub-кэш.

`make models` вызывает `python -m app.download_models` — whisper+silero (T134/T135,
финальная консолидация — при мерже). Этот модуль — silero-часть: артефакт
`v4_ru.pt` (torch.package, 40 МБ) с models.silero.ai кладётся в кэш torch.hub,
где его находит app.tts._load_via_artifact. torch.hub-путь silero требует
omegaconf и репозитория snakers4/silero-models; артефакт от них не зависит.

Запуск: `cd backend && uv run python -m app.download_silero`
"""

import logging
import pathlib
import urllib.request

import torch

logger = logging.getLogger(__name__)

# Канонический URL артефакта (models.yml silero-models, tts_models/ru/v4_ru).
ARTIFACT_URL = "https://models.silero.ai/models/tts/ru/v4_ru.pt"


def artifact_path() -> pathlib.Path:
    """Путь артефакта в кэше torch.hub (директория создаётся при необходимости)."""
    path = pathlib.Path(torch.hub.get_dir()) / "silero" / "v4_ru.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def download() -> pathlib.Path:
    """Скачать v4_ru.pt, если артефакта ещё нет; вернуть путь (идемпотентно)."""
    path = artifact_path()
    if path.is_file() and path.stat().st_size > 0:
        logger.info("download_silero: artifact exists: %s", path)
        return path
    logger.info("download_silero: downloading %s -> %s", ARTIFACT_URL, path)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(ARTIFACT_URL, tmp)
    tmp.replace(path)
    logger.info("download_silero: done (%d bytes)", path.stat().st_size)
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    download()
