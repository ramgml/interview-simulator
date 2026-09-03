"""Роутер /api/tts (T135). POST {text, voice?} → audio/wav (ARCHITECTURE §API)."""

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from app import tts
from app.errors import InterviewError, TtsEmptyText
from app.schemas import TtsRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/api/tts")
def synthesize_speech(req: TtsRequest) -> Response:
    try:
        audio = tts.synthesize(req.text, voice=req.voice)
    except TtsEmptyText as exc:
        raise _http_error(422, str(exc)) from exc
    except InterviewError as exc:
        raise _http_error(502, str(exc)) from exc
    return Response(content=audio, media_type="audio/wav")


def _http_error(status: int, detail: str) -> HTTPException:
    logger.warning("tts: %d — %s", status, detail)
    return HTTPException(status_code=status, detail=detail)
