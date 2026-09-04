"""Pydantic-схемы API (контракт — docs/ARCHITECTURE.md §API)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# Создание сессии: {vacancy_text, seniority?, language?, style?, planned_questions?}
Style = Literal["friendly", "strict", "realistic"]


class SessionCreate(BaseModel):
    vacancy_text: str = Field(min_length=1)
    seniority: Literal["junior", "middle", "senior", "lead"] | None = None
    language: str = "ru"
    style: Style = "realistic"
    planned_questions: int = Field(default=8, ge=5, le=12)


class SessionOut(BaseModel):
    id: str
    created_at: datetime
    status: str
    position_title: str
    seniority: str | None
    language: str
    style: str
    planned_questions: int


class SessionCreated(BaseModel):
    id: str



# Настройки: singleton id=1; api_key наружу маскируется '***' (ARCHITECTURE §API, §Модель данных)
SettingsProvider = Literal["local", "cloud"]


class SettingsRead(BaseModel):
    provider: str
    base_url: str
    api_key: str | None
    model: str
    whisper_model: str
    tts_voice: str
    updated_at: datetime


class SettingsUpdate(BaseModel):
    """Все поля опциональны: api_key='***' или пустой → хранимый не перетирается (роутер)."""

    provider: SettingsProvider | None = None
    base_url: str | None = None
    api_key: str | None = None
    model: str | None = None
    whisper_model: str | None = None
    tts_voice: str | None = None


class AnswerText(BaseModel):
    """Текстовый ход кандидата (альтернатива multipart audio)."""

    text: str = Field(min_length=1)


class AnswerOut(BaseModel):
    """Ответ интервьюера на ход кандидата (ARCHITECTURE §API /answer)."""

    transcript: str | None
    question_text: str | None
    done: bool
    action: str


# Озвучка: {text, voice?} → audio/wav (ARCHITECTURE §API, §Голос)
TtsVoice = Literal["aidar", "baya", "kseniya", "xenia", "eugene", "random"]


class TtsRequest(BaseModel):
    text: str = Field(min_length=1)
    voice: TtsVoice = "kseniya"
