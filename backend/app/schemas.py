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


# Озвучка: {text, voice?} → audio/wav (ARCHITECTURE §API, §Голос)
TtsVoice = Literal["aidar", "baya", "kseniya", "xenia", "eugene", "random"]


class TtsRequest(BaseModel):
    text: str = Field(min_length=1)
    voice: TtsVoice = "kseniya"
