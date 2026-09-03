"""Роутер /api/settings (T133): GET/PUT singleton-настроек + GET /test — проверка соединения.

Контракт — docs/ARCHITECTURE.md §API: api_key наружу маскируется '***' (задан) / null (пуст);
PUT все поля опциональны, api_key='***' или пустой хранимый не перетирает; /test — проверка
соединения с LLM-провайдером (local — лёгкий вызов client.models.list(), cloud — требует
заполненные base_url/api_key/model).
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session as DbSession

from app.db import get_db, init_db
from app.errors import InterviewError
from app.models import Settings
from app.schemas import SettingsRead, SettingsUpdate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/settings")

# Маска api_key наружу: задан → '***', не раскрываем значение (AGENTS.md «Безопасность»)
API_KEY_MASK = "***"


@router.get("")
def read_settings(db: DbSession = Depends(get_db)) -> SettingsRead:
    row = _get_row(db)
    return SettingsRead(
        provider=row.provider,
        base_url=row.base_url,
        api_key=API_KEY_MASK if row.api_key else None,
        model=row.model,
        whisper_model=row.whisper_model,
        tts_voice=row.tts_voice,
        updated_at=row.updated_at,
    )


@router.put("")
def update_settings(payload: SettingsUpdate, db: DbSession = Depends(get_db)) -> SettingsRead:
    row = _get_row(db)
    for name, value in payload.model_dump(exclude_unset=True).items():
        # '***' — маска с GET; пустой/'' — очистка без явного ключа; хранимый не меняем
        if name == "api_key" and (value == API_KEY_MASK or value == ""):
            continue
        setattr(row, name, value)
    row.updated_at = datetime.now(timezone.utc)
    db.commit()
    return SettingsRead(
        provider=row.provider,
        base_url=row.base_url,
        api_key=API_KEY_MASK if row.api_key else None,
        model=row.model,
        whisper_model=row.whisper_model,
        tts_voice=row.tts_voice,
        updated_at=row.updated_at,
    )


@router.get("/test")
def test_settings(db: DbSession = Depends(get_db)) -> dict[str, object]:
    """GET /api/settings/test: local — лёгкий вызов client.models.list(); cloud — требует
    заполненные base_url/api_key/model (иначе 422 с внятным текстом); ошибка соединения → 502."""
    # ЛЕНИВЫЙ импорт: app.llm появится в T130; верхний уровень сломал бы сборку тестов до его мержа
    from app.llm import get_client

    row = _get_row(db)
    if row.provider == "cloud" and not (row.base_url and row.api_key and row.model):
        raise HTTPException(
            status_code=422,
            detail="Заполните облачный провайдер в настройках: base_url, api_key и model",
        )
    try:
        client = get_client(row)
        client.models.list()
    except InterviewError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # ошибка соединения (timeout/refused/dns) → 502 с текстом
        raise HTTPException(status_code=502, detail=f"Провайдер недоступен: {exc}") from exc
    return {"ok": True}


def _get_row(db: DbSession) -> Settings:
    row = db.execute(select(Settings).where(Settings.id == 1)).scalar_one_or_none()
    if row is None:
        init_db()
        row = db.execute(select(Settings).where(Settings.id == 1)).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail="Настройки не инициализированы")
    return row
