"""Роутер /api/models (T159): список id моделей с OpenAI-совместимого провайдера.

Контракт — docs/ARCHITECTURE.md §API: GET {base_url}/models через клиент настроек;
cloud без base_url/api_key → 422; ошибка соединения/таймаут → 502. Ответ
`{"models": [str]}` — отсортированный список уникальных id для datalist настроек.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session as DbSession

from app.db import get_db
from app.errors import InterviewError
from app.routers.settings import get_settings_row

router = APIRouter(prefix="/api/models")

# Короткий таймаут списка моделей: datalist — вспомогательная функция настроек,
# не interview-ход; долгое зависание провайдера здесь недопустимо (T159).
MODELS_TIMEOUT_S = 10


@router.get("")
def list_models(db: DbSession = Depends(get_db)) -> dict[str, list[str]]:
    """GET /api/models: id моделей провайдера для datalist страницы настроек."""
    row = get_settings_row(db)
    if row.provider == "cloud" and not (row.base_url and row.api_key):
        raise HTTPException(
            status_code=422,
            detail="Заполните облачный провайдер в настройках: base_url и api_key",
        )
    # ЛЕНИВЫЙ импорт: как в settings.py /test — позволяет тестам стабить app.llm
    from app.llm import get_client

    try:
        client = get_client(row)
        page = client.with_options(timeout=MODELS_TIMEOUT_S).models.list()
    except InterviewError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # ошибка соединения (timeout/refused/dns) → 502 с текстом
        raise HTTPException(status_code=502, detail=f"Провайдер недоступен: {exc}") from exc
    return {"models": sorted({str(m.id) for m in page.data})}
