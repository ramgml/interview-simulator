"""FastAPI-приложение: CORS :3000, lifespan (create_all + сид), GET /health.

Роутеры: audio (/api/stt) — T134, /api/tts — T135, models (/api/models) — T159,
settings (/api/settings GET/PUT/test) — T133, sessions (/api/sessions, /api/progress) — T131/T132.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import init_db
from app.routers import audio, models, sessions, settings
from app.stt import router as stt_router

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="interview-simulator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stt_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(sessions.router)
app.include_router(settings.router)
app.include_router(models.router)
app.include_router(audio.router)
