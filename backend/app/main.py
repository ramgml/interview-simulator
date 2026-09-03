"""FastAPI-приложение: CORS :3000, lifespan (create_all + сид), GET /health.

Роутер audio (/api/stt) подключён в T134; sessions/settings — в T130+.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import init_db
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
