"""Доступ к БД: engine из settings.DATABASE_URL, create_all и сид настроек на старте."""

import logging
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, func, select
from sqlalchemy.orm import Session as DbSession
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.models import Base, Settings

logger = logging.getLogger(__name__)

# .env лежит в корне репозитория (каркас T128); make-цели запускаются из backend/.
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _repo_relative_sqlite_url(url: str) -> str:
    """Якорит относительный sqlite-путь (sqlite:///data/app.db) к корню репозитория."""
    prefix = "sqlite:///"
    if url.startswith(prefix) and not url.startswith(prefix + "/"):
        return prefix + str((_REPO_ROOT / url[len(prefix):]).resolve())
    return url


def ensure_data_dir(url: str) -> None:
    """SQLite не создаёт директорию под файл БД сама — делаем это до первого коннекта."""
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)


def _sqlite_connect_pragmas(dbapi_connection, _connection_record) -> None:
    """Прагмы каждого sqlite-соединения: WAL + busy_timeout (T155).

    WAL: читатели не блокируют писателя и наоборот; длинная LLM-задача больше не держит
    БД для параллельных записей. busy_timeout=10000: конкурентная запись дождётся лока
    вместо мгновенного `database is locked` (дефолт python-sqlite3 — 5с; удвоен с запасом
    на ретраи внутри одной транзакции). Прагмы персистентны только для journal_mode
    (файл), busy_timeout — на соединение, поэтому ставится на каждое.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=10000")
    cursor.close()


def make_engine(database_url: str | None = None) -> Engine:
    """Engine из settings.DATABASE_URL; sqlite-пути якорит к корню репозитория."""
    url = _repo_relative_sqlite_url(database_url or settings.database_url)
    ensure_data_dir(url)
    is_sqlite = url.startswith("sqlite")
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False} if is_sqlite else {},
        pool_pre_ping=True,
    )
    if is_sqlite:
        event.listens_for(engine, "connect")(_sqlite_connect_pragmas)
    return engine


engine = make_engine()
SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)


def get_db():
    db = SessionFactory()
    try:
        yield db
    finally:
        db.close()


def seed_settings(db: DbSession) -> None:
    """Идемпотентный сид singleton-настроек из env: если row id=1 есть — не трогаем.

    Локальные дефолты провайдера из env-конфига; секреты (cloud api_key) из env не пишем.
    """
    stmt = select(func.count()).select_from(Settings).where(Settings.id == 1)
    exists = db.execute(stmt).scalar_one()
    if exists:
        return
    row = Settings(
        id=1,
        provider="local",
        base_url=settings.local_llm_base_url,
        api_key=None,
        model=settings.local_llm_model,
        whisper_model=settings.whisper_model,
        tts_voice=settings.tts_voice,
    )
    db.add(row)
    db.commit()
    logger.info("settings row seeded from env (id=1)")


def init_db() -> None:
    """create_all + сид. Безопасно вызывать на каждом старте."""
    Base.metadata.create_all(engine)
    with SessionFactory() as db:
        seed_settings(db)
