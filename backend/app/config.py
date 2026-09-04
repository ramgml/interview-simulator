"""Конфигурация приложения: pydantic-settings, читает .env в корне репозитория."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# .env лежит в корне репозитория (каркас T128); backend запускается из backend/.
_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM: локальный провайдер (omniroute-совместимый gateway)
    local_llm_base_url: str = "http://localhost:20128/v1"
    local_llm_model: str = "glm/glm-5.3-flash"

    # LLM: облачный провайдер (значения хранятся в DB settings после настройки)
    cloud_llm_base_url: str = ""
    cloud_llm_api_key: str = ""
    cloud_llm_model: str = ""

    # LLM: бюджеты генерации PLAN/TURN/EVAL (max_tokens). На reasoning-моделях (напр.
    # glm-5.3-flash на Zhipu) reasoning тоже расходует этот бюджет — поднимай, если JSON
    # приходит пустым/обрезанным (finish_reason=length).
    plan_max_tokens: int = 4000
    turn_max_tokens: int = 1000
    eval_max_tokens: int = 4000

    # Трейсинг
    mlflow_tracking_uri: str = "sqlite:///data/mlflow.db"

    # Голос
    whisper_model: str = "large-v3-turbo"
    whisper_device: str = "auto"
    tts_voice: str = "kseniya"

    # БД
    database_url: str = "sqlite:///data/app.db"


settings = Settings()
