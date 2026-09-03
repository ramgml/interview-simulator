"""LLM-слой: get_client local/cloud, chat/json_chat. Единственная точка переключения провайдера."""

import json
import logging

from openai import OpenAI

from app.config import settings as env
from app.errors import InterviewError
from app.models import Settings

logger = logging.getLogger(__name__)

RETRY_SYSTEM_PROMPT = "Верни только валидный JSON без пояснений"


def get_client(s: Settings) -> OpenAI:
    """OpenAI-клиент из строки настроек; provider='local' → env-URL, 'cloud' → DB-строка.

    Читает DB на каждый вызов — смена провайдера без рестарта (ARCHITECTURE §LLM-слой).
    """
    if s.provider == "local":
        return OpenAI(base_url=env.local_llm_base_url, api_key="sk-local")
    if s.provider == "cloud":
        base_url = (s.base_url or "").strip()
        api_key = (s.api_key or "").strip()
        if not base_url or not api_key:
            raise InterviewError("Заполните облачный провайдер в настройках")
        return OpenAI(base_url=base_url, api_key=api_key)
    raise InterviewError(f"Unknown provider: {s.provider}")


def chat(
    client: OpenAI, model: str, messages: list[dict], *, temperature: float, max_tokens: int
) -> tuple[str, dict]:
    """Один ход LLM → (content, usage-dict). Ошибка провайдера → InterviewError (наружу 502)."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as exc:
        logger.error("LLM call failed: %s", exc)
        raise InterviewError("Сервис LLM недоступен") from exc
    content = resp.choices[0].message.content or ""
    usage = {}
    if resp.usage is not None:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "total_tokens": resp.usage.total_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
    return content, usage


def _strip_json_fences(text: str) -> str:
    """Срез markdown-заборов ```json … ``` / ``` …; голый JSON проходит как есть."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped.removeprefix("```json").removeprefix("```")
    return body.removesuffix("```").strip()


def json_chat(
    client: OpenAI, model: str, messages: list[dict], *, temperature: float, max_tokens: int
) -> dict:
    """chat → срез ```-заборов → json.loads; один ретрай с доп. system-промптом; далее ошибка."""
    last_error: ValueError | None = None
    for attempt in range(2):
        attempt_messages = list(messages)
        if attempt == 1:
            attempt_messages.append({"role": "system", "content": RETRY_SYSTEM_PROMPT})
        content, _ = chat(
            client, model, attempt_messages, temperature=temperature, max_tokens=max_tokens
        )
        try:
            return json.loads(_strip_json_fences(content))
        except ValueError as exc:
            last_error = exc
            logger.warning("LLM returned invalid JSON (attempt %d): %s", attempt + 1, exc)
    raise InterviewError("LLM returned invalid JSON") from last_error
