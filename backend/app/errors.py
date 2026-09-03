"""Доменные ошибки интервью. Наружу — типизированно (ARCHITECTURE §API)."""


class InterviewError(Exception):
    """Ошибки сценария интервью (LLM/провайдер/конфигурация). Наружу → 502."""


class EmptyTranscript(Exception):
    """STT вернул пустой текст. Наружу → 422 «Речь не распознана» (LLM не дёргаем)."""


class NotFoundError(Exception):
    """Сущность (сессия/настройки) не найдена. Наружу → 404."""
