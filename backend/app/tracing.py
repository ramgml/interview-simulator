"""MLflow-трейсинг: init, autolog, session-run. Ошибка MLflow не роняет сессию."""

import json
import logging
import re
import tempfile
from pathlib import Path

import mlflow
from mlflow import openai as mlflow_openai

from app.config import settings as env
from app.db import SessionFactory
from app.models import Session, Settings, Turn

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "interview-simulator"


def init_mlflow() -> None:
    """tracking_uri из env, эксперимент, autolog openai-трейсов."""
    mlflow.set_tracking_uri(env.mlflow_tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow_openai.autolog(log_traces=True)


def get_last_trace_id() -> str | None:
    """Id последнего активного трейса для turns.llm_trace_id; нет трейса — не ошибка."""
    try:
        return mlflow.get_last_active_trace_id()
    except Exception:
        logger.warning("mlflow.get_last_active_trace_id() failed", exc_info=True)
        return None


def _score_key(name: str) -> str:
    """Метрика score_<name>: \\W → _, lower."""
    return "score_" + re.sub(r"\W", "_", name).lower()


def log_session_run(session: Session, turns: list[Turn], report: dict) -> str | None:
    """MLflow-run сессии: метрики/params/артефакты/tags; run_id или None при ошибке."""
    try:
        run_name = f"session-{session.id}-{session.position_title[:30]}"
        tags = {"session_id": session.id}
        trace_ids = [t.llm_trace_id for t in turns if t.llm_trace_id]
        tags["llm_trace_ids"] = json.dumps(trace_ids)

        metrics: dict[str, float] = {}
        if session.overall_score is not None:
            metrics["overall_score"] = session.overall_score
        for comp in report.get("competencies", []):
            if comp.get("score") is not None:
                metrics[_score_key(comp["name"])] = float(comp["score"])

        # provider/model — из singleton-настроек (в sessions их нет); DB-ошибка → внешний except
        with SessionFactory() as db:
            row = db.get(Settings, 1)
        params = {
            "provider": row.provider if row else "",
            "model": row.model if row else "",
            "planned_questions": session.planned_questions,
            "seniority": session.seniority or "",
            "style": session.style,
            "turns_count": len(turns),
            "duration_sec": session.duration_sec if session.duration_sec is not None else "",
        }
        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_dict(report, "report.json")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "transcript.md"
                path.write_text(_transcript_md(session, turns), encoding="utf-8")
                mlflow.log_artifact(str(path))
            return run.info.run_id
    except Exception:
        logger.exception("log_session_run failed for session %s", session.id)
        return None


def _transcript_md(session: Session, turns: list[Turn]) -> str:
    lines = [f"# {session.position_title}", ""]
    for t in turns:
        who = "Интервьюер" if t.role == "interviewer" else "Кандидат"
        lines.append(f"**{who}:** {t.text}")
        lines.append("")
    return "\n".join(lines)
