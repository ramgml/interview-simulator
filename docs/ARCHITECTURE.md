# ARCHITECTURE — Interview Simulator

> Техническая архитектура, контракты и верификация MVP. Первоисточник — черновик плана MVP (решения подтверждены владельцем 2026-09-02, удалён после миграции; история — wiki Orenda W200). Продукт — `docs/PRD.md`; изменчивое окружение и порт-реестр — wiki W200 (см. AGENTS.md «Источники истины»).

## Дерево репозитория

```
interview-simulator/            (monorepo, main → origin)
├── backend/                    FastAPI :8100, uv venv (python 3.12)
│   ├── pyproject.toml          fastapi uvicorn[standard] openai mlflow faster-whisper
│   │                           torch (CPU) soundfile av sqlalchemy pydantic-settings
│   │                           python-multipart; dev: pytest ruff
│   └── app/
│       ├── main.py             FastAPI, CORS :3000, create_all, init_mlflow()
│       ├── config.py           pydantic-settings, .env
│       ├── db.py               engine sqlite:///data/app.db
│       ├── models.py           ORM: settings(id=1), sessions, turns
│       ├── schemas.py          pydantic-схемы API
│       ├── llm.py              get_client + chat + json_chat, провайдер local/cloud
│       ├── stt.py              faster-whisper singleton + decode_audio (PyAV)
│       ├── tts.py              silero singleton → wav bytes
│       ├── interviewer.py      промпты PLAN/TURN, STYLE_PROMPTS, state-machine хода
│       ├── evaluator.py        промпт EVAL, финальный отчёт (скоринг+разбор+план)
│       ├── tracing.py          mlflow init, autolog, session run (метрики+артефакты)
│       ├── errors.py           InterviewError, EmptyTranscript
│       └── routers/{models,sessions,settings,audio}.py
├── frontend/                   Next.js 16 (TS, Tailwind v4, app router, npm)
│   ├── components/ui/*         shadcn-компоненты (npx shadcn@latest add ...)
│   ├── components/{Recorder,AudioQueue,ReportView,ProgressView}.tsx
│   └── app/
│       ├── page.tsx                        новая сессия + история + вкладка «Прогресс»
│       ├── session/[id]/page.tsx           живое интервью
│       ├── session/[id]/report/page.tsx    отчёт
│       ├── settings/page.tsx               настройки провайдера/голоса
│       └── layout.tsx                      шрифты, Toaster
├── data/                       (gitignore) app.db, mlflow.db
├── docs/  PRD.md  ARCHITECTURE.md  example-vacancy.md
├── Makefile  .env.example  .gitignore  README.md
```

## Поток данных

Браузер (MediaRecorder `audio/webm;codecs=opus`, hold-to-talk) → `POST /answer` → PyAV-декод в mono 16k float32 → faster-whisper (CUDA int8_float16, vad_filter, язык из сессии, initial_prompt с техтерминами) → LLM-ход (OpenAI SDK, mlflow-autolog трейс) → JSON-решение followup/next/finish → Silero озвучка **по предложениям** (фронт режет текст, `POST /api/tts` на каждое, очередь `<audio>`) → финал: evaluator → report JSON → MLflow run + SQLite. Прогресс между сессиями агрегируется из SQLite (report_json) на бэке.

## Модель данных (SQLAlchemy)

- `settings` — singleton row `id=1`: `provider` (`local|cloud`), `base_url`, `api_key` (nullable), `model`, `whisper_model`, `tts_voice`, `updated_at`. Сид из env при первом старте (`provider=local`).
- `sessions`: `id` (uuid4 hex pk), `created_at`, `status` (`created|in_progress|completed|failed`), `position_title`, `seniority` (nullable), `language` (default `ru`), `style` (`friendly|strict|realistic`, default `realistic`), `planned_questions` (default 8), `vacancy_text`, `plan_json` (nullable), `report_json` (nullable), `overall_score` (nullable), `mlflow_run_id` (nullable), `error` (nullable), `started_at`, `completed_at`, `duration_sec` (nullable).
- `turns`: `id` autoinc pk, `session_id` FK, `idx`, `role` (`interviewer|candidate`), `text`, `stt_confidence`, `llm_trace_id`, `latency_ms`, `created_at`.
- Аудио не персистится (приватность, см. PRD non-goals) — в отчёте только текст.

## LLM-слой (`app/llm.py`)

- `get_client(s) -> openai.OpenAI`: `provider=='local'` → `base_url=env.LOCAL_LLM_BASE_URL`, `api_key='sk-local'`; `'cloud'` → `base_url/api_key/model` из DB (обязательны — иначе `InterviewError('Заполните облачный провайдер в настройках')`). Читает DB на каждый вызов — смена провайдера без рестарта.
- `chat(client, model, messages, *, temperature, max_tokens) -> tuple[str, dict]` — `(content, usage)`; `usage` всегда содержит `finish_reason`; недоступность endpoint → `InterviewError` → 502 наружу.
- `resolve_model(s) -> str`: имя модели для LLM-вызовов: `local` → `env.local_llm_model`, `cloud` → DB `settings.model` (пустая — `InterviewError('Заполните модель облачного провайдера в настройках')`); стиль сессии именем модели не является (T152).
- `json_chat(...) -> dict`: `chat` → срез ```json-заборов → `json.loads`; при ошибке один ретрай: при `finish_reason=length` — с удвоенным `max_tokens` (без доп. system-промпта), иначе с доп. system-промптом «Верни только валидный JSON без пояснений»; вторая ошибка → `InterviewError`.

## Голос

- **STT** (`app/stt.py`): ленивый singleton `WhisperModel(name, device, compute_type)`; `WHISPER_DEVICE=auto` → cuda если есть, иначе cpu; compute: cuda → `int8_float16`, cpu → `int8`. `decode_audio(data) -> np.ndarray`: `av.open(BytesIO)` → mono 16k float32. `transcribe(data, language, position_title) -> tuple[str, float]`: `beam_size=5`, `vad_filter=True`, `initial_prompt="Собеседование по программированию. Технологии: "+position_title+", Python, JavaScript, TypeScript, React, Docker, Kubernetes, PostgreSQL, gRPC, микросервисы, CI/CD, алгоритмы, Big-O."`; confidence=exp(avg_logprob). Пустой текст → `EmptyTranscript`.
- **TTS** (`app/tts.py`): ленивый singleton `torch.hub.load('snakers4/silero-models','silero_tts',language='ru',speaker='v4_ru')` (torch CPU). `synthesize(text, voice) -> bytes`: >300 символов — резать по предложениям, склеивать soundfile-ом в один wav; `apply_tts(speaker=voice, sample_rate=24000)` → wav bytes.
- `make models` = `python -m app.download_models`: `huggingface_hub.snapshot_download('Systran/faster-whisper-large-v3-turbo')` + torch.hub-загрузка silero.

## Интервьюер (`app/interviewer.py`) — LLM-протокол

Стили `STYLE_PROMPTS`: `friendly` («дружелюбный, поддерживающий»), `strict` («сухой тон, давит follow-ups, вскрывает слабости, стресс-интервью»), `realistic` («как живое собеседование в средней продуктовой компании»).

- `build_plan(client, session) -> dict`, system: «Опытный технический интервьюер. По вакансии составь план интервью. Верни ТОЛЬКО JSON: `{"position_title": str, "competencies": [str], "rounds": [{"type": "technical"|"behavioral"|"algorithms"|"system_design", "questions": [{"topic", "question", "competency"}]}]}`. Раунды и вопросы зависят от вакансии; покрой ключевые требования; уровень кандидата: {seniority}; язык: {language}» + вакансия (обрезка 8000 симв.). Через `json_chat`.
- `conduct_turn(client, session, transcript_turns) -> dict`, system: «Ты ведёшь собеседование один на один. Стиль: {style_prompt}. Решение: ответ неполный — один follow-up по теме; тема исчерпана — следующий вопрос из плана (не заданный ранее); все темы покрыты — заверши. Верни ТОЛЬКО JSON: `{"action": "followup"|"next_question"|"finish", "text": str, "covered_topic": str|null}`» + план + полный транскрипт (до 24 ходов).
- Fallback при `InterviewError` в `conduct_turn`: следующий незаданный вопрос из `plan_json` (action=next_question); если пусто — action=finish «Спасибо, интервью завершено». Деградация без падения сессии.

## Оценщик (`app/evaluator.py`)

`evaluate(client, session) -> dict`, temperature 0.2. JSON-схема отчёта: `{"overall_score": 0-10, "competencies": [{"name","score" 0-10,"comment"}], "turn_feedback": [{"turn_idx", "question", "answer", "score", "good", "missed", "strong_answer"}], "strengths": [str], "weaknesses": [str], "plan": [{"topic","action","resources_hint"}], "verdict": str, "hire_recommendation": "strong_yes|yes|no|strong_no"}`.

Fallback при двойной ошибке JSON: `{"degraded": true, "verdict": <нарратив целиком>, пустые списки, null-баллы}` — сессия всё равно завершается отчётом.

## MLflow (`app/tracing.py`)

- `init_mlflow()`: `set_tracking_uri`, `set_experiment('interview-simulator')`, `mlflow.openai.autolog(log_traces=True)`; после `chat` — `mlflow.get_last_active_trace_id()` → `turns.llm_trace_id`.
- `log_session_run(session, turns, report)`: run `session-{sid}-{position_title[:30]}`; метрики `overall_score`, `score_<name>` (sanitize `\W`→`_`, lower); params provider/model/planned_questions/seniority/style/turns_count/duration_sec; артефакты `report.json` (`log_dict`), `transcript.md` (`log_artifact`); tags `session_id`, `llm_trace_ids` (json-список). `mlflow_run_id` → sessions. Ошибка MLflow не роняет сессию (try/except+лог).

## API

| Метод/путь | Назначение |
|---|---|
| `GET /health` | liveness |
| `POST /api/sessions` | создать: `{vacancy_text, seniority?, language?, style?, planned_questions?}` → `{id}` |
| `POST /api/sessions/{id}/start` | build_plan + первый вопрос; status=in_progress |
| `POST /api/sessions/{id}/answer` | multipart `audio` ИЛИ json `{text}`; → `{transcript, question_text|null, done, action}`; пустой STT → 422 «Речь не распознана» (LLM не дёргаем); авто-fin при 24 ходах |
| `POST /api/sessions/{id}/finish` | досрочный финал |
| `POST /api/sessions/{id}/cancel` | отмена без оценки (evaluate/MLflow не вызываются): status=completed, `error`="Отменено пользователем", без report/балла; для `in_progress`/`created`, иначе 409 |
| `GET /api/sessions` · `GET /api/sessions/{id}` · `GET /api/sessions/{id}/report` | история (перед выдачей — ленивое автозакрытие: осиротевшие `in_progress` без ходов дольше `orphan_close_hours` (12, env `ORPHAN_CLOSE_HOURS`, `<=0` — выключено) закрываются как отменённые) / состояние+turns / отчёт |
| `GET /api/progress` | SQL-агрегация completed-сессий: `{date, position_title, overall_score}`, средние по компетенциям, тренд |
| `GET/PUT /api/settings` · `GET /api/settings/test` | настройки; api_key наружу маскируется `***`; test — проверка соединения |
| `GET /api/models` | список id моделей провайдера: `{"models": [str]}` — GET `{base_url}/models` (OpenAI-совместимый) через клиент настроек; провайдер cloud без base_url/api_key → 422; ошибка соединения/таймаут → 502 |
| `POST /api/tts` `{text}` | → `audio/wav` |
| `POST /api/stt` (multipart) | отладочный |
| `POST /api/debug/chat` | отладочный |

Необработанная ошибка start/answer → status=failed, `error`, 502.

## Frontend (shadcn/ui)

- shadcn-компоненты: button card input textarea label select radio-group tabs badge progress separator sonner dialog.
- `app/page.tsx`: форма новой сессии (Card): Textarea вакансии (обяз.), Select грейд (junior/middle/senior/lead), Select язык (ru/en), RadioGroup стиля, Select числа вопросов (5/8/12); Tabs «История» (таблица: дата, позиция, Badge статуса, скор, → отчёт (только completed с баллом), «Прервать» для in_progress (Dialog) → cancel) и «Прогресс» (`ProgressView`).
- `components/Recorder.tsx`: hold-to-talk (pointerdown/up), `getUserMedia` + `MediaRecorder('audio/webm;codecs=opus')` → Blob → POST /answer; красный индикатор+таймер; fallback Textarea «Ответить текстом»; disabled во время озвучки.
- `components/AudioQueue.tsx`: разбиение на предложения (`[.!?…]\s`), последовательный `POST /api/tts` → ObjectURL → очередь `<audio>`; текст вопроса показывается сразу, не ждёт озвучки.
- `app/session/[id]/page.tsx`: лента переписки (Card на ход), Recorder, «Завершить досрочно» (Dialog-подтверждение) → отчёт; «Прервать» (Dialog) → cancel → главная; при done — авто-переход.
- `app/settings/page.tsx`: RadioGroup провайдер (local/cloud), base_url, api_key (password), model (datalist, «Обновить список» моделей провайдера с фолбэком на статический список), whisper_model, tts_voice; «Проверить соединение» → sonner-toast.
- `app/session/[id]/report/page.tsx` + `ReportView.tsx`: общий балл (Progress крупно), компетенции (Card с Progress и комментарием), turn_feedback (Accordion: вопрос→ответ→что хорошо/упущено/сильный ответ), сильные/слабые (две колонки Badge), план подготовки (таблица тема→действие), вердикт + hire_recommendation-бейдж; degraded → Alert.
- Тема — системная; переключатель не делаем (next-themes — оверинжиниринг для MVP). Без AI-SDK — прямой fetch (`lib/api.ts`).

## Makefile

`setup` (uv python install 3.12; uv sync; npm ci) · `models` (предскачивание whisper+silero) · `backend` · `frontend` · `dev` (обе + wait) · `mlflow-ui` (`uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db --port 5100`) · `test` · `lint` (ruff + tsc --noEmit) · `clean`.

## Critical files & anchors

Несущие решения, реализователю перечитать перед правкой:

- `backend/app/interviewer.py` — промпты PLAN/TURN, STYLE_PROMPTS, state-machine хода; единственное место с LLM-протоколом (`action`, `covered_topic`).
- `backend/app/tracing.py` — mlflow init/autolog, `log_session_run`; контракт метрик `overall_score`/`score_<competency>`, артефактов `report.json`/`transcript.md`.
- `backend/app/llm.py` — `get_client(settings)`: единственная точка переключения local/cloud.
- `backend/app/stt.py` — `decode_audio` (PyAV→16k mono float32) и singleton-инициализация (device/compute).
- `frontend/components/Recorder.tsx` — связка MediaRecorder→answer, блокировка записи во время озвучки; `frontend/components/AudioQueue.tsx` — TTS-очередь по предложениям.

## Верификация MVP (чеклист релиза v0.1.0)

Предпосылки: живой omniroute gateway (:20128); микрофон для голосового шага (есть текстовый fallback); интернет для первичного скачивания моделей (~1.7 ГБ). Порт-реестр — wiki W200.

1. `make setup && make models` — venv собран, модели скачаны (`~/.cache/huggingface`, `~/.cache/torch/hub`).
2. `make test` — pytest зелёный (state-machine, JSON-протокол, degraded-fallback).
3. `make mlflow-ui` → `:5100` отвечает; `make dev` → `:3000` и `:8100/docs` живые.
4. E2E в браузере (вакансия из `docs/example-vacancy.md`): создать сессию (middle, friendly, 5 вопросов) → вопросы звучат → 2 ответа текстом, 1 голосом → транскрипты корректны, follow-ups тематические → «Завершить» → отчёт: баллы, turn_feedback, план подготовки, вердикт+бейдж.
5. MLflow: у сессии run с `overall_score`/`score_*`, артефакты `report.json`+`transcript.md`; трейсы glm с токенами/латентностью; `llm_trace_id` находит трейс.
6. `curl -X POST :8100/api/tts -d '{"text":"Проверка связи"}' --output /tmp/t.wav && file /tmp/t.wav` → WAV; `curl -X POST :8100/api/stt -F audio=@<webm>` → текст.
7. Настройки: provider=cloud без ключа → «Проверить соединение» показывает ошибку (sonner); обратно local — без рестарта, следующий ход проходит.
8. Прогресс: вторая завершённая сессия → `/api/progress` содержит обе, тренд отражает разницу баллов.
9. `git ls-remote origin` содержит push коммитов main.
