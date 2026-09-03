# Interview Simulator

Голосовой тренажёр собеседований: вставляете текст вакансии, выбираете грейд и стиль интервьюера — LLM-интервьюер проводит голосовое интервью по раундам (technical / behavioral / algorithms / system_design), задаёт follow-up вопросы по вашим ответам, а в конце формирует отчёт: скоринг по компетенциям, разбор каждого ответа, план подготовки и вердикт.

## Стек

| Слой | Технологии |
|---|---|
| Backend | FastAPI (порт `:8100`, Swagger — `:8100/docs`), SQLAlchemy, SQLite |
| Frontend | Next.js (порт `:3000`), Tailwind, shadcn/ui |
| STT | faster-whisper (локально, CUDA/CPU) |
| TTS | Silero (русский, локально) |
| Трекинг | MLflow (UI — `:5100`) |
| LLM | локальный omniroute-gateway или облачный OpenAI-совместимый провайдер |

## Быстрый старт

```bash
make setup       # python 3.12 через uv, зависимости backend (uv sync) и frontend (npm ci)
make models      # скачивание faster-whisper и Silero (~1.7 ГБ, кэш в ~/.cache)
make dev         # frontend :3000 + backend :8100 (Swagger: http://localhost:8100/docs)
make mlflow-ui   # MLflow UI: http://localhost:5100
```

## LLM-провайдеры

- **Локальный (по умолчанию)** — omniroute-gateway, OpenAI-совместимый: `http://localhost:20128/v1`, модель `glm/glm-5.3-flash` (см. `.env.example`).
- **Облачный** — любой OpenAI-совместимый провайдер. Base URL, API-ключ и модель задаются в настройках UI — без правки кода. Ключ в репозиторий не попадает (`.env` в `.gitignore`).

Переключение local ↔ cloud — в UI, без рестарта бэкенда.

## Голос: возможности и ограничения

Голосовой цикл работает полностью локально: faster-whisper (STT) и Silero (TTS). Целевая машина — RTX 3070 Ti 8 ГБ (whisper: CUDA `int8_float16`). Микрофон не обязателен: если его нет, вопросы приходят текстом, а отвечать можно текстом — текстовый fallback является частью продукта, а не обходным путём.

### TTS: Silero (T135)

- `POST /api/tts` `{"text": "...", "voice": "kseniya"}` → `audio/wav` (24 кГц, mono). Голоса: `aidar/baya/kseniya/xenia/eugene/random`; смена голоса — без рестарта (модель — singleton).
- Текст длиннее 300 символов режется по предложениям (сверхдлинные предложения — ещё и по словам, каждая часть ≤300 символов) и склеивается в один непрерывный wav.
- Модель строго CPU (VRAM не занимает). Артефакт `v4_ru.pt` (~40 МБ, models.silero.ai) кладётся в `~/.cache/torch/hub/silero/`; скачивание входит в `make models` (консолидация T141). Отдельный прогрев при необходимости: `cd backend && uv run python -m app.download_silero`.
- Порядок загрузки в `app/tts.py`: артефакт `v4_ru.pt` → `torch.hub.load('snakers4/silero-models', ...)` → `source='local'`. Фолбэк W200: torch.hub-путь требует `omegaconf` (в зависимостях проекта нет) и доступности репозитория `snakers4/silero-models` на Hugging Face (сейчас отдаёт 404) — поэтому основной путь в этом окружении артефактный.

### Голосовые модели: make models (T141)

`make models` скачивает обе модели одним прогоном (шаги whisper → silero, каждый идемпотентен: повтор — «already cached»):

| Шаг | Модель | Размер | Кэш |
|---|---|---|---|
| STT | faster-whisper large-v3-turbo (ct2) | ~1.6 ГБ | `~/.cache/huggingface` |
| TTS | silero `v4_ru.pt` | ~40 МБ | `~/.cache/torch/hub/silero/` |

Офлайн-фолбэки при первом старте без сети:

| Компонент | Настройка | Поведение без сети/кэша |
|---|---|---|
| STT | `WHISPER_DEVICE=cpu\|cuda\|auto` (дефолт `auto` → CUDA при наличии, иначе CPU) | Скачивание через `make models` обязательно; без кэша whisper не стартует |
| TTS | порядок в `app/tts.py`: артефакт → `torch.hub` → `source='local'` | Без артефакта основной путь недоступен; фолбэк `source='local'` требует репозитория `snakers4/silero-models` |

## Состояние репозитория (каркас, T128)

Директорий `backend/` и `frontend/` ещё нет — они появятся в следующих задачах (T129+). Поэтому make-цели `setup`, `models`, `backend`, `frontend`, `dev`, `test`, `lint` сейчас завершаются ошибкой с пояснением (ненулевой код выхода — ожидаемое поведение).

- `clean` — уже рабочий: удаляет `data/`, `.venv/`, `node_modules/`, `.next/`, `__pycache__/`.
- `mlflow-ui` — рецепт финальный (`uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db --port 5100` из `backend/`); заработает, как только появится backend-окружение с MLflow.

## Документация

- [`AGENTS.md`](AGENTS.md) — конституция проекта: правила для агентов (git-flow, коммиты, стиль кода).
- [`docs/PRD.md`](docs/PRD.md) — продукт: проблема, решение, user stories, scope/non-goals, метрики успеха.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — архитектура: дерево репо, модель данных, LLM-протоколы (PLAN/TURN/EVAL), API, MLflow-контракт.
- [`docs/example-vacancy.md`](docs/example-vacancy.md) — тестовая вакансия для сквозной E2E-проверки.

## For agents (EN)

Voice interview-training web app: paste a vacancy, get a voice interview (technical / behavioral / algorithms / system_design rounds with follow-ups) and a scored report. Stack: FastAPI (port `:8100`, Swagger at `:8100/docs`) + Next.js (port `:3000`, shadcn/ui), faster-whisper STT, Silero TTS, MLflow (`:5100`), SQLite. Quickstart: `make setup` → `make models` → `make dev`; MLflow UI via `make mlflow-ui`. LLM providers: local omniroute-gateway at `http://localhost:20128/v1` (model `glm/glm-5.3-flash`) or any OpenAI-compatible cloud provider configured in the UI — no code changes. Voice runs fully locally (target GPU: RTX 3070 Ti 8 GB); a text fallback exists for setups without a microphone. Note: `backend/` and `frontend/` are not scaffolded yet — the corresponding make targets exit with an explanatory error (non-zero exit code is expected). Agents must read [`AGENTS.md`](AGENTS.md) before writing code; product scope: [`docs/PRD.md`](docs/PRD.md), architecture and contracts: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
