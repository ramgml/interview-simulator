# Верификация релиза MVP v0.1.0

> Отчёт о прогоне чеклиста «Верификация MVP» (`docs/ARCHITECTURE.md`, раздел «Верификация MVP») на актуальном `develop` (50bef9f, эпик T127, задача T137, 2026-09-03). Живое окружение: omniroute gateway `glm/glm-5.3-flash`, preview-инстанс из worktree.

## Результаты по пунктам чеклиста

| # | Пункт | Результат | Доказательство |
|---|---|---|---|
| 1 | `make setup && make models` — venv, модели | Выполнено ранее (эпики T121–T126); модели в кэше — TTS/STT работают живьём (см. п. 6) | Прогон п. 6 |
| 2 | `make test` — pytest зелёный | **132 passed** | Прогон 2026-09-03 в worktree T137 |
| 3 | `make mlflow-ui` → `:5100`; `make dev` → `:3000`, `:8100/docs` | MLflow UI **200**; бэк **health ok**, **/docs 200** | curl-прогоны того же дня |
| 4 | E2E в браузере (вакансия из `docs/example-vacancy.md`) | Полный E2E пройден в T136 (комментарий в задаче от 2026-09-03 13:39, QA-чеклист 8/8): сессия middle/friendly/5 → вопросы звучат, текст виден сразу, текстовые и голосовые ответы, follow-up'ы тематические, finish → отчёт (баллы, turn_feedback, план, вердикт+бейдж) | T136, QA-чеклист задачи T136 |
| 5 | MLflow: run с метриками, артефакты, трейсы | У всех completed-сессий заполнен `mlflow_run_id`; артефакты `report.json`+`transcript.md`, трейсы glm с токенами/латентностью | 13 тестов `test_tracing.py`; `mlflow_run_id` в БД сессий |
| 6 | TTS/STT curl | `POST :8100/api/tts` → **WAV PCM 16-bit mono 24 kHz**; round-trip TTS→STT: «Проверка связи один два три» → текст, confidence 0.71; синтетический тон (без речи) → **422 «Речь не распознана»** (контракт пустого транскрипта) | Вывод `file /tmp/t.wav`; curl-прогоны |
| 7 | Настройки: cloud без ключа → ошибка; обратно local без рестарта | sonner-тост «Заполните облачный провайдер в настройках: base_url, api_key и model»; возврат в local → `/api/settings/test` `{"ok":true}` без рестарта, следующий ход проходит | Скриншоты в QA-чеклисте задачи T137 |
| 8 | Прогресс: вторая завершённая сессия, тренд | `/api/progress` содержит все completed-сессии прогона (3 шт.); тренд `null` — в этом прогоне все EVAL ушли в degraded-фоллбек (см. «Известные ограничения» п. 2). Логика тренда на скорах покрыта тестами `test_progress_two_sessions_scores_averages_trend_down`, `test_progress_trend_up` и подтверждена живьём в T136 (валидный отчёт 6.5/10) | curl `/api/progress`; юнит-тесты |
| 9 | Push в `main` | Выполняется мержем релизного PR владельцем | GitFlow (W201) |

## Известные ограничения (наблюдения прогона, вне скоупа MVP)

1. **SQLite `database is locked`.** При конкурентных долгих записях (EVAL + MLflow-артефакты) `/finish` дважды вернул 500 с `sqlite3.OperationalError: database is locked`. Для MVP некритично (клиентский ретрай, сессия остаётся в согласованном состоянии), кандидат в backlog: WAL-режим / таймаут busy_timeout.
2. **Rate-limit шлюза на тяжёлых EVAL.** Финальная оценка (полный транскрипт в одном промпте) стабильно упирается в лимит очереди OmniRoute (`maxWaitMs=60000`, волны 503 в вечерние часы) → срабатывает штатный degraded-фоллбек: отчёт помечается `degraded: true` с пояснением в UI (Alert). Более короткие вызовы (ходы интервью, TTS) проходят штатно. Кандидат в backlog: чанкинг EVAL-промпта или повышение `maxWaitMs`.
3. Интервьюер может задавать дополнительные вопросы сверх `planned_questions` (наблюдалось «Вопрос 9 из 5» при живом LLM). Продуктовый вопрос о соответствии счётчика фактическому числу — на решение владельца.

## Покрытие тест-набора постановки T137

Аудит показал: все перечисленные в постановке T137 тест-кейсы уже реализованы и зелёные (набор рос в задачах T131–T136, T152), дублирование не создавалось. Соответствие:

| Постановка T137 | Реализовано |
|---|---|
| `test_llm`: чистый JSON / JSON в заборах | `test_json_chat_plain_json`, `test_json_chat_strips_json_fence`, `test_json_chat_strips_bare_fence` |
| `test_llm`: ретрай при мусоре | `test_json_chat_retry_adds_system_prompt_and_succeeds` |
| `test_llm`: InterviewError при втором фейле | `test_json_chat_both_invalid_raises` |
| `test_interviewer`: ветки followup/next_question/finish | `test_conduct_turn_followup_branch`, `test_conduct_turn_next_question_branch`, `test_conduct_turn_finish_branch` |
| `test_interviewer`: fallback при InterviewError | `test_conduct_turn_llm_error_falls_back_to_unasked_question`, `..._all_asked_finishes`, `..._without_plan_finishes` |
| `test_interviewer`: cap 24 хода | `test_conduct_turn_sends_plan_and_transcript_capped_at_24` |
| `test_evaluator`: парсинг отчёта | `test_evaluate_valid_json_returns_report_by_schema` (+ fences, temperature) |
| `test_evaluator`: degraded-fallback | `test_evaluate_garbage_twice_returns_degraded_with_verdict_and_nulls`, `test_evaluate_llm_down_returns_degraded_not_raises` |
| `test_stt`: decode_audio wav → 16k mono | `test_decode_audio_mono_16k_float32`, `test_decode_audio_contract_resample_to_mono_16k`, `test_decode_audio_webm_opus_container` |
| `test_sessions_api`: created→in_progress→completed | 32 теста `test_sessions.py` (start/answer/finish/report/progress) |
| `test_sessions_api`: mlflow-ошибка не роняет сессию | `test_finish_completes_when_log_session_run_raises`, `test_finish_survives_broken_mlflow_tracking_uri` |
