# interview-simulator — корневой Makefile (каркас, T128).
# Директории backend/ и frontend/ появятся в задачах T129+; пока их нет,
# соответствующие цели завершаются ошибкой с пояснением (см. README.md).

BACKEND_PORT  := 8100
FRONTEND_PORT := 3000
MLFLOW_PORT   := 5100

# Guard: цель доступна только при наличии директории.
# $(error) срабатывает на этапе раскрытия рецепта — сообщение видно
# и при обычном запуске, и при `make -n <цель>`.
define need
$(if $(wildcard $(1)/),,$(error Директория '$(1)' ещё не создана — появится в следующих задачах (T129+). Цель '$@' недоступна до её появления. См. README.md и docs/ARCHITECTURE.md.))
endef

.PHONY: setup models backend frontend dev mlflow-ui test lint clean

setup:
	$(call need,backend)
	$(call need,frontend)
	uv python install 3.12
	cd backend && uv sync
	cd frontend && npm ci

models:
	$(call need,backend)
	cd backend && uv run python -m app.download_models

backend:
	$(call need,backend)
	cd backend && uv run uvicorn app.main:app --reload --port $(BACKEND_PORT)

frontend:
	$(call need,frontend)
	cd frontend && npm run dev

dev:
	$(call need,backend)
	$(call need,frontend)
	cd backend && uv run uvicorn app.main:app --reload --port $(BACKEND_PORT) & \
	cd frontend && npm run dev & \
	wait

mlflow-ui:
	mkdir -p data
	cd backend && uv run mlflow ui --backend-store-uri sqlite:///data/mlflow.db --port $(MLFLOW_PORT)

test:
	$(call need,backend)
	cd backend && uv run pytest

lint:
	$(call need,backend)
	$(call need,frontend)
	cd backend && uv run ruff check .
	cd frontend && npx tsc --noEmit

clean:
	rm -rf data .venv node_modules .next
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
