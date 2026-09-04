"""Pytest: пакет app импортируется из backend/ (uv run pytest запускается из backend/)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
