$ErrorActionPreference = "Stop"

uv sync --extra dev
uv run pytest
uv run fdl-speech doctor
uv run fdl-speech download
uv run fdl-speech prepare
uv run fdl-speech eda
uv run fdl-speech train-all
uv run fdl-speech leaderboard
uv run fdl-speech promote

Write-Host "Model selection complete. Evaluate only the selected run shown above."

