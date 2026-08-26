$ErrorActionPreference = "Stop"

uv sync --extra dev
uv run pytest
uv run fdl-speech doctor
uv run fdl-speech download
uv run fdl-speech prepare
uv run fdl-speech eda
uv run fdl-speech train-all
uv run fdl-speech leaderboard
uv run fdl-speech repeat-best
uv run fdl-speech finalize

Write-Host "Full experiment and final frozen-test evaluation complete."
