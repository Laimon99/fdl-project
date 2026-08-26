# Reproducibility guide

## Supported environment

- Python 3.13 is selected by `.python-version`; Python 3.12 is also tested in CI.
- Exact Python dependencies are locked in `uv.lock`.
- TensorFlow runs on CPU on native Windows. CUDA is optional and does not alter the commands.
- Random sources in Python, NumPy, and TensorFlow are seeded; deterministic TensorFlow ops
  are requested wherever the installed backend supports them.

## Clean reproduction

From a fresh clone:

```bash
uv sync --extra dev
uv run fdl-speech doctor
uv run pytest
uv run fdl-speech download
uv run fdl-speech prepare
uv run fdl-speech eda
uv run fdl-speech train-all
uv run fdl-speech leaderboard
uv run fdl-speech promote
uv run fdl-speech evaluate artifacts/runs/<selected-run>
```

The first three commands require no dataset. The download is resumable and the preparation
commands may be rerun safely.

## Expected evidence

After data preparation:

- `data/processed/dataset_source.json`: URL, version, byte count, and archive SHA-256.
- `data/processed/raw_inventory.csv`: header-level metadata for every WAV.
- `data/processed/manifest.csv`: immutable examples, labels, split, speaker, and silence offset.
- `data/processed/manifest.metadata.json`: protocol settings, counts, and manifest SHA-256.

After each experiment:

- resolved YAML, environment metadata, text model summary, epoch log, validation metrics,
  TensorBoard log, and `best_model.keras` under `artifacts/runs/<experiment-id>/`.

After final evaluation:

- predictions and class report; clean and normalized confusion matrices; calibration bins and
  reliability plot; bootstrap confidence intervals; robustness table/figure; latency and size;
  selected WAV clips and spectrograms for qualitative inspection.

## Reproduction levels

- **Integrity check:** `uv run pytest` and `uv run fdl-speech smoke-test`.
- **Fast training check:** add `--epochs 1` to `train` for one configuration.
- **Full reproduction:** use the locked epoch/early-stopping settings without overrides.

Small floating-point differences across CPU instruction sets are possible. Scientific
conclusions are based on validation ranking and repeated-seed statistics, not bitwise-equal
weights.

