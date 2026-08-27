# Reproducibility guide

## Supported environment

- Python 3.13 is selected by `.python-version`; Python 3.12 is also tested in CI.
- CI runs linting and unit/synthetic-data tests only. The raw archive is intentionally not
  stored in Git, so end-to-end preparation, training, and evaluation are verified through
  the separate full-reproduction procedure below rather than claimed as CI coverage.
- Exact Python dependencies are locked in `uv.lock`.
- TensorFlow runs on CPU on native Windows. CUDA is optional and does not alter the commands.
- On native Windows, TensorBoard events are redirected to the system temporary directory only
  when the repository path contains non-ASCII characters, working around a TensorFlow writer
  limitation. CSV histories, checkpoints, metrics, and every submission artifact remain local.
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
uv run fdl-speech repeat-best
uv run fdl-speech robustness-ablation
uv run fdl-speech finalize
uv run fdl-speech report
uv run fdl-speech audit
uv run fdl-speech package
```

The first three commands require no dataset. The download is resumable and the preparation
commands may be rerun safely. `finalize` is the deliberate one-time test-set gate: it reads
the frozen validation selection, promotes the chosen run, and only then evaluates the test
split. `package` refuses to build a submission when the audit fails or Git is dirty.

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

After reporting and packaging:

- `docs/final_report.md` and `docs/model_card.md`: generated scientific narrative and model facts.
- `presentation/FDL_Speech_Commands.pptx` and `.pdf`: editable and submission-ready decks.
- `presentation/presentation_script.md`: timed 15-minute, three-speaker script.
- `presentation/q_and_a.md`: defence preparation grounded in the recorded results.
- `submission/fdl_speech_commands_elearning.zip`: clean, audited eLearning upload bundle.
- `submission/audit.json`: machine-readable preflight result and file hashes.

## Reproduction levels

- **CI-equivalent integrity check:** `uv run ruff check .` and
  `uv run pytest -m "not data"` (unit and synthetic-data coverage; no raw archive).
- **Local data-pipeline smoke check:** `uv run fdl-speech smoke-test` after downloading and
  preparing the official release.
- **Fast training check:** add `--epochs 1` to `train` for one configuration.
- **Full reproduction:** use the locked epoch/early-stopping settings without overrides.

Small floating-point differences across CPU instruction sets are possible. Scientific
conclusions are based on validation ranking and repeated-seed statistics, not bitwise-equal
weights.
