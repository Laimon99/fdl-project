# Speech Commands: robust keyword spotting from scratch

Foundations of Deep Learning project for the 2025/2026 academic year. The project
studies a reproducible 12-class keyword-spotting task on **Google Speech Commands
v0.01**: ten commands (`yes`, `no`, `up`, `down`, `left`, `right`, `on`, `off`,
`stop`, `go`), plus `unknown` and `silence`.

The repository is intentionally more than a single training notebook. It contains a
versioned data protocol, exploratory analysis, multiple neural architectures,
controlled ablations, robustness and calibration evaluation, qualitative error
analysis, saved models, an inference CLI, and all presentation material.

## Reproduce the project

Install [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
uv sync --extra dev
uv run fdl-speech doctor
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

Every command is restartable. Raw data and transient caches are ignored by Git;
manifests, metrics, figures, final models, and presentation files are versioned.

For a fast integrity check without full training:

```bash
uv run pytest
uv run fdl-speech smoke-test
```

## Experiment matrix

| ID | Input representation | Neural architecture | Augmentation | Purpose |
|---|---|---|---|---|
| E01 | 13 MFCCs | MLP | none | low-capacity neural baseline |
| E02 | 40-bin log-Mel | compact 2D CNN | none | convolutional baseline |
| E03 | 40-bin log-Mel | depthwise-separable CNN | none | efficient KWS model |
| E04 | 40-bin log-Mel | depthwise-separable CNN | waveform + SpecAugment | augmentation ablation |
| E05 | 40-bin log-Mel | convolutional BiGRU | waveform + SpecAugment | temporal model comparison |

The best architecture is repeated with three seeds. The test split remains untouched
until model selection is complete; selection uses validation macro-F1 and accuracy.
`finalize` promotes the selected model and performs its one-time clean, robustness,
calibration, efficiency, and qualitative test evaluation.

## Repository map

- `src/fdl_speech_commands/`: reusable data, model, training, evaluation, and inference code.
- `configs/`: immutable experiment definitions.
- `tests/`: unit and data-integrity tests.
- `artifacts/`: committed figures, tables, metrics, model files, and predictions.
- `docs/`: data card, experiment protocol, model card, reproducibility guide, and talk script.
- `presentation/`: editable PowerPoint and required PDF export.

## Scientific protocol

- The official `validation_list.txt` and `testing_list.txt` define speaker-disjoint splits.
- `unknown` is sampled deterministically from non-target words at the standard 10% rate.
- `silence` is synthesized from background-noise recordings using non-overlapping temporal
  regions across train, validation, and test.
- Preprocessing statistics are learned from training data only and embedded in saved models.
- Test metrics include bootstrap confidence intervals, per-class performance, calibration,
  latency, model size, corrupted-audio robustness, and a traceable error gallery.

## Dataset and attribution

Speech Commands v0.01 contains 64,727 one-second-or-shorter WAV recordings of 30
spoken words. It was created by Google and released under CC BY 4.0. The dataset is
downloaded from its original TensorFlow URL; it is never redistributed in this repo.
See `CITATION.cff`, `docs/data_card.md`, and the dataset `LICENSE` after download.

## Submission

The final eLearning package must contain the Python code, trained weights, and the
presentation **as a PDF file**. The presentation and oral discussion are in English.
The internal deadline is kept earlier than the official deadline of 8 September 2026,
23:59 Europe/Rome, to leave time for the group review.
