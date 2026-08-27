# Speech Commands: robust keyword spotting from scratch

[![quality](https://github.com/Laimon99/fdl-project/actions/workflows/ci.yml/badge.svg)](https://github.com/Laimon99/fdl-project/actions/workflows/ci.yml)

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
| E04 | 40-bin log-Mel | depthwise-separable CNN | waveform + SpecAugment | strict E03 augmentation ablation |
| E05 | 40-bin log-Mel | convolutional BiGRU | waveform + SpecAugment | temporal model comparison |

## Main result

The validation-selected E05 CRNN reaches **93.12% accuracy** and **0.9315 macro-F1**
on the frozen 3,081-example test split. Its stratified-bootstrap macro-F1 95% confidence
interval is **[0.9228, 0.9402]** over 10,000 resamples. The model has 292,271 parameters,
occupies 3.42 MiB, and records 3.33 ms median compiled feature-extraction-plus-inference
latency on the documented CPU environment. At 0 dB background noise, macro-F1 falls to
0.8199; `unknown` is the weakest clean class at 0.8404 F1. These negative results are part
of the conclusion, not omitted from the headline.

See the [final report](docs/final_report.md), [model card](docs/model_card.md), and
[frozen evaluation artifacts](artifacts/runs/e05_logmel_crnn_aug/evaluation_testing) for
the full quantitative and qualitative evidence.

The best architecture is repeated with three seeds. The test split remains untouched
until model selection is complete; selection uses validation macro-F1 and accuracy.
`finalize` promotes the selected model and performs its gated clean, robustness,
calibration, efficiency, and qualitative test evaluation.

The green CI badge covers linting plus unit and synthetic-data tests. The raw archive is not
stored in Git, so complete data preparation, training, and evaluation are a separate local
reproduction documented in `docs/reproducibility.md`.

## Repository map

- `src/fdl_speech_commands/`: reusable data, model, training, evaluation, and inference code.
- `configs/`: immutable experiment definitions.
- `tests/`: unit and data-integrity tests.
- `artifacts/`: committed figures, tables, metrics, model files, and predictions.
- `docs/`: data card, experiment protocol, model card, final report, reproducibility guide,
  demo guide, and submission checklist.
- `presentation/`: editable PowerPoint, required PDF export, timed talk script, and defence Q&A.

## Scientific protocol

- The official `validation_list.txt` and `testing_list.txt` define speaker-disjoint splits.
- `unknown` is sampled deterministically from non-target words at the standard 10% rate.
- `silence` is synthesized from background-noise recordings using non-overlapping temporal
  regions across train, validation, and test.
- Training augmentation, validation corruptions, and test corruptions use only their own
  temporally reserved 80%/10%/10% background-noise regions.
- Preprocessing statistics are learned from training data only and embedded in saved models.
- Test metrics include bootstrap confidence intervals, per-class performance, calibration,
  latency, model size, corrupted-audio robustness, and a traceable error gallery.

## Dataset and attribution

Speech Commands v0.01 contains 64,721 one-second-or-shorter spoken-word clips across
30 words, plus six longer background-noise recordings. It was created by Google and
released under CC BY 4.0. The dataset is downloaded from its original TensorFlow URL;
it is never redistributed in this repo. See `CITATION.cff`, `docs/data_card.md`, and
the dataset `LICENSE` after download.

## Submission

The final eLearning package must contain the Python code, trained weights, and the
presentation **as a PDF file**. The presentation and oral discussion are in English.
The internal deadline is kept earlier than the official deadline of 8 September 2026,
23:59 Europe/Rome, to leave time for the group review. The exact requirement-to-file
mapping and the manual upload procedure are in `docs/submission_checklist.md`.
