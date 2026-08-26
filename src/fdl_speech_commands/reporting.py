from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .constants import PROJECT_ROOT
from .utils import ProjectError, read_json


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    display = frame.loc[:, columns].copy()
    headers = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = []
    for values in display.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, float):
                formatted.append(f"{value:.4f}")
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([headers, separator, *rows])


def _require(path: Path) -> Path:
    if not path.exists():
        raise ProjectError(f"Required reporting artifact is missing: {path}")
    return path


def generate_final_report(output: str | Path = PROJECT_ROOT / "docs" / "final_report.md") -> Path:
    table_dir = PROJECT_ROOT / "artifacts" / "tables"
    model_dir = PROJECT_ROOT / "artifacts" / "models"
    eda: dict[str, Any] = read_json(_require(table_dir / "eda_summary.json"))
    leaderboard = pd.read_csv(_require(table_dir / "experiment_leaderboard.csv"))
    seed_summary: dict[str, Any] = read_json(
        _require(table_dir / "seed_stability_validation.summary.json")
    )
    selection: dict[str, Any] = read_json(_require(model_dir / "selection.json"))
    selected_id = selection["selected_experiment"]
    evaluation_dir = PROJECT_ROOT / "artifacts" / "runs" / selected_id / "evaluation_testing"
    metrics: dict[str, Any] = read_json(_require(evaluation_dir / "metrics.json"))
    robustness = pd.read_csv(_require(evaluation_dir / "robustness.csv"))
    classes = pd.read_csv(_require(evaluation_dir / "classification_report.csv"), index_col=0)
    class_rows = classes.loc[[index for index in classes.index if index.startswith("_") or index.isalpha()]]
    intervals = metrics["bootstrap_confidence_intervals"]

    content = f"""# Final technical report

## Abstract

We study speaker-independent keyword spotting on Google Speech Commands v0.01 as a
12-class classification problem: ten actionable words, unknown speech, and silence. We
compare MFCC and log-Mel representations across a dense baseline, a conventional CNN, an
efficient depthwise-separable CNN, and a convolutional recurrent model. All decisions are
made on the official validation split. The selected model reaches **{metrics['accuracy']:.2%}
test accuracy** and **{metrics['macro_f1']:.4f} macro-F1**, while remaining small enough for
interactive CPU inference.

## Data and protocol

The raw release contains **{eda['raw_speech_recordings']:,} spoken clips**, **{eda['original_words']}
original words**, and **{eda['unique_speakers']:,} anonymized speakers** in this local audit.
Audio is mono at {eda['sample_rate'][0]:,} Hz and no longer than one second. The official
validation/testing lists produce speaker-disjoint partitions. Non-target words and
background recordings create deterministic unknown and silence classes at 10% each,
yielding **{eda['modeling_examples']:,} modeled examples**. The manifest builder fails on
speaker or cross-split silence overlap.

Input clips are padded to 16,000 samples. The principal representation is a 40-bin log-Mel
spectrogram from 30 ms Hann windows and a 10 ms stride. MFCCs form a compact baseline.
Normalization statistics are learned only from the training split.

## Controlled experiments

{_markdown_table(leaderboard, ['experiment_id', 'features', 'model', 'augmentation', 'accuracy', 'macro_f1', 'parameters'])}

The validation-selected configuration is **{selected_id}**. Its three-seed validation
macro-F1 is **{seed_summary['validation_macro_f1_mean']:.4f} +/-
{seed_summary['validation_macro_f1_std']:.4f}**; validation accuracy is
**{seed_summary['validation_accuracy_mean']:.4f} +/-
{seed_summary['validation_accuracy_std']:.4f}**. This stability check is performed after
architecture selection and does not change the test protocol.

## Frozen test evaluation

- Accuracy: **{metrics['accuracy']:.4f}**, stratified-bootstrap 95% CI
  [{intervals['accuracy']['lower_95']:.4f}, {intervals['accuracy']['upper_95']:.4f}].
- Macro-F1: **{metrics['macro_f1']:.4f}**, 95% CI
  [{intervals['macro_f1']['lower_95']:.4f}, {intervals['macro_f1']['upper_95']:.4f}].
- Balanced accuracy: **{metrics['balanced_accuracy']:.4f}**.
- Expected calibration error (15 bins): **{metrics['expected_calibration_error_15_bins']:.4f}**.
- Parameters: **{metrics['parameters']:,}**; serialized size:
  **{metrics['model_size_bytes'] / 2**20:.2f} MiB**.
- Native CPU batch-1 latency: median **{metrics['latency']['median_ms']:.2f} ms**,
  p95 **{metrics['latency']['p95_ms']:.2f} ms** on the recorded evaluation machine.

### Per-class results

{_markdown_table(class_rows.reset_index(names='label'), ['label', 'precision', 'recall', 'f1-score', 'support'])}

### Robustness

{_markdown_table(robustness, ['condition', 'accuracy', 'macro_f1'])}

The noise sweep is deterministic and uses held-out temporal regions of the release's own
background recordings. It is a stress test, not a claim of real-world deployment coverage.

## Qualitative findings

`qualitative_gallery.png` contrasts the six most confident mistakes with six low-confidence
correct predictions, and `qualitative_samples.csv` links every panel to an exported playable
WAV. The confusion matrices and these samples are used together: aggregate metrics identify
systematic class pairs, while listening separates phonetic ambiguity, truncation, low energy,
background interference, and plausible label uncertainty.

## Limitations

The data consists of isolated English words and does not cover continuous speech. Volunteer
recordings cannot represent all accents, microphones, rooms, or speaking styles, and no
demographic attributes are available for subgroup analysis. The unknown and silence classes
are protocol constructs. Results therefore demonstrate a rigorous educational keyword spotter,
not a safety-critical or production ASR system.

## Reproducibility and references

The repository locks dependencies, seeds, manifests, experiment configurations, saved models,
predictions, and presentation figures. Full commands are in `docs/reproducibility.md`.

1. Pete Warden. *Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition*.
   [arXiv:1804.03209](https://arxiv.org/abs/1804.03209), 2018.
2. Google Research. [Launching the Speech Commands Dataset](https://research.google/blog/launching-the-speech-commands-dataset/), 2017.
3. Daniel S. Park et al. *SpecAugment: A Simple Data Augmentation Method for Automatic Speech
   Recognition*. [arXiv:1904.08779](https://arxiv.org/abs/1904.08779), 2019.
"""
    output = Path(output)
    output.write_text(content, encoding="utf-8")
    _write_final_model_card(metrics, selected_id, output.parent / "model_card.md")
    return output


def _write_final_model_card(
    metrics: dict[str, Any],
    selected_id: str,
    output: Path,
) -> None:
    model_size = metrics["model_size_bytes"] / 2**20
    text = f"""# Model card: Speech Commands keyword spotter

## Model summary

- Task: classify one second of mono 16 kHz audio into 12 keyword-spotting classes.
- Selected experiment: `{selected_id}` by validation macro-F1.
- Framework: TensorFlow 2.21 with Keras 3.15.
- Parameters: {metrics['parameters']:,}.
- Serialized Keras model: {model_size:.2f} MiB at `artifacts/models/speech_commands_best.keras`.
- Test accuracy: {metrics['accuracy']:.4f}; macro-F1: {metrics['macro_f1']:.4f}.
- Test calibration ECE: {metrics['expected_calibration_error_15_bins']:.4f}.
- Native CPU latency: {metrics['latency']['median_ms']:.2f} ms median at batch size one.

## Intended use

This is an educational, speaker-independent keyword spotter and reproducible research
artifact. It is appropriate for comparing compact neural architectures on Speech Commands.
It is not suitable for authentication, surveillance, unrestricted transcription, or
safety-critical voice control.

## Data and evaluation

Training uses the 12-class Speech Commands v0.01 protocol documented in `data_card.md`.
Architecture selection uses only the official validation split. The frozen selected model is
evaluated on the official speaker-disjoint test split with per-class metrics, 10,000-sample
bootstrap intervals, calibration, efficiency, deterministic noise/time-shift stress tests,
and auditable qualitative examples.

## Limitations

Coverage of accents, microphones, ages, and environments is incomplete; demographic metadata
is unavailable. Inputs outside isolated English words are not validated. `unknown` and
`silence` are aggregate protocol classes. Deployment would require in-domain collection,
rejection thresholds, monitoring, and a fresh risk assessment.
"""
    output.write_text(text, encoding="utf-8")
