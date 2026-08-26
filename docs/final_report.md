# Final technical report

## Abstract

We study speaker-independent keyword spotting on Google Speech Commands v0.01 as a
12-class classification problem: ten actionable words, unknown speech, and silence. We
compare MFCC and log-Mel representations across a dense baseline, a conventional CNN, an
efficient depthwise-separable CNN, and a convolutional recurrent model. All decisions are
made on the official validation split. The selected model reaches **94.19%
test accuracy** and **0.9421 macro-F1**, while remaining small enough for
interactive CPU inference.

## Data and protocol

The raw release contains **64,721 spoken clips**, **30
original words**, and **1,881 anonymized speakers** in this local audit.
Audio is mono at 16,000 Hz and no longer than one second. The official
validation/testing lists produce speaker-disjoint partitions. Non-target words and
background recordings create deterministic unknown and silence classes at 10% each,
yielding **28,420 modeled examples**. The manifest builder fails on
speaker or cross-split silence overlap.

Input clips are padded to 16,000 samples. The principal representation is a 40-bin log-Mel
spectrogram from 30 ms Hann windows and a 10 ms stride. MFCCs form a compact baseline.
Normalization statistics are learned only from the training split.

## Controlled experiments

| experiment_id | features | model | augmentation | accuracy | macro_f1 | parameters |
|---|---|---|---|---|---|---|
| e05_logmel_crnn_aug | log_mel | crnn | True | 0.9421 | 0.9427 | 292,271 |
| e03_logmel_dscnn | log_mel | ds_cnn | False | 0.9321 | 0.9324 | 46,863 |
| e04_logmel_dscnn_aug | log_mel | ds_cnn | True | 0.9043 | 0.9050 | 46,863 |
| e02_logmel_cnn | log_mel | small_cnn | False | 0.8771 | 0.8759 | 76,463 |
| e01_mfcc_mlp | mfcc | mlp | False | 0.8380 | 0.8390 | 361,999 |

The validation-selected configuration is **e05_logmel_crnn_aug**. Its three-seed validation
macro-F1 is **0.9424 ± 0.0009**;
validation accuracy is **0.9419 ± 0.0007**.
This stability check is performed after
architecture selection and does not change the test protocol.

The controlled comparison exposes useful trade-offs rather than a single monotonic story.
The MFCC MLP is the weakest baseline. The compact unaugmented DS-CNN reaches
0.9324 macro-F1 with only 46,863 parameters.
Applying the full augmentation recipe to that same architecture changes clean validation
macro-F1 by **-2.74 percentage points**. The selected CRNN recovers and exceeds
the clean DS-CNN, but uses **6.24x** as many parameters. This accuracy versus
efficiency trade-off is retained in the conclusion rather than hidden by the final ranking.

### Post-selection robustness ablation (validation)

| condition | e03_logmel_dscnn | e04_logmel_dscnn_aug | e05_logmel_crnn_aug |
|---|---|---|---|
| clean | 0.9324 | 0.9050 | 0.9427 |
| background_20db | 0.8642 | 0.9003 | 0.9337 |
| background_10db | 0.6754 | 0.8692 | 0.9078 |
| background_0db | 0.1660 | 0.6895 | 0.7929 |
| shift_plus_100ms | 0.9222 | 0.8992 | 0.9332 |
| shift_minus_100ms | 0.9239 | 0.9022 | 0.9351 |

This diagnostic is run only after clean-validation selection. It shows why the apparently
negative DS-CNN augmentation result is still informative: at 0 dB, augmentation improves its
macro-F1 by **52.35 percentage points**, despite the clean-score cost. The CRNN
combines the highest clean score with the strongest noise curve among the compared variants.

## Frozen test evaluation

- Accuracy: **0.9419**, stratified-bootstrap 95% CI
  [0.9335, 0.9500].
- Macro-F1: **0.9421**, 95% CI
  [0.9338, 0.9501].
- Balanced accuracy: **0.9417**.
- Expected calibration error (15 bins): **0.0198**.
- Parameters: **292,271**; serialized size:
  **3.42 MiB**.
- Compiled end-to-end CPU batch-1 latency (feature extraction + model): median
  **3.41 ms**, p95 **3.85 ms**
  on the recorded evaluation machine (200 timed repetitions after
  warm-up).

The validation-to-test macro-F1 change is **-0.06 percentage points**, so the
selected result transfers to the frozen split without a material generalization drop.

### Per-class results

| label | precision | recall | f1-score | support |
|---|---|---|---|---|
| _silence_ | 1.0000 | 0.9611 | 0.9802 | 257 |
| _unknown_ | 0.8645 | 0.8444 | 0.8543 | 257 |
| yes | 0.9802 | 0.9688 | 0.9745 | 256 |
| no | 0.9228 | 0.9484 | 0.9354 | 252 |
| up | 0.8660 | 0.9743 | 0.9170 | 272 |
| down | 0.9573 | 0.8854 | 0.9199 | 253 |
| left | 0.9552 | 0.9588 | 0.9570 | 267 |
| right | 0.9800 | 0.9459 | 0.9627 | 259 |
| on | 0.9478 | 0.9593 | 0.9535 | 246 |
| off | 0.9577 | 0.9504 | 0.9540 | 262 |
| stop | 0.9643 | 0.9759 | 0.9701 | 249 |
| go | 0.9246 | 0.9283 | 0.9264 | 251 |

### Robustness

| condition | accuracy | macro_f1 |
|---|---|---|
| clean | 0.9419 | 0.9421 |
| background_20db | 0.9305 | 0.9308 |
| background_10db | 0.9098 | 0.9096 |
| background_0db | 0.8121 | 0.8146 |
| shift_plus_100ms | 0.9400 | 0.9401 |
| shift_minus_100ms | 0.9383 | 0.9386 |

The noise sweep is deterministic and uses held-out temporal regions of the release's own
background recordings. Macro-F1 drops by **3.25 points** at 10 dB and
**12.75 points** at 0 dB, while either ±100 ms shift costs at most
**0.34 points**. It is a stress test, not a claim of real-world deployment
coverage.

## Qualitative findings

`qualitative_gallery.png` contrasts the six most confident mistakes with six low-confidence
correct predictions, and `qualitative_samples.csv` links every panel to an exported playable
WAV. The weakest class is **_unknown_** (F1 **0.8543**), and the most common
directed error is **down → no** (14 clips). The six most
confident errors all have confidence at least **0.998**. Thus, the low
aggregate ECE does not eliminate rare overconfident failures—a central negative qualitative
finding. Listening helps distinguish phonetic ambiguity, truncation, low energy, background
interference, and plausible label uncertainty.

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
4. Tara N. Sainath and Carolina Parada. *Convolutional Neural Networks for Small-Footprint
   Keyword Spotting*. Interspeech, 2015.
5. Yundong Zhang et al. *Hello Edge: Keyword Spotting on Microcontrollers*.
   [arXiv:1711.07128](https://arxiv.org/abs/1711.07128), 2017.
6. Chuan Guo et al. *On Calibration of Modern Neural Networks*. ICML, 2017.
