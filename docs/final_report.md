# Final technical report

## Abstract

We study speaker-independent keyword spotting on Google Speech Commands v0.01 as a
12-class classification problem: ten actionable words, unknown speech, and silence. We
compare MFCC and log-Mel representations across a dense baseline, a conventional CNN, an
efficient depthwise-separable CNN, and a convolutional recurrent model. All decisions are
made on the official validation split. The selected model reaches **93.12%
test accuracy** and **0.9315 macro-F1**, while remaining small enough for
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
| e05_logmel_crnn_aug | log_mel | crnn | True | 0.9395 | 0.9399 | 292,271 |
| e03_logmel_dscnn | log_mel | ds_cnn | False | 0.9321 | 0.9324 | 46,863 |
| e04_logmel_dscnn_aug | log_mel | ds_cnn | True | 0.9088 | 0.9095 | 46,863 |
| e02_logmel_cnn | log_mel | small_cnn | False | 0.8771 | 0.8759 | 76,463 |
| e01_mfcc_mlp | mfcc | mlp | False | 0.8380 | 0.8390 | 361,999 |

The validation-selected configuration is **e05_logmel_crnn_aug**. Its three-seed validation
macro-F1 is **0.9394 ± 0.0005**;
validation accuracy is **0.9389 ± 0.0006**.
This stability check is performed after
architecture selection and does not change the test protocol.

The controlled comparison exposes useful trade-offs rather than a single monotonic story.
The MFCC MLP is the weakest baseline. The compact unaugmented DS-CNN reaches
0.9324 macro-F1 with only 46,863 parameters.
Applying the full augmentation recipe to that same architecture changes clean validation
macro-F1 by **-2.29 percentage points**. The selected CRNN recovers and exceeds
the clean DS-CNN, but uses **6.24x** as many parameters. This accuracy versus
efficiency trade-off is retained in the conclusion rather than hidden by the final ranking.

### Post-selection robustness ablation (validation)

| condition | e03_logmel_dscnn | e04_logmel_dscnn_aug | e05_logmel_crnn_aug |
|---|---|---|---|
| clean | 0.9324 | 0.9095 | 0.9399 |
| background_20db | 0.8644 | 0.8991 | 0.9292 |
| background_10db | 0.6855 | 0.8664 | 0.9015 |
| background_0db | 0.2093 | 0.7112 | 0.7841 |
| shift_plus_100ms | 0.9222 | 0.8967 | 0.9364 |
| shift_minus_100ms | 0.9239 | 0.9012 | 0.9331 |

This diagnostic is run only after clean-validation selection. It shows why the apparently
negative DS-CNN augmentation result is still informative: at 0 dB, augmentation improves its
macro-F1 by **50.20 percentage points**, despite the clean-score cost. The CRNN
combines the highest clean score with the strongest noise curve among the compared variants.

## Frozen test evaluation

- Accuracy: **0.9312**, stratified-bootstrap 95% CI
  [0.9224, 0.9400].
- Macro-F1: **0.9315**, 95% CI
  [0.9228, 0.9402].
- Balanced accuracy: **0.9311**.
- Expected calibration error (15 bins): **0.0303**.
- Parameters: **292,271**; serialized size:
  **3.42 MiB**.
- Compiled end-to-end CPU batch-1 latency (feature extraction + model): median
  **3.33 ms**, p95 **3.73 ms**
  on the recorded evaluation machine (200 timed repetitions after
  warm-up).

The validation-to-test macro-F1 change is **-0.85 percentage points**, so the
selected result transfers to the frozen split without a material generalization drop.

### Per-class results

| label | precision | recall | f1-score | support |
|---|---|---|---|---|
| _silence_ | 1.0000 | 0.8755 | 0.9336 | 257 |
| _unknown_ | 0.8739 | 0.8093 | 0.8404 | 257 |
| yes | 0.9800 | 0.9570 | 0.9684 | 256 |
| no | 0.8889 | 0.9524 | 0.9195 | 252 |
| up | 0.8457 | 0.9669 | 0.9022 | 272 |
| down | 0.9544 | 0.9091 | 0.9312 | 253 |
| left | 0.8955 | 0.9625 | 0.9278 | 267 |
| right | 0.9574 | 0.9537 | 0.9555 | 259 |
| on | 0.9518 | 0.9634 | 0.9576 | 246 |
| off | 0.9685 | 0.9389 | 0.9535 | 262 |
| stop | 0.9640 | 0.9679 | 0.9659 | 249 |
| go | 0.9274 | 0.9163 | 0.9218 | 251 |

### Robustness

| condition | accuracy | macro_f1 |
|---|---|---|
| clean | 0.9312 | 0.9315 |
| background_20db | 0.9247 | 0.9246 |
| background_10db | 0.9111 | 0.9106 |
| background_0db | 0.8202 | 0.8199 |
| shift_plus_100ms | 0.9273 | 0.9276 |
| shift_minus_100ms | 0.9309 | 0.9310 |

The noise sweep is deterministic and uses held-out temporal regions of the release's own
background recordings. Macro-F1 drops by **2.09 points** at 10 dB and
**11.16 points** at 0 dB, while either ±100 ms shift costs at most
**0.38 points**. It is a stress test, not a claim of real-world deployment
coverage.

## Qualitative findings

`qualitative_gallery.png` contrasts the six most confident mistakes with six low-confidence
correct predictions, and `qualitative_samples.csv` links every panel to an exported playable
WAV. The weakest class is **_unknown_** (F1 **0.8404**), and the most common
directed error is **_silence_ → up** (17 clips). The six most
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
