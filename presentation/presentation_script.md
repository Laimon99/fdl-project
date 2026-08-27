# Presentation script

Target duration: 15 minutes. The script is intentionally slightly shorter than the available time so that speakers can pause on figures without rushing.

## Speaker allocation

- Slides 1-5: Simone (INSERT SURNAME), approximately 5 minutes.
- Slides 6-10: Vlad (INSERT SURNAME), approximately 5 minutes.
- Slides 11-15: INSERT FULL NAME, approximately 5 minutes.

## Slide 1 - Robust keyword spotting from scratch (0:00-0:45)

Good morning. We are Simone, Vlad, and INSERT FULL NAME. Our project studies speaker-independent keyword spotting on Google Speech Commands. We did not treat this as a single model-training exercise. We built a complete and reproducible pipeline, from raw WAV files to a frozen Keras model, and we evaluated both where it works and where it fails. Our central question is: which audio representation, neural architecture, and augmentation strategy remain convincing when the data protocol prevents speaker leakage and every design choice is locked before final test evaluation?

Transition: first, we define the exact task and the evidence available to us.

## Slide 2 - One second becomes one intent (0:45-1:35)

Every input is a mono waveform sampled at 16 kilohertz and lasting at most one second. The output has twelve classes: ten commands, plus unknown speech and silence. Unknown prevents the classifier from treating every spoken word as a command, while silence provides a rejection class for background audio. We intentionally used no pretrained component. That choice keeps the project aligned with the course and lets us attribute improvements directly to our own feature pipeline and neural architectures.

Transition: the task is compact, but the source data is substantially richer than twelve labels.

## Slide 3 - Near-balanced modeling preserves the raw audit (1:35-2:30)

Our local audit contains 64,721 spoken clips from 1,881 anonymized speakers and thirty original words. We preserve this raw inventory in the repository. The locked modeling manifest contains 28,420 examples and is approximately balanced across the twelve output classes. We keep every target-command recording; their source counts are already similar. Non-target words are sampled deterministically into unknown rather than discarded, and silence is synthesized from the supplied background recordings. This gives us a manageable classification task without pretending that only the ten target words exist.

Transition: before choosing a network, we inspected the signal characteristics that preprocessing must handle.

## Slide 4 - Clip length is stable; energy is not (2:30-3:25)

The duration plot shows that most recordings are already close to one second, so a fixed input length is natural. We pad shorter clips and trim longer ones to exactly 16,000 samples. The energy audit tells a different story: target and unknown recordings overlap, but amplitude varies widely. We therefore avoid global normalization computed from the full dataset. Statistics are learned only on training data, and augmentation perturbs gain, position, and background noise to reflect the observed variability.

Transition: correct preprocessing is not enough if the split itself leaks identity.

## Slide 5 - Speaker identity never crosses a split (3:25-5:00)

The official validation and testing lists define held-out speakers. Everything else forms training. Our manifest validator checks that no speaker identifier crosses these boundaries. Synthetic silence creates another leakage risk, so we also reserve non-overlapping temporal regions of each background recording for train, validation, and test. Finally, feature-normalization statistics are adapted only on training examples. These are hard guards: if a speaker or silence region overlaps, preparation fails. This protocol produces 22,246 training, 3,093 validation, and 3,081 test examples.

Handover: Vlad will now explain how we represent the audio and compare the neural models.

## Slide 6 - Log-Mel preserves time-frequency structure (5:00-5:55)

We standardize the waveform, compute a short-time Fourier transform with a 30 millisecond Hann window and a 10 millisecond stride, and project the spectrum onto forty Mel filters. With no end padding, each clip becomes a 40 by 98 time-frequency map after the logarithm and train-only normalization. This representation keeps local frequency patterns and their evolution over time. We also compute thirteen MFCCs for a lower-dimensional baseline, but the main models operate on the full log-Mel map.

Transition: we use this pipeline in a controlled matrix rather than changing several variables at once.

## Slide 7 - Five experiments isolate each choice (5:55-6:55)

E01 is an MFCC multilayer perceptron. E02 introduces a conventional CNN over log-Mel input. E03 replaces standard convolutions with depthwise-separable blocks to test efficiency. E04 changes only the augmentation recipe while keeping the same DS-CNN architecture. E05 combines convolution with a bidirectional GRU to model temporal context. The selection rule is fixed in advance: validation macro-F1, with validation accuracy only as a tie-breaker. Test labels are not part of this comparison.

Transition: these architectures differ not only in size, but also in the structure they assume.

## Slide 8 - Task structure beats parameter count (6:55-7:50)

The MLP has 361,999 parameters, yet it flattens the input and has no explicit locality. The DS-CNN uses only 46,863 parameters because depthwise convolutions learn a spatial filter per channel and pointwise convolutions mix channels. The final CRNN has 292,271 parameters. Its convolutional front end extracts local patterns, while bidirectional recurrence integrates evidence across time. This comparison demonstrates why raw parameter count is not a measure of model suitability: inductive bias matters.

Transition: to avoid optimistic reporting, the order of training, selection, repetition, and testing is explicit.

## Slide 9 - The test set stays sealed (7:50-8:50)

Each configuration is trained with Adam, early stopping, fixed seeds, and the same manifest. We select the best configuration on validation, then repeat only that configuration with seeds 7, 21, and 42. Its validation macro-F1 is 0.9394 plus or minus 0.0005, so the result is stable. The seed study does not reopen architecture selection. Once the corrected protocol is frozen, the seed-42 model receives the final clean test evaluation, followed by diagnostic corruptions and error analysis.

Transition: the validation ranking shows both the winner and a useful efficiency alternative.

## Slide 10 - The CRNN wins validation (8:50-10:00)

The augmented CRNN reaches 0.9399 validation macro-F1. The clean DS-CNN is second at 0.9324, about eight tenths of a point lower but with roughly one sixth of the parameters. The conventional CNN reaches 0.8759, and the MFCC MLP is last at 0.8390 despite having the largest parameter count. E04 is especially important: in a strict comparison where only augmentation changes, it lowers clean validation performance for the DS-CNN. We did not hide that negative result; it motivates the robustness analysis on the next slide.

Handover: INSERT FULL NAME will discuss the negative result, frozen test performance, and failure modes.

## Slide 11 - Augmentation buys noise robustness (10:00-11:05)

On clean validation data, adding the full augmentation recipe to the otherwise identical DS-CNN costs 2.29 macro-F1 points. If we stopped at the clean leaderboard, we would call it a failure. Under held-out background noise at zero decibels, however, the same augmented DS-CNN improves by 50.20 points over the clean model. The CRNN combines the best clean result with the strongest curve. This is a real trade-off: augmentation can reduce in-distribution fit while substantially improving invariance.

Transition: after the corrected protocol and validation selection were locked, we ran the final test evaluation.

## Slide 12 - Frozen test confirms validation (11:05-12:05)

The frozen CRNN reaches 93.12 percent test accuracy and 0.9315 macro-F1 on 3,081 examples. Ten-thousand stratified bootstrap resamples give a 95 percent interval from 0.9228 to 0.9402 for macro-F1. The validation-to-test change is minus 0.85 points, so the model transfers well without claiming identical performance. The normalized confusion matrix is strongly diagonal, but unknown remains visibly weaker than the command classes.

Transition: aggregate accuracy is not the whole operational story.

## Slide 13 - Noise remains the main failure axis (12:05-13:00)

At 20 decibels the test macro-F1 is still 0.9246, at 10 decibels it is 0.9106, and at zero decibels it falls to 0.8199. By contrast, shifting the waveform by plus or minus 100 milliseconds costs at most 0.38 points. Expected calibration error is 0.0303. The serialized model is 3.42 MiB, and compiled feature extraction plus CPU inference takes a median of 3.33 milliseconds on the recorded evaluation machine. These figures demonstrate interactivity, not production certification.

Transition: the qualitative audit exposes failures that calibration averages cannot summarize.

## Slide 14 - Rare confident failures survive calibration (13:00-14:05)

Unknown is the weakest class with F1 0.840. The most common directed confusion is silence predicted as up, occurring seventeen times. More importantly, the six most confident errors all have confidence of at least 0.998. Low aggregate ECE therefore does not mean the system is never overconfident. We export every selected example as a playable WAV, so during the defense we can listen to phonetic ambiguity, truncation, low energy, background interference, or possible label uncertainty instead of relying only on a matrix.

Transition: these findings lead to a deliberately qualified conclusion.

## Slide 15 - Protocol matters as much as architecture (14:05-15:00)

Our first conclusion is that preserving time-frequency structure and adding temporal recurrence gives the strongest result. Second, the 46.9-thousand-parameter DS-CNN remains an attractive option if model footprint matters more than the last accuracy point. Third, the remaining risks are severe noise, open-set unknown speech, and rare overconfidence. Every claim is reproducible from the committed manifest, five experiment runs, frozen weights, bootstrap outputs, qualitative WAV gallery, and command-line demo. Thank you; we are ready for questions.

## Rehearsal checklist

- Fill in every group member's full name before generating the final deck and package.
- Rehearse handovers at slides 5 and 10.
- Keep the live demo optional; never depend on internet access.
- Open the two WAV examples and terminal before the presentation starts.
- Aim for 4:40-4:55 per speaker to preserve a small timing buffer.
