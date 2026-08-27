# Defense Q&A

## Data and protocol

### Why did you use Speech Commands v0.01 rather than another release?

It is the professor-proposed dataset release used by the project brief. We lock its official archive checksum and record the exact source in `data/processed/dataset_source.json`, so the experiment is reproducible. Results should not be compared numerically with v0.02 without accounting for the different release.

### How are the unknown and silence classes constructed?

Unknown examples are sampled deterministically from the twenty non-target spoken words at ten percent of the target-class count in each split. Silence examples are deterministic one-second windows from the supplied background recordings, also at ten percent. The manifest stores each source and generated example explicitly.

### How do you know there is no speaker leakage?

The official validation and testing lists assign held-out files. Speaker identifiers are parsed from filenames, and `validate_manifest` checks that every speaker belongs to exactly one split. Unit tests intentionally create overlap and require preparation to fail.

### Can silence leak even if speakers do not?

Yes. Windows cut from the same background track can be nearly identical. We partition temporal regions of every background recording by split and verify that the reserved regions do not overlap.

### Why is the modeled set approximately balanced?

Every recording of each target command is retained; target classes are not downsampled to a common count. Their source counts happen to be similar. Unknown and silence are each set to ten percent of the total target count, which is approximately the size of one target class. We still select with macro-F1 so that small residual differences cannot let a frequent class dominate. The untouched raw inventory and exact modeled counts remain available in the EDA artifacts.

## Representation and models

### Why log-Mel rather than raw waveforms?

Log-Mel maps are compact, stable to small waveform variations, and preserve local time-frequency structure that CNNs can exploit. Raw-waveform modeling would add a much larger architectural search outside the project scope. MFCCs provide a lower-dimensional control.

### Why does the MLP have more parameters but perform worse?

Flattening discards spatial locality and forces dense layers to relearn relationships between neighboring time-frequency bins. CNNs encode that prior directly, so they use parameters more effectively.

### What is a depthwise-separable convolution?

A depthwise convolution filters each input channel independently; a pointwise 1x1 convolution then mixes channels. This factorization greatly reduces parameters and multiply-accumulate operations compared with a full convolution.

### Why a bidirectional GRU?

The whole one-second clip is available before classification, so using both temporal directions is valid. The GRU can integrate phonetic evidence across time after the convolutional front end extracts local patterns. It would not be suitable for strictly streaming inference without modification.

### Is the comparison fair when the models have different parameter counts?

The goal is to compare useful design points, not claim parameter-matched causality. E03 versus E04 is the strict augmentation ablation with identical architecture. The broader leaderboard reports parameter counts explicitly and retains the accuracy-footprint trade-off in the conclusion.

## Training and selection

### Why select on macro-F1?

The task contains semantically distinct classes, including unknown and silence. Macro-F1 gives equal importance to each class and penalizes a system that performs well only on easy commands. Accuracy is reported and used only as a tie-breaker.

### Why not choose the best of the three repeated seeds?

That would introduce post-selection optimism. Seed 42 was selected and frozen before the stability repeats. Seeds 7 and 21 estimate variability; they do not replace the final model.

### How was the test set protected?

All architecture and augmentation decisions use validation metrics. `selection.json` records the frozen experiment before `finalize` computes test predictions. After external review found protocol defects unrelated to test performance, we defined the corrections, retrained every affected run, and rebuilt validation selection before regenerating the final test artifact. Neither the earlier nor current test metrics influenced architecture or hyperparameters.

### Why does augmentation reduce clean DS-CNN performance?

The combined waveform perturbations and SpecAugment make the training distribution harder. For the compact DS-CNN, the regularization cost exceeds its clean-data benefit. The noise stress test shows that the learned invariance is real even though the clean score drops.

## Evaluation

### How were confidence intervals computed?

We use 10,000 stratified bootstrap resamples of the frozen test predictions with seed 2026. Stratification preserves class composition. The interval is the percentile interval of the resampled metric distribution.

### What does ECE 0.0303 mean?

With fifteen confidence bins, the weighted average gap between predicted confidence and empirical accuracy is about three percentage points. ECE is an aggregate diagnostic and can hide rare, highly confident errors, as the qualitative gallery demonstrates.

### How is the noise robustness test generated?

We mix deterministic held-out regions of the release's own background recordings at 20, 10, and 0 dB signal-to-noise ratios. Training augmentation uses only the first 80% of every recording; validation uses 80–90% and test uses 90–100%. It is a controlled stress test, not a comprehensive simulation of real rooms or microphones.

### Why is the 0 dB result still relatively high?

The final model was trained with real background mixing and time-frequency masking. The CRNN also aggregates temporal evidence. Nevertheless, the 11.16-point macro-F1 drop is substantial and is reported as the main operational limitation.

### What exactly does the latency number include?

It includes compiled log-Mel feature extraction and the neural model for batch size one on CPU, after warm-up, over 200 timed repetitions. It excludes audio capture, disk I/O, application scheduling, and network transport.

### Why is unknown the weakest class?

Unknown is not one coherent acoustic category; it groups twenty different words and many phonetic patterns. It is therefore an open-set proxy rather than a naturally compact class.

## Scope and reproducibility

### Could this be deployed as a production speech recognizer?

No. It recognizes isolated English words from a research dataset. It does not model continuous speech, arbitrary vocabulary, all accents, microphones, rooms, or demographic subgroups. The model card states these limits explicitly.

### What work is original?

The group implementation covers deterministic manifest construction, leakage validation, TensorFlow feature extraction, five from-scratch model configurations, training orchestration, frozen selection, bootstrap and calibration evaluation, corruption tests, qualitative audio export, inference CLI, documentation, and presentation artifacts. External methods are cited.

### How can the professor reproduce the result?

Run `uv sync --extra dev`, then follow `docs/reproducibility.md` or `scripts/reproduce.ps1`. Dataset checksum, dependencies, configurations, manifests, seeds, trained Keras files, metrics, and predictions are versioned. Raw audio is downloaded from the official source.

### Does the green CI badge reproduce the full experiment?

No. CI verifies linting plus unit and synthetic-data tests without downloading the raw archive. The full data preparation, training, selection, and evaluation workflow is a separate local reproduction documented in `docs/reproducibility.md`. We distinguish these two levels explicitly rather than presenting CI as end-to-end experimental coverage.

### What would you try next?

Three priorities are streaming-compatible causal models, a broader external-noise and device evaluation, and explicit open-set calibration for unknown speech. A parameter-matched CRNN versus DS-CNN study would sharpen the architecture comparison.
