# Experiment protocol

## Primary question

How much do audio representation, architecture, and realistic augmentation contribute to
speaker-independent 12-class keyword spotting under a fixed data and training protocol?

## Locked protocol

1. Use Speech Commands v0.01 and its official validation/testing file lists.
2. Treat ten words as commands; map all remaining word folders to `unknown` and sample that
   class deterministically at 10% of the wanted-word split size.
3. Synthesize the same number of `silence` examples from background-noise recordings. Lock
   temporally disjoint 80%/10%/10% regions for training, validation, and test; use the same
   boundaries for augmentation and robustness corruptions.
4. Select models using validation macro-F1, with validation accuracy as the tie-breaker.
5. Do not inspect test results to tune architecture, augmentation, or early stopping.
6. Train E01-E05 once with seed 42. Repeat the selected best configuration with seeds
   7, 21, and 42, then report mean and standard deviation.
7. Evaluate the frozen selected model only after the corrected protocol and validation
   selection are locked, on clean test data and deterministic corruptions.
8. Report 10,000-sample stratified-bootstrap 95% confidence intervals for accuracy and
   macro-F1.

## Evaluation dimensions

- Predictive quality: accuracy, balanced accuracy, macro/weighted F1, per-class precision,
  recall and F1, raw and normalized confusion matrices.
- Reliability: expected calibration error, reliability curve, negative log-likelihood.
- Robustness: background-noise SNR sweep and temporal-shift sensitivity.
- Efficiency: parameter count, serialized model size, and CPU batch-1 latency.
- Qualitative behavior: highest-confidence mistakes, lowest-confidence correct predictions,
  confusion pairs, waveforms, log-Mel plots, and playable source files.

## Leakage controls

- Speaker IDs are derived by removing `_nohash_...` from filenames.
- The manifest build fails when a speaker occurs in multiple data splits.
- Test data never enters normalization, class selection, early stopping, or threshold choices.
- Silence source regions are temporally disjoint across splits.
- Training augmentation draws noise only from the training 0–80% regions. Validation and test
  corruption sweeps use the reserved 80–90% and 90–100% regions respectively.
- Stochastic augmentations are applied to the training split only.

## Protocol revision record

The final artifacts supersede the earlier `d6bdfdf` evaluation after external review found
two protocol defects: E03/E04 differed in more than augmentation, and the augmentation noise
bank read complete background recordings. The corrections were defined without consulting
test outcomes, all affected models were retrained, model selection and validation robustness
were recomputed, and only then was the test evaluation regenerated. No architecture or
hyperparameter was chosen from either the earlier or current test metrics.
