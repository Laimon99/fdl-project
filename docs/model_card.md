# Model card: Speech Commands keyword spotter

> This card is finalized automatically after model selection. Values marked `TBD` must not
> appear in the submitted package.

## Model summary

- Task: classify one second of mono 16 kHz audio into 12 keyword-spotting classes.
- Input: waveform converted to the feature representation locked by the selected experiment.
- Output: 12 uncalibrated logits in the label order documented in `data_card.md`.
- Framework: TensorFlow 2.21 with Keras 3.15.
- Selected experiment: **TBD after validation ranking**.
- Parameters and serialized size: **TBD**.

## Selection and evaluation

Architecture and augmentation are selected by validation macro-F1, with validation accuracy
as tie-breaker. The test split is used only for the frozen selected model. The final report
will include:

- clean accuracy and macro-F1 with stratified-bootstrap 95% confidence intervals;
- precision, recall, and F1 for all classes;
- calibration and negative log-likelihood;
- batch-1 CPU latency and serialized size;
- controlled background-noise and temporal-shift stress tests;
- qualitative confident errors and ambiguous correct predictions.

## Intended use

The model is an educational keyword spotter and reproducible research artifact. It is suitable
for comparing compact neural architectures on Speech Commands. It is not suitable for safety-
critical voice control, authentication, surveillance, speaker recognition, or unrestricted ASR.

## Ethical and technical limitations

Coverage of accents, microphones, ages, and environments is incomplete and demographic
metadata is unavailable. Predictions outside the isolated-English-word setting are not
validated. `unknown` and `silence` are protocol-defined aggregates rather than single natural
categories. Users should apply a confidence/rejection policy and collect in-domain data before
any deployment.
