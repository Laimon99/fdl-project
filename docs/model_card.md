# Model card: Speech Commands keyword spotter

## Model summary

- Task: classify one second of mono 16 kHz audio into 12 keyword-spotting classes.
- Selected experiment: `e05_logmel_crnn_aug` by validation macro-F1.
- Framework: TensorFlow 2.21 with Keras 3.15.
- Parameters: 292,271.
- Serialized Keras model: 3.42 MiB at `artifacts/models/speech_commands_best.keras`.
- Test accuracy: 0.9312; macro-F1: 0.9315.
- Test calibration ECE: 0.0303.
- Compiled end-to-end CPU latency: 3.33 ms median at batch size
  one, including feature extraction (200 timed repetitions).
- Test macro-F1 95% stratified-bootstrap CI: [0.9228, 0.9402].

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

## Known failure modes

The aggregate `unknown` class is the weakest class, short `down` clips can be confused with
`no`, and a small number of errors remain highly confident despite low aggregate ECE. Noise
at 0 dB causes a substantial degradation. Confidence must therefore not be treated as a
safety guarantee, and deployment needs an explicit rejection/abstention policy.

## Limitations

Coverage of accents, microphones, ages, and environments is incomplete; demographic metadata
is unavailable. Inputs outside isolated English words are not validated. `unknown` and
`silence` are aggregate protocol classes. Deployment would require in-domain collection,
rejection thresholds, monitoring, and a fresh risk assessment.
