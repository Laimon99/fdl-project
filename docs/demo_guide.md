# Offline inference demo

The demo is fully offline and uses the committed final model. Open the terminal in the repository root before the presentation.

## Setup check

```powershell
uv sync --extra dev
uv run fdl-speech doctor
```

## Demo A - correct but uncertain

```powershell
uv run fdl-speech infer artifacts/runs/e05_logmel_crnn_aug/evaluation_testing/qualitative_audio/uncertain_correct_scv1-027703.wav
```

Expected top prediction: `off`, confidence approximately `0.190`. The competing `down` probability is approximately `0.187`. This example illustrates that a correct decision can remain ambiguous.

Saved output: `artifacts/demo/correct_off_prediction.json`.

## Demo B - confident failure

```powershell
uv run fdl-speech infer artifacts/runs/e05_logmel_crnn_aug/evaluation_testing/qualitative_audio/high_confidence_error_scv1-026775.wav
```

Ground truth: `down`. Expected prediction: `no`, confidence approximately `0.9996`. This is the first qualitative example on slide 14 and demonstrates why aggregate calibration is not sufficient.

Saved output: `artifacts/demo/failure_down_as_no_prediction.json`.

## Presentation procedure

1. Play the WAV once.
2. Ask the audience which word they hear.
3. Run the command and show the top-three probabilities.
4. Connect the result to slide 14: ambiguous acoustics can produce both low-confidence successes and rare overconfident failures.

If terminal time is limited, skip the live command and play the WAVs while showing the committed JSON outputs. No network access is required.
