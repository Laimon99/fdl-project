from __future__ import annotations

from pathlib import Path
from typing import Any

import librosa
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .config import load_config
from .constants import LABELS, PROJECT_ROOT
from .features import extract_features
from .utils import ProjectError, write_json


def predict_file(
    audio_path: str | Path,
    model_path: str | Path = PROJECT_ROOT / "artifacts" / "models" / "speech_commands_best.keras",
    config_path: str | Path = PROJECT_ROOT / "artifacts" / "models" / "speech_commands_best.yaml",
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    audio_path, model_path, config_path = map(Path, (audio_path, model_path, config_path))
    if not audio_path.exists():
        raise ProjectError(f"Audio file does not exist: {audio_path}")
    if not model_path.exists() or not config_path.exists():
        raise ProjectError("Final model/config not found. Run model promotion after training.")
    config = load_config(config_path)
    waveform, _ = librosa.load(audio_path, sr=config.features.sample_rate, mono=True)
    waveform = waveform[: config.features.clip_samples]
    waveform = np.pad(
        waveform,
        (0, max(0, config.features.clip_samples - len(waveform))),
    ).astype(np.float32)
    features = extract_features(tf.convert_to_tensor(waveform), config.features)[tf.newaxis]
    model = keras.models.load_model(model_path)
    logits = model(features, training=False).numpy()[0]
    probabilities = tf.nn.softmax(logits).numpy()
    ranking = np.argsort(probabilities)[::-1]
    result = {
        "audio_path": str(audio_path.resolve()),
        "predicted_label": LABELS[int(ranking[0])],
        "confidence": float(probabilities[ranking[0]]),
        "top_3": [
            {"label": LABELS[int(index)], "probability": float(probabilities[index])}
            for index in ranking[:3]
        ],
    }
    if output_json is not None:
        write_json(output_json, result)
    return result

